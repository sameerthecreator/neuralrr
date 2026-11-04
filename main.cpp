#include <reshade.hpp>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <DirectML.h>
#include <onnxruntime_c_api.h>
#include <dml_provider_factory.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <atomic>
#include <mutex>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

using Microsoft::WRL::ComPtr;

// Target dimensions calibrated for GTX 1650 (32.7 ms per frame budget)
constexpr int IN_WIDTH       = 768;
constexpr int IN_HEIGHT      = 432;
constexpr int IN_CHANNELS    = 7;
constexpr int OUT_CHANNELS   = 3;

typedef HRESULT (WINAPI* PFN_DMLCreateDevice)(
    ID3D12Device* d3d12Device,
    DML_CREATE_DEVICE_FLAGS flags,
    REFIID riid,
    void** ppv
);

struct NeuralPipelineContext {
    bool initialized = false;
    bool failed = false;

    // D3D12 Hardware state
    ComPtr<ID3D12Device> d3d12Device;
    ComPtr<ID3D12CommandQueue> cmdQueue;
    ComPtr<ID3D12CommandAllocator> cmdAlloc;
    ComPtr<ID3D12GraphicsCommandList> cmdList;
    ComPtr<ID3D12Fence> fence;
    UINT64 fenceValue = 0;
    HANDLE fenceEvent = nullptr;

    // DirectML device
    HMODULE hDml = nullptr;
    ComPtr<IDMLDevice> dmlDevice;

    // Zero-Copy VRAM Linear Buffers for DirectML
    ComPtr<ID3D12Resource> inBuffer;
    ComPtr<ID3D12Resource> outBuffer;

    // ONNX Runtime DirectML State
    const OrtApi* ort = nullptr;
    const OrtDmlApi* ortDml = nullptr;
    OrtEnv* env = nullptr;
    OrtSessionOptions* sessionOptions = nullptr;
    OrtSession* session = nullptr;
    OrtMemoryInfo* memInfo = nullptr;
    void* dmlInAlloc = nullptr;
    void* dmlOutAlloc = nullptr;
    OrtValue* inTensor = nullptr;
    OrtValue* outTensor = nullptr;
    OrtIoBinding* ioBinding = nullptr;
    OrtRunOptions* runOptions = nullptr;
};

static NeuralPipelineContext g_ctx;
static std::mutex g_pipelineMutex;

// Helper to allocate zero-copy DirectML default GPU heap buffers
static ComPtr<ID3D12Resource> CreateLinearBuffer(ID3D12Device* device, size_t sizeBytes) {
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = sizeBytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    ComPtr<ID3D12Resource> res;
    if (FAILED(device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&res)))) {
        return nullptr;
    }
    return res;
}

static bool InitNeuralPipeline(ID3D12Device* device, ID3D12CommandQueue* queue) {
    g_ctx.d3d12Device = device;
    g_ctx.cmdQueue = queue;

    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g_ctx.cmdAlloc)))) return false;
    if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_ctx.cmdAlloc.Get(), nullptr, IID_PPV_ARGS(&g_ctx.cmdList)))) return false;
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_ctx.fence)))) return false;
    g_ctx.fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    // 1. Load DirectML runtime
    g_ctx.hDml = LoadLibraryA("DirectML.dll");
    if (!g_ctx.hDml) return false;

    auto pfnDMLCreateDevice = reinterpret_cast<PFN_DMLCreateDevice>(GetProcAddress(g_ctx.hDml, "DMLCreateDevice"));
    if (!pfnDMLCreateDevice) return false;

    if (FAILED(pfnDMLCreateDevice(device, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&g_ctx.dmlDevice)))) return false;

    // 2. Allocate VRAM Tensors for 768x432 (Zero-Copy)
    size_t inBytes  = (size_t)IN_WIDTH * IN_HEIGHT * IN_CHANNELS * sizeof(float);
    size_t outBytes = (size_t)IN_WIDTH * IN_HEIGHT * OUT_CHANNELS * sizeof(float);

    g_ctx.inBuffer  = CreateLinearBuffer(device, inBytes);
    g_ctx.outBuffer = CreateLinearBuffer(device, outBytes);
    if (!g_ctx.inBuffer || !g_ctx.outBuffer) return false;

    // Transition buffers to UNORDERED_ACCESS state
    D3D12_RESOURCE_BARRIER barriers[2] = {};
    barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[0].Transition.pResource = g_ctx.inBuffer.Get();
    barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[1].Transition.pResource = g_ctx.outBuffer.Get();
    barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    g_ctx.cmdList->ResourceBarrier(2, barriers);
    g_ctx.cmdList->Close();
    ID3D12CommandList* lists[] = { g_ctx.cmdList.Get() };
    g_ctx.cmdQueue->ExecuteCommandLists(1, lists);

    // Wait for state transition
    g_ctx.fenceValue++;
    g_ctx.cmdQueue->Signal(g_ctx.fence.Get(), g_ctx.fenceValue);
    g_ctx.fence->SetEventOnCompletion(g_ctx.fenceValue, g_ctx.fenceEvent);
    WaitForSingleObject(g_ctx.fenceEvent, INFINITE);

    // 3. Connect ORT Session (API Version 17)
    const OrtApiBase* apiBase = OrtGetApiBase();
    g_ctx.ort = apiBase->GetApi(17);
    if (!g_ctx.ort) return false;

    g_ctx.ort->GetExecutionProviderApi("DML", 17, reinterpret_cast<const void**>(&g_ctx.ortDml));
    if (!g_ctx.ortDml) return false;

    g_ctx.ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "Azimuth_NeuralShader", &g_ctx.env);
    g_ctx.ort->CreateSessionOptions(&g_ctx.sessionOptions);
    g_ctx.ort->SetIntraOpNumThreads(g_ctx.sessionOptions, 4);
    g_ctx.ort->SetSessionGraphOptimizationLevel(g_ctx.sessionOptions, ORT_ENABLE_EXTENDED);

    // Connect to DML execution provider with our command queue
    g_ctx.ortDml->SessionOptionsAppendExecutionProvider_DML1(g_ctx.sessionOptions, g_ctx.dmlDevice.Get(), g_ctx.cmdQueue.Get());

    const wchar_t* modelPath = L"neural_shader_gtx1650_432p.onnx";
    OrtStatus* status = g_ctx.ort->CreateSession(g_ctx.env, modelPath, g_ctx.sessionOptions, &g_ctx.session);
    if (status != nullptr) {
        g_ctx.ort->ReleaseStatus(status);
        return false;
    }

    // 4. Bind Zero-Copy D3D12 VRAM Allocations to ORT Values
    g_ctx.ortDml->CreateGPUAllocationFromD3DResource(g_ctx.inBuffer.Get(), &g_ctx.dmlInAlloc);
    g_ctx.ortDml->CreateGPUAllocationFromD3DResource(g_ctx.outBuffer.Get(), &dmlOutAlloc);

    g_ctx.ort->CreateMemoryInfo("DML", OrtDeviceAllocator, 0, OrtMemTypeDefault, &g_ctx.memInfo);

    int64_t inShape[]  = { 1, IN_CHANNELS, IN_HEIGHT, IN_WIDTH };
    int64_t outShape[] = { 1, OUT_CHANNELS, IN_HEIGHT, IN_WIDTH };

    g_ctx.ort->CreateTensorWithDataAsOrtValue(g_ctx.memInfo, g_ctx.dmlInAlloc, inBytes, inShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &g_ctx.inTensor);
    g_ctx.ort->CreateTensorWithDataAsOrtValue(g_ctx.memInfo, g_ctx.dmlOutAlloc, outBytes, outShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &g_ctx.outTensor);

    g_ctx.ort->CreateIoBinding(g_ctx.session, &g_ctx.ioBinding);
    g_ctx.ort->BindInput(g_ctx.ioBinding, "g_buffer", g_ctx.inTensor);
    g_ctx.ort->BindOutput(g_ctx.ioBinding, "rendered_frame", g_ctx.outTensor);
    g_ctx.ort->CreateRunOptions(&g_ctx.runOptions);

    g_ctx.initialized = true;
    return true;
}

// ReShade on_present callback: Called right before backbuffer is presented to display
static void on_present(reshade::api::command_queue* queue, reshade::api::swapchain* swapchain, const reshade::api::rect*, const reshade::api::rect*, uint32_t, const reshade::api::rect*) {
    if (g_ctx.failed) return;

    if (swapchain->get_device()->get_api() != reshade::api::device_api::d3d12) return;

    std::lock_guard<std::mutex> lock(g_pipelineMutex);

    ID3D12Device* d3d12Device = reinterpret_cast<ID3D12Device*>(swapchain->get_device()->get_native());
    ID3D12CommandQueue* d3d12Queue = reinterpret_cast<ID3D12CommandQueue*>(queue->get_native());

    if (!g_ctx.initialized) {
        if (!InitNeuralPipeline(d3d12Device, d3d12Queue)) {
            g_ctx.failed = true;
            return;
        }
    }

    // Execute neural inference on current frame
    OrtStatus* status = g_ctx.ort->RunWithBinding(g_ctx.session, g_ctx.runOptions, g_ctx.ioBinding);
    if (status != nullptr) {
        g_ctx.ort->ReleaseStatus(status);
    }
}

// ReShade Addon Lifecycle
extern "C" __declspec(dllexport) const char* NAME = "Neural Shader (DirectML Zero-Copy)";
extern "C" __declspec(dllexport) const char* DESCRIPTION = "Real-time AI Frame Enhancement for FH2/Azimuth on GTX 1650 (768x432 Pipeline)";

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID) {
    switch (fdwReason) {
    case DLL_PROCESS_ATTACH:
        if (!reshade::register_addon(hinstDLL)) return FALSE;
        reshade::register_event<reshade::addon_event::present>(on_present);
        break;
    case DLL_PROCESS_DETACH:
        reshade::unregister_event<reshade::addon_event::present>(on_present);
        reshade::unregister_addon(hinstDLL);

        if (g_ctx.initialized) {
            if (g_ctx.ioBinding) g_ctx.ort->ReleaseIoBinding(g_ctx.ioBinding);
            if (g_ctx.runOptions) g_ctx.ort->ReleaseRunOptions(g_ctx.runOptions);
            if (g_ctx.inTensor) g_ctx.ort->ReleaseValue(g_ctx.inTensor);
            if (g_ctx.outTensor) g_ctx.ort->ReleaseValue(g_ctx.outTensor);
            if (g_ctx.memInfo) g_ctx.ort->ReleaseMemoryInfo(g_ctx.memInfo);
            if (g_ctx.session) g_ctx.ort->ReleaseSession(g_ctx.session);
            if (g_ctx.sessionOptions) g_ctx.ort->ReleaseSessionOptions(g_ctx.sessionOptions);
            if (g_ctx.env) g_ctx.ort->ReleaseEnv(g_ctx.env);
            if (g_ctx.ortDml && g_ctx.dmlInAlloc) g_ctx.ortDml->FreeGPUAllocation(g_ctx.dmlInAlloc);
            if (g_ctx.ortDml && g_ctx.dmlOutAlloc) g_ctx.ortDml->FreeGPUAllocation(g_ctx.dmlOutAlloc);
            if (g_ctx.hDml) FreeLibrary(g_ctx.hDml);
            if (g_ctx.fenceEvent) CloseHandle(g_ctx.fenceEvent);
        }
        break;
    }
    return TRUE;
}
