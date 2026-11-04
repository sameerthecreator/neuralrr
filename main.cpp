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
#include <thread>
#include <fstream>
#include <string>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

using Microsoft::WRL::ComPtr;

// 432p Target (768x432: 32.7 ms per pass on GTX 1650)
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

struct NeuralContext {
    std::atomic<bool> is_ready{false};
    std::atomic<bool> is_loading{false};
    std::atomic<bool> has_failed{false};
    std::atomic<uint64_t> total_frames{0};
    std::atomic<uint64_t> ai_frames{0};

    ComPtr<ID3D12Device> d3d12Device;
    ComPtr<ID3D12CommandQueue> cmdQueue;

    HMODULE hDml = nullptr;
    ComPtr<IDMLDevice> dmlDevice;

    ComPtr<ID3D12Resource> inBuffer;
    ComPtr<ID3D12Resource> outBuffer;

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

    std::wstring modelDirectory;
    std::ofstream logFile;
};

static NeuralContext g_ctx;
static std::atomic<bool> g_isInferring{false};

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

static void AsyncInitDirectML() {
    if (g_ctx.logFile.is_open()) {
        g_ctx.logFile << "[NeuralShader] Background initialization thread started (432p: 768x432).\n";
        g_ctx.logFile.flush();
    }

    g_ctx.hDml = LoadLibraryA("DirectML.dll");
    if (!g_ctx.hDml) {
        if (g_ctx.logFile.is_open()) {
            g_ctx.logFile << "[NeuralShader] ERROR: Failed to load DirectML.dll\n";
            g_ctx.logFile.flush();
        }
        g_ctx.has_failed = true;
        return;
    }

    auto pfnDMLCreateDevice = reinterpret_cast<PFN_DMLCreateDevice>(GetProcAddress(g_ctx.hDml, "DMLCreateDevice"));
    if (!pfnDMLCreateDevice || FAILED(pfnDMLCreateDevice(g_ctx.d3d12Device.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&g_ctx.dmlDevice)))) {
        if (g_ctx.logFile.is_open()) {
            g_ctx.logFile << "[NeuralShader] ERROR: Failed to create DirectML device.\n";
            g_ctx.logFile.flush();
        }
        g_ctx.has_failed = true;
        return;
    }

    size_t inBytes  = (size_t)IN_WIDTH * IN_HEIGHT * IN_CHANNELS * sizeof(float);
    size_t outBytes = (size_t)IN_WIDTH * IN_HEIGHT * OUT_CHANNELS * sizeof(float);

    g_ctx.inBuffer  = CreateLinearBuffer(g_ctx.d3d12Device.Get(), inBytes);
    g_ctx.outBuffer = CreateLinearBuffer(g_ctx.d3d12Device.Get(), outBytes);
    if (!g_ctx.inBuffer || !g_ctx.outBuffer) {
        if (g_ctx.logFile.is_open()) {
            g_ctx.logFile << "[NeuralShader] ERROR: Failed to allocate GPU VRAM linear buffers.\n";
            g_ctx.logFile.flush();
        }
        g_ctx.has_failed = true;
        return;
    }

    const OrtApiBase* apiBase = OrtGetApiBase();
    g_ctx.ort = apiBase->GetApi(17);
    if (!g_ctx.ort) {
        g_ctx.has_failed = true;
        return;
    }

    g_ctx.ort->GetExecutionProviderApi("DML", 17, reinterpret_cast<const void**>(&g_ctx.ortDml));
    if (!g_ctx.ortDml) {
        g_ctx.has_failed = true;
        return;
    }

    g_ctx.ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "Azimuth_NeuralShader", &g_ctx.env);
    g_ctx.ort->CreateSessionOptions(&g_ctx.sessionOptions);
    g_ctx.ort->SetIntraOpNumThreads(g_ctx.sessionOptions, 4);
    g_ctx.ort->SetSessionGraphOptimizationLevel(g_ctx.sessionOptions, ORT_ENABLE_EXTENDED);

    SetEnvironmentVariableA("DML_DISABLE_METACMDS", "1");
    g_ctx.ortDml->SessionOptionsAppendExecutionProvider_DML1(g_ctx.sessionOptions, g_ctx.dmlDevice.Get(), g_ctx.cmdQueue.Get());

    std::wstring fullModelPath = g_ctx.modelDirectory + L"\\neural_shader_gtx1650_432p.onnx";
    OrtStatus* status = g_ctx.ort->CreateSession(g_ctx.env, fullModelPath.c_str(), g_ctx.sessionOptions, &g_ctx.session);
    if (status != nullptr) {
        if (g_ctx.logFile.is_open()) {
            g_ctx.logFile << "[NeuralShader] ERROR: Failed to create session for model: " << g_ctx.ort->GetErrorMessage(status) << "\n";
            g_ctx.logFile.flush();
        }
        g_ctx.ort->ReleaseStatus(status);
        g_ctx.has_failed = true;
        return;
    }

    g_ctx.ortDml->CreateGPUAllocationFromD3DResource(g_ctx.inBuffer.Get(), &g_ctx.dmlInAlloc);
    g_ctx.ortDml->CreateGPUAllocationFromD3DResource(g_ctx.outBuffer.Get(), &g_ctx.dmlOutAlloc);

    g_ctx.ort->CreateMemoryInfo("DML", OrtDeviceAllocator, 0, OrtMemTypeDefault, &g_ctx.memInfo);

    int64_t inShape[]  = { 1, IN_CHANNELS, IN_HEIGHT, IN_WIDTH };
    int64_t outShape[] = { 1, OUT_CHANNELS, IN_HEIGHT, IN_WIDTH };

    g_ctx.ort->CreateTensorWithDataAsOrtValue(g_ctx.memInfo, g_ctx.dmlInAlloc, inBytes, inShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &g_ctx.inTensor);
    g_ctx.ort->CreateTensorWithDataAsOrtValue(g_ctx.memInfo, g_ctx.dmlOutAlloc, outBytes, outShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &g_ctx.outTensor);

    g_ctx.ort->CreateIoBinding(g_ctx.session, &g_ctx.ioBinding);
    g_ctx.ort->BindInput(g_ctx.ioBinding, "g_buffer", g_ctx.inTensor);
    g_ctx.ort->BindOutput(g_ctx.ioBinding, "rendered_frame", g_ctx.outTensor);
    g_ctx.ort->CreateRunOptions(&g_ctx.runOptions);

    if (g_ctx.logFile.is_open()) {
        g_ctx.logFile << "[NeuralShader] DirectML 432p session initialized and ready for real-time inference!\n";
        g_ctx.logFile.flush();
    }

    g_ctx.is_ready = true;
}

static void on_present(reshade::api::command_queue* queue, reshade::api::swapchain* swapchain, const reshade::api::rect*, const reshade::api::rect*, uint32_t, const reshade::api::rect*) {
    uint64_t frame = ++g_ctx.total_frames;

    if (swapchain->get_device()->get_api() != reshade::api::device_api::d3d12) return;
    if (g_ctx.has_failed) return;

    // Start background initialization after passing frame 30
    if (!g_ctx.is_ready && !g_ctx.is_loading && frame > 30) {
        g_ctx.is_loading = true;
        g_ctx.d3d12Device = reinterpret_cast<ID3D12Device*>(swapchain->get_device()->get_native());
        g_ctx.cmdQueue = reinterpret_cast<ID3D12CommandQueue*>(queue->get_native());

        std::thread initThread(AsyncInitDirectML);
        initThread.detach();
    }

    // Paced execution: run only on alternate frames and guard against queue saturation
    if (g_ctx.is_ready && (frame % 2 == 0)) {
        bool expected = false;
        if (g_isInferring.compare_exchange_strong(expected, true)) {
            OrtStatus* status = g_ctx.ort->RunWithBinding(g_ctx.session, g_ctx.runOptions, g_ctx.ioBinding);
            if (status != nullptr) {
                if (g_ctx.logFile.is_open()) {
                    g_ctx.logFile << "[NeuralShader] Error during RunWithBinding: " << g_ctx.ort->GetErrorMessage(status) << "\n";
                    g_ctx.logFile.flush();
                }
                g_ctx.ort->ReleaseStatus(status);
            } else {
                uint64_t aiCount = ++g_ctx.ai_frames;
                if (aiCount % 60 == 0 && g_ctx.logFile.is_open()) {
                    g_ctx.logFile << "[NeuralShader] Inferred AI frames: " << aiCount << " (Game frame: " << frame << ")\n";
                    g_ctx.logFile.flush();
                }
            }
            g_isInferring.store(false);
        }
    }
}

extern "C" __declspec(dllexport) const char* NAME = "Neural Shader (432p DirectML)";
extern "C" __declspec(dllexport) const char* DESCRIPTION = "Paced DirectML Zero-Copy Addon for Forza Horizon 2 (768x432)";

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID) {
    switch (fdwReason) {
    case DLL_PROCESS_ATTACH: {
        wchar_t modulePath[MAX_PATH];
        GetModuleFileNameW(hinstDLL, modulePath, MAX_PATH);
        std::wstring pathStr(modulePath);
        size_t lastSlash = pathStr.find_last_of(L"\\/");
        if (lastSlash != std::wstring::npos) {
            g_ctx.modelDirectory = pathStr.substr(0, lastSlash);
        }

        g_ctx.logFile.open("neural_shader_debug.txt", std::ios::out);
        if (g_ctx.logFile.is_open()) {
            g_ctx.logFile << "[NeuralShader] Addon attached (432p target).\n";
            g_ctx.logFile.flush();
        }

        if (!reshade::register_addon(hinstDLL)) return FALSE;
        reshade::register_event<reshade::addon_event::present>(on_present);
        break;
    }
    case DLL_PROCESS_DETACH:
        reshade::unregister_event<reshade::addon_event::present>(on_present);
        reshade::unregister_addon(hinstDLL);

        if (g_ctx.is_ready) {
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
        }

        if (g_ctx.logFile.is_open()) {
            g_ctx.logFile << "[NeuralShader] Addon safely detached.\n";
            g_ctx.logFile.close();
        }
        break;
    }
    return TRUE;
}
