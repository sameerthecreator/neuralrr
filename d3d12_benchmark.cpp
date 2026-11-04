#include <iostream>
#include <vector>
#include <chrono>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <DirectML.h>
#include <onnxruntime_c_api.h>
#include <dml_provider_factory.h>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "DirectML.lib")

using Microsoft::WRL::ComPtr;

constexpr int WIDTH = 1920;
constexpr int HEIGHT = 1080;
constexpr int IN_CHANNELS = 7;
constexpr int OUT_CHANNELS = 3;

ComPtr<ID3D12Resource> CreateD3D12Buffer(ID3D12Device* device, size_t sizeBytes) {
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

    ComPtr<ID3D12Resource> resource;
    HRESULT hr = device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&resource)
    );

    if (FAILED(hr)) {
        std::cerr << "Failed to allocate D3D12 buffer: " << sizeBytes << " bytes\n";
        return nullptr;
    }
    return resource;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << " DirectML D3D12 Zero-Copy Inference Benchmark\n";
    std::cout << " Model   : neural_shader_gtx1650_sharp.onnx\n";
    std::cout << " Shape   : [1, 7, 1080, 1920] -> [1, 3, 1080, 1920]\n";
    std::cout << " Hardware: Native GPU VRAM (Zero Host Copying)\n";
    std::cout << "========================================================\n\n";

    // 1. Initialize DXGI and D3D12 on discrete GPU (GTX 1650)
    ComPtr<IDXGIFactory4> factory;
    CreateDXGIFactory1(IID_PPV_ARGS(&factory));

    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;

        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
            std::wcout << L"Selected GPU: " << desc.Description << L"\n";
            std::cout << "Dedicated VRAM: " << desc.DedicatedVideoMemory / (1024 * 1024) << " MB\n\n";
            break;
        }
    }

    if (!device) {
        std::cerr << "Error: Could not create D3D12 device.\n";
        return -1;
    }

    // 2. Create D3D12 Command Queue & Command List
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

    ComPtr<ID3D12CommandQueue> cmdQueue;
    device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&cmdQueue));

    ComPtr<ID3D12CommandAllocator> cmdAlloc;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&cmdAlloc));

    ComPtr<ID3D12GraphicsCommandList> cmdList;
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, cmdAlloc.Get(), nullptr, IID_PPV_ARGS(&cmdList));

    // 3. Create DirectML Device on top of D3D12 Device
    ComPtr<IDMLDevice> dmlDevice;
    HRESULT hr = DMLCreateDevice(device.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&dmlDevice));
    if (FAILED(hr)) {
        std::cerr << "Error: DMLCreateDevice failed (HRESULT: 0x" << std::hex << hr << ")\n";
        return -1;
    }

    // 4. Allocate VRAM Buffers in GPU Default Heap
    size_t inBytes = (size_t)WIDTH * HEIGHT * IN_CHANNELS * sizeof(float);
    size_t outBytes = (size_t)WIDTH * HEIGHT * OUT_CHANNELS * sizeof(float);

    std::cout << "Allocating VRAM Input Buffer:  " << inBytes / (1024 * 1024) << " MB...\n";
    ComPtr<ID3D12Resource> inBuffer = CreateD3D12Buffer(device.Get(), inBytes);

    std::cout << "Allocating VRAM Output Buffer: " << outBytes / (1024 * 1024) << " MB...\n";
    ComPtr<ID3D12Resource> outBuffer = CreateD3D12Buffer(device.Get(), outBytes);

    if (!inBuffer || !outBuffer) return -1;

    // Transition buffers to UNORDERED_ACCESS state
    D3D12_RESOURCE_BARRIER barriers[2] = {};
    barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[0].Transition.pResource = inBuffer.Get();
    barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[1].Transition.pResource = outBuffer.Get();
    barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    cmdList->ResourceBarrier(2, barriers);
    cmdList->Close();

    ID3D12CommandList* ppCommandLists[] = { cmdList.Get() };
    cmdQueue->ExecuteCommandLists(1, ppCommandLists);

    // Synchronize queue
    ComPtr<ID3D12Fence> fence;
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    cmdQueue->Signal(fence.Get(), 1);
    HANDLE fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    fence->SetEventOnCompletion(1, fenceEvent);
    WaitForSingleObject(fenceEvent, INFINITE);
    CloseHandle(fenceEvent);

    // 5. Initialize ORT Session with native IDMLDevice and D3D12 Command Queue
    const OrtApiBase* apiBase = OrtGetApiBase();
    const OrtApi* ort = apiBase->GetApi(17);
    if (!ort) {
        std::cerr << "Error: Could not obtain OrtApi version 17.\n";
        return -1;
    }

    const OrtDmlApi* ortDmlApi = nullptr;
    ort->GetExecutionProviderApi("DML", 17, reinterpret_cast<const void**>(&ortDmlApi));
    if (!ortDmlApi) {
        std::cerr << "Error: DirectML execution provider unavailable.\n";
        return -1;
    }

    OrtEnv* env = nullptr;
    ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "DirectML_Benchmark", &env);

    OrtSessionOptions* sessionOptions = nullptr;
    ort->CreateSessionOptions(&sessionOptions);
    ort->SetIntraOpNumThreads(sessionOptions, 4);
    ort->SetSessionGraphOptimizationLevel(sessionOptions, ORT_ENABLE_EXTENDED);

    // Pass valid IDMLDevice and ID3D12CommandQueue
    ortDmlApi->SessionOptionsAppendExecutionProvider_DML1(sessionOptions, dmlDevice.Get(), cmdQueue.Get());

    const wchar_t* modelPath = L"neural_shader_gtx1650_sharp.onnx";
    std::cout << "Loading ONNX model graph into DirectML...\n";
    OrtSession* session = nullptr;
    OrtStatus* status = ort->CreateSession(env, modelPath, sessionOptions, &session);
    if (status != nullptr) {
        std::cerr << "Model load error: " << ort->GetErrorMessage(status) << "\n";
        ort->ReleaseStatus(status);
        return -1;
    }
    std::cout << "Model loaded successfully!\n\n";

    // 6. Wrap D3D12 VRAM Resources into DirectML Tensors (Zero-Copy)
    void* dmlInAlloc = nullptr;
    void* dmlOutAlloc = nullptr;
    ortDmlApi->CreateGPUAllocationFromD3DResource(inBuffer.Get(), &dmlInAlloc);
    ortDmlApi->CreateGPUAllocationFromD3DResource(outBuffer.Get(), &dmlOutAlloc);

    OrtMemoryInfo* dmlMemInfo = nullptr;
    ort->CreateMemoryInfo("DML", OrtDeviceAllocator, 0, OrtMemTypeDefault, &dmlMemInfo);

    int64_t inShape[] = { 1, IN_CHANNELS, HEIGHT, WIDTH };
    OrtValue* inTensor = nullptr;
    ort->CreateTensorWithDataAsOrtValue(
        dmlMemInfo, dmlInAlloc, inBytes, inShape, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inTensor
    );

    int64_t outShape[] = { 1, OUT_CHANNELS, HEIGHT, WIDTH };
    OrtValue* outTensor = nullptr;
    ort->CreateTensorWithDataAsOrtValue(
        dmlMemInfo, dmlOutAlloc, outBytes, outShape, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &outTensor
    );

    // 7. Setup IOBinding
    OrtIoBinding* ioBinding = nullptr;
    ort->CreateIoBinding(session, &ioBinding);
    ort->BindInput(ioBinding, "g_buffer", inTensor);
    ort->BindOutput(ioBinding, "rendered_frame", outTensor);

    OrtRunOptions* runOptions = nullptr;
    ort->CreateRunOptions(&runOptions);

    // 8. Warmup Run
    std::cout << "Warming up DirectML pipeline on GPU...\n";
    status = ort->RunWithBinding(session, runOptions, ioBinding);
    if (status != nullptr) {
        std::cerr << "Warmup error: " << ort->GetErrorMessage(status) << "\n";
        ort->ReleaseStatus(status);
        return -1;
    }
    std::cout << "Warmup successful!\n\n";

    // 9. Benchmark 50 Sequential Executions
    std::cout << "Benchmarking 50 frames of GPU-to-GPU inference...\n";
    auto start = std::chrono::high_resolution_clock::now();

    const int iterations = 50;
    for (int i = 0; i < iterations; ++i) {
        ort->RunWithBinding(session, runOptions, ioBinding);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(end - start).count();
    double avgMs = totalMs / iterations;
    double fps = 1000.0 / avgMs;

    std::cout << "\n================ Benchmark Results ================\n";
    std::cout << "Average Execution Time: " << avgMs << " ms per frame\n";
    std::cout << "Maximum Potential FPS:  " << fps << " FPS\n";
    std::cout << "====================================================\n\n";

    // Cleanup
    ort->ReleaseIoBinding(ioBinding);
    ort->ReleaseRunOptions(runOptions);
    ort->ReleaseValue(inTensor);
    ort->ReleaseValue(outTensor);
    ort->ReleaseMemoryInfo(dmlMemInfo);
    ort->ReleaseSession(session);
    ort->ReleaseSessionOptions(sessionOptions);
    ort->ReleaseEnv(env);
    ortDmlApi->FreeGPUAllocation(dmlInAlloc);
    ortDmlApi->FreeGPUAllocation(dmlOutAlloc);

    return 0;
}
