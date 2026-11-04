#include <iostream>
#include <vector>
#include <chrono>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

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

    // 2. Allocate VRAM Buffers in GPU Default Heap
    size_t inBytes = (size_t)WIDTH * HEIGHT * IN_CHANNELS * sizeof(float);
    size_t outBytes = (size_t)WIDTH * HEIGHT * OUT_CHANNELS * sizeof(float);

    std::cout << "Allocating VRAM Input Buffer:  " << inBytes / (1024 * 1024) << " MB...\n";
    ComPtr<ID3D12Resource> inBuffer = CreateD3D12Buffer(device.Get(), inBytes);

    std::cout << "Allocating VRAM Output Buffer: " << outBytes / (1024 * 1024) << " MB...\n";
    ComPtr<ID3D12Resource> outBuffer = CreateD3D12Buffer(device.Get(), outBytes);

    if (!inBuffer || !outBuffer) return -1;

    // 3. Initialize ONNX Runtime with DirectML
    const OrtApi& ortApi = Ort::GetApi();
    const OrtDmlApi* ortDmlApi = nullptr;
    ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi));

    if (!ortDmlApi) {
        std::cerr << "Error: DirectML execution provider API unavailable.\n";
        return -1;
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DirectML_Benchmark");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(4);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    OrtSessionOptionsAppendExecutionProvider_DML(sessionOptions, 0);

    const wchar_t* modelPath = L"neural_shader_gtx1650_sharp.onnx";
    std::cout << "Loading ONNX model graph into DirectML...\n";
    Ort::Session session(nullptr);
    try {
        session = Ort::Session(env, modelPath, sessionOptions);
    } catch (const Ort::Exception& e) {
        std::cerr << "Model load error: " << e.what() << "\n";
        return -1;
    }
    std::cout << "Model loaded successfully!\n\n";

    // 4. Wrap D3D12 VRAM Resources into DirectML OrtValue Tensors (Zero-Copy)
    void* dmlInAlloc = nullptr;
    void* dmlOutAlloc = nullptr;
    ortDmlApi->CreateGPUAllocationFromD3DResource(inBuffer.Get(), &dmlInAlloc);
    ortDmlApi->CreateGPUAllocationFromD3DResource(outBuffer.Get(), &dmlOutAlloc);

    OrtMemoryInfo* dmlMemInfo = nullptr;
    ortApi.CreateMemoryInfo("DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault, &dmlMemInfo);

    int64_t inShape[] = { 1, IN_CHANNELS, HEIGHT, WIDTH };
    OrtValue* rawInVal = nullptr;
    ortApi.CreateTensorWithDataAsOrtValue(
        dmlMemInfo, dmlInAlloc, inBytes, inShape, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &rawInVal
    );

    int64_t outShape[] = { 1, OUT_CHANNELS, HEIGHT, WIDTH };
    OrtValue* rawOutVal = nullptr;
    ortApi.CreateTensorWithDataAsOrtValue(
        dmlMemInfo, dmlOutAlloc, outBytes, outShape, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &rawOutVal
    );

    Ort::Value inTensor(rawInVal);
    Ort::Value outTensor(rawOutVal);

    // 5. Setup IOBinding (Locks tensors to DirectML GPU queue)
    Ort::IoBinding ioBinding(session);
    ioBinding.BindInput("g_buffer", inTensor);
    ioBinding.BindOutput("rendered_frame", outTensor);

    Ort::RunOptions runOptions;

    // 6. Warmup Run (Compiles the DirectML kernel execution plans on GPU)
    std::cout << "Warming up DirectML pipeline on GPU...\n";
    try {
        session.Run(runOptions, ioBinding);
    } catch (const Ort::Exception& e) {
        std::cerr << "Warmup error: " << e.what() << "\n";
        return -1;
    }
    std::cout << "Warmup successful!\n\n";

    // 7. Benchmark 50 Sequential Executions
    std::cout << "Benchmarking 50 frames of GPU-to-GPU inference...\n";
    auto start = std::chrono::high_resolution_clock::now();

    const int iterations = 50;
    for (int i = 0; i < iterations; ++i) {
        session.Run(runOptions, ioBinding);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(end - start).count();
    double avgMs = totalMs / iterations;
    double fps = 1000.0 / avgMs;

    std::cout << "\n================ Benchmark Results ================\n";
    std::cout << "Average Execution Time: " << avgMs << " ms per frame\n";
    std::cout << "Maximum Potential FPS:  " << fps << " FPS\n";
    std::cout << "====================================================\n\n";

    // Cleanup allocations
    ortApi.ReleaseMemoryInfo(dmlMemInfo);
    ortDmlApi->FreeGPUAllocation(dmlInAlloc);
    ortDmlApi->FreeGPUAllocation(dmlOutAlloc);

    std::cout << "Press Enter to exit...";
    std::cin.get();
    return 0;
}
