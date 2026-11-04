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
    ComPtr<ID3D12CommandQueue> dedicatedDmlQueue;
    HMODULE hDml = nullptr;
    ComPtr<IDMLDevice> dmlDevice;

    const OrtApi* ort = nullptr;
    const OrtDmlApi* ortDml = nullptr;
    OrtEnv* env = nullptr;
    OrtSessionOptions* sessionOptions = nullptr;
    OrtSession* session = nullptr;
    OrtMemoryInfo* memInfo = nullptr;
    OrtValue* inTensor = nullptr;
    OrtValue* outTensor = nullptr;

    const char* inputNames[1]  = { "g_buffer" };
    const char* outputNames[1] = { "rendered_frame" };

    std::vector<float> inputData;
    std::vector<float> outputData;

    std::wstring modelDirectory;
    std::ofstream logFile;
};

static NeuralContext g_ctx;
static std::atomic<bool> g_isInferring{false};

static void AsyncInitDirectML() {
    if (g_ctx.logFile.is_open()) {
        g_ctx.logFile << "[NeuralShader] Background thread initializing on GTX 1650...\n";
        g_ctx.logFile.flush();
    }

    // 1. Create a dedicated Direct D3D12 Command Queue on the GTX 1650 for DirectML
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type     = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    queueDesc.Flags    = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.NodeMask = 0;

    HRESULT hr = g_ctx.d3d12Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&g_ctx.dedicatedDmlQueue));
    if (FAILED(hr)) {
        if (g_ctx.logFile.is_open()) g_ctx.logFile << "[NeuralShader] ERROR: Failed to create dedicated D3D12 queue for DirectML.\n";
        g_ctx.has_failed = true;
        return;
    }

    // 2. Load DirectML runtime
    g_ctx.hDml = LoadLibraryA("DirectML.dll");
    if (!g_ctx.hDml) {
        if (g_ctx.logFile.is_open()) g_ctx.logFile << "[NeuralShader] ERROR: Failed to load DirectML.dll\n";
        g_ctx.has_failed = true;
        return;
    }

    auto pfnDMLCreateDevice = reinterpret_cast<PFN_DMLCreateDevice>(GetProcAddress(g_ctx.hDml, "DMLCreateDevice"));
    if (!pfnDMLCreateDevice || FAILED(pfnDMLCreateDevice(g_ctx.d3d12Device.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&g_ctx.dmlDevice)))) {
        if (g_ctx.logFile.is_open()) g_ctx.logFile << "[NeuralShader] ERROR: Failed to create DirectML device.\n";
        g_ctx.has_failed = true;
        return;
    }

    // 3. Initialize ORT environment
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
    g_ctx.ort->SetIntraOpNumThreads(g_ctx.sessionOptions, 2);
    g_ctx.ort->SetSessionGraphOptimizationLevel(g_ctx.sessionOptions, ORT_ENABLE_BASIC);

    SetEnvironmentVariableA("DML_DISABLE_METACMDS", "1");

    // Pass valid dmlDevice AND valid dedicated command queue (never nullptr)
    g_ctx.ortDml->SessionOptionsAppendExecutionProvider_DML1(
        g_ctx.sessionOptions,
        g_ctx.dmlDevice.Get(),
        g_ctx.dedicatedDmlQueue.Get()
    );

    std::wstring fullModelPath = g_ctx.modelDirectory + L"\\neural_shader_gtx1650_432p.onnx";
    OrtStatus* status = g_ctx.ort->CreateSession(g_ctx.env, fullModelPath.c_str(), g_ctx.sessionOptions, &g_ctx.session);
    if (status != nullptr) {
        if (g_ctx.logFile.is_open()) {
            g_ctx.logFile << "[NeuralShader] ERROR: Session creation failed: " << g_ctx.ort->GetErrorMessage(status) << "\n";
            g_ctx.logFile.flush();
        }
        g_ctx.ort->ReleaseStatus(status);
        g_ctx.has_failed = true;
        return;
    }

    // 4. Allocate memory buffers
    size_t inElements  = (size_t)IN_WIDTH * IN_HEIGHT * IN_CHANNELS;
    size_t outElements = (size_t)IN_WIDTH * IN_HEIGHT * OUT_CHANNELS;
    g_ctx.inputData.assign(inElements, 0.5f);
    g_ctx.outputData.assign(outElements, 0.0f);

    g_ctx.ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &g_ctx.memInfo);

    int64_t inShape[]  = { 1, IN_CHANNELS, IN_HEIGHT, IN_WIDTH };
    int64_t outShape[] = { 1, OUT_CHANNELS, IN_HEIGHT, IN_WIDTH };

    g_ctx.ort->CreateTensorWithDataAsOrtValue(
        g_ctx.memInfo, g_ctx.inputData.data(), inElements * sizeof(float),
        inShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &g_ctx.inTensor
    );
    g_ctx.ort->CreateTensorWithDataAsOrtValue(
        g_ctx.memInfo, g_ctx.outputData.data(), outElements * sizeof(float),
        outShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &g_ctx.outTensor
    );

    if (g_ctx.logFile.is_open()) {
        g_ctx.logFile << "[NeuralShader] DirectML session initialized on dedicated GTX 1650 queue!\n";
        g_ctx.logFile.flush();
    }

    g_ctx.is_ready = true;
}

static void on_present(reshade::api::command_queue*, reshade::api::swapchain* swapchain, const reshade::api::rect*, const reshade::api::rect*, uint32_t, const reshade::api::rect*) {
    uint64_t frame = ++g_ctx.total_frames;

    if (swapchain->get_device()->get_api() != reshade::api::device_api::d3d12) return;
    if (g_ctx.has_failed) return;

    if (!g_ctx.is_ready && !g_ctx.is_loading && frame > 60) {
        g_ctx.is_loading = true;
        g_ctx.d3d12Device = reinterpret_cast<ID3D12Device*>(swapchain->get_device()->get_native());
        std::thread initThread(AsyncInitDirectML);
        initThread.detach();
    }

    if (g_ctx.is_ready && (frame % 2 == 0)) {
        bool expected = false;
        if (g_isInferring.compare_exchange_strong(expected, true)) {
            OrtValue* outputs[1] = { g_ctx.outTensor };

            OrtStatus* status = g_ctx.ort->Run(
                g_ctx.session,
                nullptr,
                g_ctx.inputNames,
                (const OrtValue* const*)&g_ctx.inTensor,
                1,
                g_ctx.outputNames,
                1,
                outputs
            );

            if (status != nullptr) {
                if (g_ctx.logFile.is_open()) {
                    g_ctx.logFile << "[NeuralShader] Error during Run: " << g_ctx.ort->GetErrorMessage(status) << "\n";
                    g_ctx.logFile.flush();
                }
                g_ctx.ort->ReleaseStatus(status);
                g_ctx.has_failed = true;
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

extern "C" __declspec(dllexport) const char* NAME = "Neural Shader (GTX 1650 Dedicated Queue)";
extern "C" __declspec(dllexport) const char* DESCRIPTION = "Dedicated Queue DirectML Addon for Forza Horizon 2";

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
            g_ctx.logFile << "[NeuralShader] Addon attached.\n";
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
            if (g_ctx.inTensor) g_ctx.ort->ReleaseValue(g_ctx.inTensor);
            if (g_ctx.outTensor) g_ctx.ort->ReleaseValue(g_ctx.outTensor);
            if (g_ctx.memInfo) g_ctx.ort->ReleaseMemoryInfo(g_ctx.memInfo);
            if (g_ctx.session) g_ctx.ort->ReleaseSession(g_ctx.session);
            if (g_ctx.sessionOptions) g_ctx.ort->ReleaseSessionOptions(g_ctx.sessionOptions);
            if (g_ctx.env) g_ctx.ort->ReleaseEnv(g_ctx.env);
            if (g_ctx.hDml) FreeLibrary(g_ctx.hDml);
        }

        if (g_ctx.logFile.is_open()) {
            g_ctx.logFile << "[NeuralShader] Addon detached cleanly.\n";
            g_ctx.logFile.close();
        }
        break;
    }
    return TRUE;
}
