#include <reshade.hpp>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <DirectML.h>
#include <onnxruntime_c_api.h>
#include <dml_provider_factory.h>

#include <vector>
#include <atomic>
#include <fstream>
#include <string>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

using Microsoft::WRL::ComPtr;

constexpr int IN_WIDTH     = 768;
constexpr int IN_HEIGHT    = 432;
constexpr int IN_CHANNELS  = 7;
constexpr int OUT_CHANNELS = 3;

typedef HRESULT (WINAPI* PFN_DMLCreateDevice)(
    ID3D12Device* d3d12Device,
    DML_CREATE_DEVICE_FLAGS flags,
    REFIID riid,
    void** ppv
);

struct NeuralState {
    std::atomic<bool> ready{false};
    std::atomic<bool> failed{false};
    std::atomic<bool> running{false};
    uint64_t frame_count = 0;
    uint64_t ai_count = 0;

    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    HMODULE hDml = nullptr;
    ComPtr<IDMLDevice> dmlDevice;

    const OrtApi* ort = nullptr;
    const OrtDmlApi* ortDml = nullptr;
    OrtEnv* env = nullptr;
    OrtSessionOptions* sessionOptions = nullptr;
    OrtSession* session = nullptr;
    OrtMemoryInfo* memInfo = nullptr;
    OrtValue* inTensor = nullptr;

    const char* inputNames[1]  = { "g_buffer" };
    const char* outputNames[1] = { "rendered_frame" };

    std::vector<float> inputData;
    std::wstring moduleDir;
    std::ofstream log;
};

static NeuralState g_st;

static void CleanupORT() {
    g_st.ready = false;
    if (g_st.inTensor && g_st.ort) { g_st.ort->ReleaseValue(g_st.inTensor); g_st.inTensor = nullptr; }
    if (g_st.memInfo && g_st.ort)  { g_st.ort->ReleaseMemoryInfo(g_st.memInfo); g_st.memInfo = nullptr; }
    if (g_st.session && g_st.ort)  { g_st.ort->ReleaseSession(g_st.session); g_st.session = nullptr; }
    if (g_st.sessionOptions && g_st.ort) { g_st.ort->ReleaseSessionOptions(g_st.sessionOptions); g_st.sessionOptions = nullptr; }
    if (g_st.env && g_st.ort)      { g_st.ort->ReleaseEnv(g_st.env); g_st.env = nullptr; }
    if (g_st.dmlDevice)            { g_st.dmlDevice.Reset(); }
    if (g_st.queue)                { g_st.queue.Reset(); }
    if (g_st.device)               { g_st.device.Reset(); }
    if (g_st.hDml)                 { FreeLibrary(g_st.hDml); g_st.hDml = nullptr; }
}

static bool InitDirectML() {
    if (!g_st.device) return false;

    D3D12_COMMAND_QUEUE_DESC qDesc = {};
    qDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    qDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    if (FAILED(g_st.device->CreateCommandQueue(&qDesc, IID_PPV_ARGS(&g_st.queue)))) {
        if (g_st.log.is_open()) g_st.log << "[NeuralShader] Failed to create D3D12 queue\n";
        return false;
    }

    g_st.hDml = LoadLibraryA("DirectML.dll");
    if (!g_st.hDml) {
        if (g_st.log.is_open()) g_st.log << "[NeuralShader] Failed to load DirectML.dll\n";
        return false;
    }

    auto pfnDML = reinterpret_cast<PFN_DMLCreateDevice>(GetProcAddress(g_st.hDml, "DMLCreateDevice"));
    if (!pfnDML || FAILED(pfnDML(g_st.device.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&g_st.dmlDevice)))) {
        if (g_st.log.is_open()) g_st.log << "[NeuralShader] DMLCreateDevice failed\n";
        return false;
    }

    const OrtApiBase* apiBase = OrtGetApiBase();
    g_st.ort = apiBase->GetApi(17);
    if (!g_st.ort) return false;

    g_st.ort->GetExecutionProviderApi("DML", 17, reinterpret_cast<const void**>(&g_st.ortDml));
    if (!g_st.ortDml) return false;

    g_st.ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "Azimuth_Inference", &g_st.env);
    g_st.ort->CreateSessionOptions(&g_st.sessionOptions);
    g_st.ort->SetIntraOpNumThreads(g_st.sessionOptions, 1);
    g_st.ort->SetSessionGraphOptimizationLevel(g_st.sessionOptions, ORT_ENABLE_BASIC);

    SetEnvironmentVariableA("DML_DISABLE_METACMDS", "1");
    g_st.ortDml->SessionOptionsAppendExecutionProvider_DML1(g_st.sessionOptions, g_st.dmlDevice.Get(), g_st.queue.Get());

    std::wstring modelPath = g_st.moduleDir + L"\\neural_shader_gtx1650_432p.onnx";
    OrtStatus* status = g_st.ort->CreateSession(g_st.env, modelPath.c_str(), g_st.sessionOptions, &g_st.session);
    if (status != nullptr) {
        if (g_st.log.is_open()) {
            g_st.log << "[NeuralShader] CreateSession failed: " << g_st.ort->GetErrorMessage(status) << "\n";
        }
        g_st.ort->ReleaseStatus(status);
        return false;
    }

    size_t inElems = (size_t)IN_WIDTH * IN_HEIGHT * IN_CHANNELS;
    g_st.inputData.assign(inElems, 0.5f);
    g_st.ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &g_st.memInfo);

    int64_t inShape[] = { 1, IN_CHANNELS, IN_HEIGHT, IN_WIDTH };
    g_st.ort->CreateTensorWithDataAsOrtValue(
        g_st.memInfo, g_st.inputData.data(), inElems * sizeof(float),
        inShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &g_st.inTensor
    );

    if (g_st.log.is_open()) {
        g_st.log << "[NeuralShader] DirectML active on GTX 1650.\n";
        g_st.log.flush();
    }

    g_st.ready = true;
    return true;
}

static void on_init_device(reshade::api::device* device) {
    if (device->get_api() != reshade::api::device_api::d3d12) return;
    g_st.device = reinterpret_cast<ID3D12Device*>(device->get_native());
    if (g_st.log.is_open()) {
        g_st.log << "[NeuralShader] D3D12 Device attached: " << g_st.device.Get() << "\n";
        g_st.log.flush();
    }
}

static void on_destroy_device(reshade::api::device* device) {
    if (device->get_api() != reshade::api::device_api::d3d12) return;
    if (g_st.log.is_open()) {
        g_st.log << "[NeuralShader] D3D12 Device destroying, cleaning up...\n";
        g_st.log.flush();
    }
    CleanupORT();
}

static void on_present(reshade::api::command_queue*, reshade::api::swapchain*, const reshade::api::rect*, const reshade::api::rect*, uint32_t, const reshade::api::rect*) {
    if (g_st.failed || !g_st.device) return;

    g_st.frame_count++;

    if (!g_st.ready) {
        if (g_st.frame_count > 120) {
            if (!InitDirectML()) {
                g_st.failed = true;
            }
        }
        return;
    }

    if (g_st.frame_count % 2 == 0) {
        bool expected = false;
        if (g_st.running.compare_exchange_strong(expected, true)) {
            OrtValue* outputTensor = nullptr;

            OrtStatus* status = g_st.ort->Run(
                g_st.session,
                nullptr,
                g_st.inputNames,
                (const OrtValue* const*)&g_st.inTensor,
                1,
                g_st.outputNames,
                1,
                &outputTensor
            );

            if (status != nullptr) {
                if (g_st.log.is_open()) {
                    g_st.log << "[NeuralShader] Inference error: " << g_st.ort->GetErrorMessage(status) << "\n";
                    g_st.log.flush();
                }
                g_st.ort->ReleaseStatus(status);
                g_st.failed = true;
            } else {
                if (outputTensor) {
                    g_st.ort->ReleaseValue(outputTensor);
                }
                g_st.ai_count++;
                if (g_st.ai_count % 60 == 0 && g_st.log.is_open()) {
                    g_st.log << "[NeuralShader] Rendered AI frames: " << g_st.ai_count << " (Game: " << g_st.frame_count << ")\n";
                    g_st.log.flush();
                }
            }
            g_st.running.store(false);
        }
    }
}

extern "C" __declspec(dllexport) const char* NAME = "Neural Shader (GTX 1650 Direct)";
extern "C" __declspec(dllexport) const char* DESCRIPTION = "DirectML Runtime for Forza Horizon 2";

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID) {
    switch (fdwReason) {
    case DLL_PROCESS_ATTACH: {
        wchar_t path[MAX_PATH];
        GetModuleFileNameW(hinstDLL, path, MAX_PATH);
        std::wstring p(path);
        size_t idx = p.find_last_of(L"\\/");
        if (idx != std::wstring::npos) g_st.moduleDir = p.substr(0, idx);

        g_st.log.open("neural_shader_debug.txt", std::ios::out);
        if (g_st.log.is_open()) {
            g_st.log << "[NeuralShader] Addon loaded.\n";
            g_st.log.flush();
        }

        if (!reshade::register_addon(hinstDLL)) return FALSE;
        reshade::register_event<reshade::addon_event::init_device>(on_init_device);
        reshade::register_event<reshade::addon_event::destroy_device>(on_destroy_device);
        reshade::register_event<reshade::addon_event::present>(on_present);
        break;
    }
    case DLL_PROCESS_DETACH:
        reshade::unregister_event<reshade::addon_event::present>(on_present);
        reshade::unregister_event<reshade::addon_event::destroy_device>(on_destroy_device);
        reshade::unregister_event<reshade::addon_event::init_device>(on_init_device);
        reshade::unregister_addon(hinstDLL);
        CleanupORT();
        if (g_st.log.is_open()) {
            g_st.log << "[NeuralShader] Addon unloaded cleanly.\n";
            g_st.log.close();
        }
        break;
    }
    return TRUE;
}
