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
#include <thread>
#include <chrono>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

constexpr int IN_WIDTH     = 768;
constexpr int IN_HEIGHT    = 432;
constexpr int IN_CHANNELS  = 7;
constexpr int OUT_CHANNELS = 3;

struct NeuralState {
    std::atomic<bool> ready{false};
    std::atomic<bool> initializing{false};
    std::atomic<bool> failed{false};
    std::atomic<bool> running{false};

    std::atomic<uint64_t> frame_count{0};
    std::atomic<uint64_t> ai_count{0};
    std::atomic<float> last_ai_ms{0.0f};

    const OrtApi* ort = nullptr;
    OrtEnv* env = nullptr;
    OrtSessionOptions* sessionOptions = nullptr;
    OrtSession* session = nullptr;
    OrtMemoryInfo* memInfo = nullptr;
    OrtValue* inTensor = nullptr;

    const char* inputNames[1]  = { "g_buffer" };
    const char* outputNames[1] = { "rendered_frame" };

    std::vector<float> inputData;
    std::vector<char> modelBytes;
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
}

static void InitDirectMLWorker() {
    if (g_st.log.is_open()) {
        g_st.log << "[NeuralShader] Step 1: Worker thread started.\n";
        g_st.log.flush();
    }

    const OrtApiBase* apiBase = OrtGetApiBase();
    g_st.ort = apiBase->GetApi(17);
    if (!g_st.ort) {
        if (g_st.log.is_open()) g_st.log << "[NeuralShader] ERROR: Failed to get OrtApi v17\n";
        g_st.failed = true;
        return;
    }

    g_st.ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "Azimuth_Inference", &g_st.env);
    g_st.ort->CreateSessionOptions(&g_st.sessionOptions);
    g_st.ort->SetIntraOpNumThreads(g_st.sessionOptions, 1);
    g_st.ort->SetSessionGraphOptimizationLevel(g_st.sessionOptions, ORT_ENABLE_BASIC);

    SetEnvironmentVariableA("DML_DISABLE_METACMDS", "1");

    if (g_st.log.is_open()) {
        g_st.log << "[NeuralShader] Step 2: Appending DirectML execution provider...\n";
        g_st.log.flush();
    }

    OrtSessionOptionsAppendExecutionProvider_DML(g_st.sessionOptions, 0);

    std::wstring modelPath = g_st.moduleDir + L"\\neural_shader_gtx1650_432p.onnx";
    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        if (g_st.log.is_open()) {
            g_st.log << "[NeuralShader] ERROR: Could not open neural_shader_gtx1650_432p.onnx\n";
            g_st.log.flush();
        }
        g_st.failed = true;
        return;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    g_st.modelBytes.resize(size);
    if (!file.read(g_st.modelBytes.data(), size)) {
        if (g_st.log.is_open()) g_st.log << "[NeuralShader] ERROR: Failed to read model bytes.\n";
        g_st.failed = true;
        return;
    }

    if (g_st.log.is_open()) {
        g_st.log << "[NeuralShader] Step 3: Compiling session from memory (" << size << " bytes)...\n";
        g_st.log.flush();
    }

    OrtStatus* status = g_st.ort->CreateSessionFromArray(
        g_st.env,
        g_st.modelBytes.data(),
        g_st.modelBytes.size(),
        g_st.sessionOptions,
        &g_st.session
    );

    if (status != nullptr) {
        if (g_st.log.is_open()) {
            g_st.log << "[NeuralShader] ERROR: CreateSessionFromArray failed: " << g_st.ort->GetErrorMessage(status) << "\n";
            g_st.log.flush();
        }
        g_st.ort->ReleaseStatus(status);
        g_st.failed = true;
        return;
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
        g_st.log << "[NeuralShader] DirectML active on GTX 1650! Inference loop ready.\n";
        g_st.log.flush();
    }

    g_st.ready = true;
}

static void on_present(reshade::api::command_queue* queue, reshade::api::swapchain* swapchain, const reshade::api::rect*, const reshade::api::rect*, uint32_t, const reshade::api::rect*) {
    uint64_t current_frame = ++g_st.frame_count;

    // Trigger initialization after passing frame 120
    if (!g_st.ready && !g_st.initializing && !g_st.failed && current_frame > 120) {
        g_st.initializing = true;
        std::thread(InitDirectMLWorker).detach();
        return;
    }

    // Run inference every 2nd frame
    if (g_st.ready && (current_frame % 2 == 0)) {
        bool expected = false;
        if (g_st.running.compare_exchange_strong(expected, true)) {
            OrtValue* outputTensor = nullptr;

            auto t0 = std::chrono::high_resolution_clock::now();

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

            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            g_st.last_ai_ms.store(ms);

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
                uint64_t ai = ++g_st.ai_count;

                if (ai % 60 == 0) {
                    char titleBuf[128];
                    snprintf(titleBuf, sizeof(titleBuf), "Forza Horizon 2 [AI: %llu Frames | %.1f ms]", (unsigned long long)ai, ms);
                    SetConsoleTitleA(titleBuf);

                    if (g_st.log.is_open()) {
                        g_st.log << "[NeuralShader] Rendered AI frames: " << ai << " (Game: " << current_frame << " | " << ms << " ms)\n";
                        g_st.log.flush();
                    }
                }
            }
            g_st.running.store(false);
        }
    }

    // Screen Visual Effect: Render HUD indicator on top of active game frame
    if (swapchain != nullptr && queue != nullptr) {
        // If AI is actively running or has failed, draw an on-screen status rectangle
        if (g_st.ready || g_st.failed) {
            reshade::api::resource current_backbuffer = swapchain->get_current_back_buffer();
            reshade::api::resource_view rtv = swapchain->get_device()->get_resource_view(current_backbuffer);

            // Green/Cyan pulse when working normally, Solid Red if failed
            float color[4];
            if (g_st.failed) {
                color[0] = 1.0f; color[1] = 0.0f; color[2] = 0.0f; color[3] = 1.0f; // Red
            } else {
                // Subtle pulse effect tied to AI frame count
                float pulse = 0.6f + 0.4f * sinf(g_st.ai_count.load() * 0.1f);
                color[0] = 0.0f; color[1] = pulse; color[2] = 0.8f * pulse; color[3] = 1.0f; // Neon Cyan/Green
            }

            // Top-left HUD box: (left: 20, top: 20, right: 220, bottom: 35)
            reshade::api::rect hudRect = { 20, 20, 220, 35 };

            reshade::api::command_list* cmdList = queue->get_immediate_command_list();
            if (cmdList != nullptr) {
                cmdList->clear_render_target_view(rtv, color, 1, &hudRect);
            }
        }
    }
}

extern "C" __declspec(dllexport) const char* NAME = "Neural Shader (HUD Visual Feed)";
extern "C" __declspec(dllexport) const char* DESCRIPTION = "DirectML Runtime with Screen Overlay Effect";

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
        reshade::register_event<reshade::addon_event::present>(on_present);
        break;
    }
    case DLL_PROCESS_DETACH:
        reshade::unregister_event<reshade::addon_event::present>(on_present);
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
