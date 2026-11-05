#include <reshade.hpp>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <onnxruntime_c_api.h>

#include <vector>
#include <atomic>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>
#include <algorithm>
#include <mutex>
#include <condition_variable>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

constexpr int IN_WIDTH     = 768;
constexpr int IN_HEIGHT    = 432;
constexpr int IN_CHANNELS  = 7;
constexpr int OUT_CHANNELS = 3;

typedef const OrtApiBase* (ORT_API_CALL *PFN_OrtGetApiBase)(void);
typedef OrtStatus* (ORT_API_CALL *PFN_OrtSessionOptionsAppendExecutionProvider_DML)(OrtSessionOptions* options, int device_id);

struct NeuralState {
    std::atomic<bool> ready{false};
    std::atomic<bool> initializing{false};
    std::atomic<bool> failed{false};
    
    // Transition and Thread Safety
    std::atomic<bool> should_exit{false};
    std::atomic<bool> trigger_inference{false};
    std::atomic<bool> is_paused{false};
    std::atomic<bool> has_new_frame{false};
    
    std::mutex queueMutex;
    std::condition_variable cv;
    std::mutex inferMutex; // Locks GPU execution during swapchain teardown
    std::thread workerThread;

    std::atomic<uint64_t> frame_count{0};
    std::atomic<uint64_t> ai_count{0};
    std::atomic<float> last_ai_ms{0.0f};

    HMODULE hOrt = nullptr;
    const OrtApi* ort = nullptr;
    PFN_OrtSessionOptionsAppendExecutionProvider_DML pfnAppendDml = nullptr;

    OrtEnv* env = nullptr;
    OrtSessionOptions* sessionOptions = nullptr;
    OrtSession* session = nullptr;
    OrtMemoryInfo* memInfo = nullptr;
    OrtValue* inTensor = nullptr;

    const char* inputNames[1]  = { "g_buffer" };
    const char* outputNames[1] = { "rendered_frame" };

    std::vector<float> inputData;
    std::vector<uint8_t> pixelBufferRGBA;
    std::vector<char> modelBytes;

    reshade::api::resource aiTextureRes = {};

    std::wstring moduleDir;
    std::ofstream log;
};

static NeuralState g_st;

static void CleanupORT() {
    g_st.should_exit = true;
    g_st.cv.notify_all();
    if (g_st.workerThread.joinable()) {
        g_st.workerThread.join();
    }

    g_st.ready = false;
    if (g_st.inTensor && g_st.ort) { g_st.ort->ReleaseValue(g_st.inTensor); g_st.inTensor = nullptr; }
    if (g_st.memInfo && g_st.ort)  { g_st.ort->ReleaseMemoryInfo(g_st.memInfo); g_st.memInfo = nullptr; }
    if (g_st.session && g_st.ort)  { g_st.ort->ReleaseSession(g_st.session); g_st.session = nullptr; }
    if (g_st.sessionOptions && g_st.ort) { g_st.ort->ReleaseSessionOptions(g_st.sessionOptions); g_st.sessionOptions = nullptr; }
    if (g_st.env && g_st.ort)      { g_st.ort->ReleaseEnv(g_st.env); g_st.env = nullptr; }
    if (g_st.hOrt)                 { FreeLibrary(g_st.hOrt); g_st.hOrt = nullptr; }
}

static void InferenceLoop() {
    if (g_st.log.is_open()) {
        g_st.log << "[NeuralShader] Inference thread active.\n";
        g_st.log.flush();
    }

    while (!g_st.should_exit) {
        std::unique_lock<std::mutex> lock(g_st.queueMutex);
        g_st.cv.wait(lock, [] { return g_st.trigger_inference.load() || g_st.should_exit.load(); });

        if (g_st.should_exit) break;
        g_st.trigger_inference = false;

        if (g_st.is_paused) continue;

        // Lock inference so the swapchain cannot be destroyed while DirectML is computing
        std::lock_guard<std::mutex> inferLock(g_st.inferMutex);

        // Double check pause state after acquiring lock
        if (g_st.is_paused) continue;

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
            break;
        } else {
            if (outputTensor) {
                float* outData = nullptr;
                g_st.ort->GetTensorMutableData(outputTensor, reinterpret_cast<void**>(&outData));

                if (outData != nullptr) {
                    const int planeSize = IN_WIDTH * IN_HEIGHT;
                    const float* rPlane = outData;
                    const float* gPlane = outData + planeSize;
                    const float* bPlane = outData + 2 * planeSize;
                    uint8_t* dst = g_st.pixelBufferRGBA.data();

                    for (int i = 0; i < planeSize; ++i) {
                        dst[i * 4 + 0] = static_cast<uint8_t>(std::clamp(rPlane[i], 0.0f, 1.0f) * 255.0f);
                        dst[i * 4 + 1] = static_cast<uint8_t>(std::clamp(gPlane[i], 0.0f, 1.0f) * 255.0f);
                        dst[i * 4 + 2] = static_cast<uint8_t>(std::clamp(bPlane[i], 0.0f, 1.0f) * 255.0f);
                        dst[i * 4 + 3] = 255;
                    }
                    g_st.has_new_frame.store(true);
                }
                g_st.ort->ReleaseValue(outputTensor);
            }

            uint64_t ai = ++g_st.ai_count;
            if (ai % 60 == 0) {
                char titleBuf[128];
                snprintf(titleBuf, sizeof(titleBuf), "Forza Horizon 2 [AI Frames: %llu | %.1f ms]", (unsigned long long)ai, ms);
                SetConsoleTitleA(titleBuf);

                if (g_st.log.is_open()) {
                    g_st.log << "[NeuralShader] Output frame: " << ai << " (Latency: " << ms << " ms)\n";
                    g_st.log.flush();
                }
            }
        }
    }
}

static void InitDirectMLWorker() {
    g_st.hOrt = LoadLibraryA("onnxruntime.dll");
    if (!g_st.hOrt) {
        if (g_st.log.is_open()) g_st.log << "[NeuralShader] ERROR: Failed to load onnxruntime.dll\n";
        g_st.failed = true;
        return;
    }

    auto pfnGetApiBase = reinterpret_cast<PFN_OrtGetApiBase>(GetProcAddress(g_st.hOrt, "OrtGetApiBase"));
    g_st.pfnAppendDml = reinterpret_cast<PFN_OrtSessionOptionsAppendExecutionProvider_DML>(GetProcAddress(g_st.hOrt, "OrtSessionOptionsAppendExecutionProvider_DML"));

    if (!pfnGetApiBase) {
        g_st.failed = true;
        return;
    }

    const OrtApiBase* apiBase = pfnGetApiBase();
    g_st.ort = apiBase->GetApi(17);
    if (!g_st.ort) {
        g_st.failed = true;
        return;
    }

    g_st.ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "Azimuth_Inference", &g_st.env);
    g_st.ort->CreateSessionOptions(&g_st.sessionOptions);
    g_st.ort->SetIntraOpNumThreads(g_st.sessionOptions, 1);
    
    // VRAM Optimization for 4GB GTX 1650: Disable Arena to prevent Out-of-Memory TDR during loading screens
    g_st.ort->DisableCpuMemArena(g_st.sessionOptions);
    g_st.ort->DisableMemPattern(g_st.sessionOptions);
    g_st.ort->SetSessionGraphOptimizationLevel(g_st.sessionOptions, ORT_ENABLE_BASIC);

    if (g_st.pfnAppendDml) {
        g_st.pfnAppendDml(g_st.sessionOptions, 0);
    }

    std::wstring modelPath = g_st.moduleDir + L"\\neural_shader_gtx1650_432p.onnx";
    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        g_st.failed = true;
        return;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    g_st.modelBytes.resize(size);
    file.read(g_st.modelBytes.data(), size);

    OrtStatus* status = g_st.ort->CreateSessionFromArray(
        g_st.env, g_st.modelBytes.data(), g_st.modelBytes.size(), g_st.sessionOptions, &g_st.session
    );

    if (status != nullptr) {
        g_st.ort->ReleaseStatus(status);
        g_st.failed = true;
        return;
    }

    size_t inElems = (size_t)IN_WIDTH * IN_HEIGHT * IN_CHANNELS;
    g_st.inputData.assign(inElems, 0.5f);
    g_st.pixelBufferRGBA.resize((size_t)IN_WIDTH * IN_HEIGHT * 4, 255);

    g_st.ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &g_st.memInfo);

    int64_t inShape[] = { 1, IN_CHANNELS, IN_HEIGHT, IN_WIDTH };
    g_st.ort->CreateTensorWithDataAsOrtValue(
        g_st.memInfo, g_st.inputData.data(), inElems * sizeof(float),
        inShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &g_st.inTensor
    );

    g_st.workerThread = std::thread(InferenceLoop);

    if (g_st.log.is_open()) {
        g_st.log << "[NeuralShader] DirectML active on GTX 1650. Protections enabled.\n";
        g_st.log.flush();
    }

    g_st.ready = true;
}

// Safely resume inference when swapchain builds
static void on_init_swapchain(reshade::api::swapchain* swapchain) {
    g_st.is_paused = false;
    if (g_st.log.is_open()) {
        g_st.log << "[NeuralShader] Swapchain initialized. Resuming.\n";
        g_st.log.flush();
    }
}

// Safely pause inference and destroy textures when game hits loading screens
static void on_destroy_swapchain(reshade::api::swapchain* swapchain) {
    g_st.is_paused = true;
    
    // Block until the active AI frame finishes computing to prevent TDR
    std::lock_guard<std::mutex> inferLock(g_st.inferMutex);

    if (g_st.aiTextureRes.handle != 0 && swapchain != nullptr) {
        swapchain->get_device()->destroy_resource(g_st.aiTextureRes);
        g_st.aiTextureRes = {};
        g_st.has_new_frame = false;
    }

    if (g_st.log.is_open()) {
        g_st.log << "[NeuralShader] Swapchain destroyed. Inference paused safely.\n";
        g_st.log.flush();
    }
}

static void on_present(reshade::api::command_queue* queue, reshade::api::swapchain* swapchain, const reshade::api::rect*, const reshade::api::rect*, uint32_t, const reshade::api::rect*) {
    if (g_st.failed || g_st.is_paused) return;

    uint64_t current_frame = ++g_st.frame_count;

    if (!g_st.ready && !g_st.initializing && current_frame > 300) {
        g_st.initializing = true;
        std::thread(InitDirectMLWorker).detach();
        return;
    }

    // Lazy create screen texture safely
    if (g_st.ready && g_st.aiTextureRes.handle == 0 && swapchain != nullptr) {
        auto device = swapchain->get_device();
        reshade::api::resource_desc desc = {};
        desc.type = reshade::api::resource_type::texture_2d;
        desc.texture.width = IN_WIDTH;
        desc.texture.height = IN_HEIGHT;
        desc.texture.depth_or_layers = 1;
        desc.texture.levels = 1;
        desc.texture.format = reshade::api::format::r8g8b8a8_unorm;
        desc.texture.samples = 1;
        desc.usage = reshade::api::resource_usage::copy_source | reshade::api::resource_usage::copy_dest;

        device->create_resource(desc, nullptr, reshade::api::resource_usage::copy_dest, &g_st.aiTextureRes);
    }

    // Wake the inference thread without blocking the game
    if (g_st.ready && (current_frame % 2 == 0) && !g_st.trigger_inference.load()) {
        g_st.trigger_inference = true;
        g_st.cv.notify_one();
    }

    // Blit active frame
    if (g_st.has_new_frame.load() && g_st.aiTextureRes.handle != 0 && swapchain != nullptr && queue != nullptr) {
        auto device = swapchain->get_device();
        auto cmdList = queue->get_immediate_command_list();

        reshade::api::subresource_data subData = {};
        subData.data = g_st.pixelBufferRGBA.data();
        subData.row_pitch = IN_WIDTH * 4;
        subData.slice_pitch = IN_WIDTH * IN_HEIGHT * 4;

        device->update_texture_region(subData, g_st.aiTextureRes, 0, nullptr);

        reshade::api::resource backbuffer = swapchain->get_current_back_buffer();
        if (cmdList != nullptr && backbuffer.handle != 0) {
            reshade::api::subresource_box srcBox = {};
            srcBox.left   = 0;
            srcBox.top    = 0;
            srcBox.front  = 0;
            srcBox.right  = static_cast<uint32_t>(IN_WIDTH);
            srcBox.bottom = static_cast<uint32_t>(IN_HEIGHT);
            srcBox.back   = 1;

            cmdList->copy_texture_region(
                g_st.aiTextureRes, 0, &srcBox,
                backbuffer, 0, nullptr,
                reshade::api::filter_mode::min_mag_mip_linear
            );
        }
    }
}

extern "C" __declspec(dllexport) const char* NAME = "Neural Shader (Ultimate TDR Fix)";
extern "C" __declspec(dllexport) const char* DESCRIPTION = "Async DirectML Runtime with Transition Safety";

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
            g_st.log << "[NeuralShader] Addon attached.\n";
            g_st.log.flush();
        }

        if (!reshade::register_addon(hinstDLL)) return FALSE;
        reshade::register_event<reshade::addon_event::init_swapchain>(on_init_swapchain);
        reshade::register_event<reshade::addon_event::destroy_swapchain>(on_destroy_swapchain);
        reshade::register_event<reshade::addon_event::present>(on_present);
        break;
    }
    case DLL_PROCESS_DETACH:
        reshade::unregister_event<reshade::addon_event::present>(on_present);
        reshade::unregister_event<reshade::addon_event::destroy_swapchain>(on_destroy_swapchain);
        reshade::unregister_event<reshade::addon_event::init_swapchain>(on_init_swapchain);
        reshade::unregister_addon(hinstDLL);
        CleanupORT();
        if (g_st.log.is_open()) {
            g_st.log << "[NeuralShader] Addon detached cleanly.\n";
            g_st.log.close();
        }
        break;
    }
    return TRUE;
}
