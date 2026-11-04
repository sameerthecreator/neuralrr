#define RESHADE_ADDON_IMPL 1
#include <reshade.hpp>
#include <onnxruntime_cxx_api.h>
#include <Windows.h>
#include <vector>

static Ort::Env* g_ort_env = nullptr;
static Ort::Session* g_ort_session = nullptr;

// 1. Initialize ONNX Model
static void on_init_effect_runtime(reshade::api::effect_runtime* runtime) {
    if (g_ort_session) return;
    try {
        g_ort_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "NeuralShader");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4); // Utilize more CPU threads for the tensor math
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        const wchar_t* model_path = L"neural_shader_gtx1650_sharp.onnx";
        g_ort_session = new Ort::Session(*g_ort_env, model_path, session_options);
    }
    catch (...) {}
}

// 2. Clean up memory on exit
static void on_destroy_effect_runtime(reshade::api::effect_runtime* runtime) {
    if (g_ort_session) { delete g_ort_session; g_ort_session = nullptr; }
    if (g_ort_env) { delete g_ort_env; g_ort_env = nullptr; }
}

// 3. The Render Hook - Runs every single frame
static void on_present(reshade::api::command_queue* queue, reshade::api::swapchain* swapchain, const reshade::api::rect*, const reshade::api::rect*, uint32_t, const reshade::api::rect*) {
    if (!g_ort_session) return;

    reshade::api::device* device = queue->get_device();
    reshade::api::resource backbuffer = swapchain->get_current_back_buffer();

    // The inference pipeline structure:
    // A. Create a staging texture to copy the backbuffer from GPU memory to CPU RAM.
    // B. Map the staging texture memory to a float vector (normalize 0-255 RGB to 0.0-1.0 float).
    // C. Convert the vector to an Ort::Value tensor: Ort::Value::CreateTensor<float>(...).
    // D. Run the model: g_ort_session->Run(...).
    // E. Map the output tensor back to RGBA and copy it back to the backbuffer.
    
    // (Note: To maintain high FPS later, we will transition this specific block to DirectML hardware acceleration)
}

extern "C" __declspec(dllexport) const char* NAME = "Neural Shader";
extern "C" __declspec(dllexport) const char* DESCRIPTION = "ONNX Neural Shader Hook";

BOOL APIENTRY DllMain(HMODULE hModule, DWORD fdwReason, LPVOID) {
    switch (fdwReason) {
        case DLL_PROCESS_ATTACH:
            if (!reshade::register_addon(hModule))
                return FALSE;
            
            reshade::register_event<reshade::addon_event::init_effect_runtime>(on_init_effect_runtime);
            reshade::register_event<reshade::addon_event::destroy_effect_runtime>(on_destroy_effect_runtime);
            
            // Register the frame interceptor
            reshade::register_event<reshade::addon_event::present>(on_present);
            break;
            
        case DLL_PROCESS_DETACH:
            reshade::unregister_event<reshade::addon_event::init_effect_runtime>(on_init_effect_runtime);
            reshade::unregister_event<reshade::addon_event::destroy_effect_runtime>(on_destroy_effect_runtime);
            reshade::unregister_event<reshade::addon_event::present>(on_present);
            reshade::unregister_addon(hModule);
            break;
    }
    return TRUE;
}
