#define RESHADE_ADDON_IMPL 1
#include <reshade.hpp>
#include <onnxruntime_cxx_api.h>

static Ort::Env* g_ort_env = nullptr;
static Ort::Session* g_ort_session = nullptr;

static void on_init_effect_runtime(reshade::api::effect_runtime* runtime) {
    if (g_ort_session) return;
    try {
        g_ort_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "NeuralShader");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        const wchar_t* model_path = L"neural_shader_gtx1650_sharp.onnx";
        g_ort_session = new Ort::Session(*g_ort_env, model_path, session_options);
    }
    catch (...) {}
}

static void on_destroy_effect_runtime(reshade::api::effect_runtime* runtime) {
    if (g_ort_session) { delete g_ort_session; g_ort_session = nullptr; }
    if (g_ort_env) { delete g_ort_env; g_ort_env = nullptr; }
}

// Official ReShade entry point
extern "C" __declspec(dllexport) bool ReShadePluginInit(HMODULE hmodule) {
    if (!reshade::register_addon(hmodule))
        return false;

    reshade::register_event<reshade::addon_event::init_effect_runtime>(on_init_effect_runtime);
    reshade::register_event<reshade::addon_event::destroy_effect_runtime>(on_destroy_effect_runtime);
    return true;
}

extern "C" __declspec(dllexport) void ReShadePluginUninstall(HMODULE hmodule) {
    reshade::unregister_event<reshade::addon_event::init_effect_runtime>(on_init_effect_runtime);
    reshade::unregister_event<reshade::addon_event::destroy_effect_runtime>(on_destroy_effect_runtime);
    reshade::unregister_addon(hmodule);
}
