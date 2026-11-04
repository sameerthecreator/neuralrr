#define RESHADE_ADDON_IMPL 1

// 1. Fix the ONNX Runtime SAL macro error on MinGW
#ifndef _Frees_ptr_opt_
#define _Frees_ptr_opt_
#endif

#include "reshade/include/reshade.hpp"
#include "onnxruntime/onnxruntime-win-x64-1.16.1/include/onnxruntime_cxx_api.h"

Ort::Env* ort_env = nullptr;
Ort::Session* ort_session = nullptr;

// 2. Store the module handles globally to fix the API arguments error
HMODULE g_addon_module = nullptr;
HMODULE g_reshade_module = nullptr;

static void on_init_effect_runtime(reshade::api::effect_runtime *runtime) {
    try {
        ort_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "NeuralShader");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        const wchar_t* model_path = L"neural_shader_gtx1650_sharp.onnx";
        ort_session = new Ort::Session(*ort_env, model_path, session_options);
    } catch (...) {}
}

static void on_destroy_effect_runtime(reshade::api::effect_runtime *runtime) {
    if (ort_session) { delete ort_session; ort_session = nullptr; }
    if (ort_env) { delete ort_env; ort_env = nullptr; }
}

extern "C" __declspec(dllexport) bool ReShadePluginInit(HMODULE addon_module, HMODULE reshade_module) {
    g_addon_module = addon_module;
    g_reshade_module = reshade_module;
    
    // Pass the handles directly into register_addon
    if (!reshade::register_addon(addon_module, reshade_module)) return false;
    
    reshade::register_event<reshade::addon_event::init_effect_runtime>(on_init_effect_runtime);
    reshade::register_event<reshade::addon_event::destroy_effect_runtime>(on_destroy_effect_runtime);
    return true;
}

extern "C" __declspec(dllexport) void ReShadePluginUninstall() {
    reshade::unregister_event<reshade::addon_event::init_effect_runtime>(on_init_effect_runtime);
    reshade::unregister_event<reshade::addon_event::destroy_effect_runtime>(on_destroy_effect_runtime);
    
    // Unregister using the stored handles
    reshade::unregister_addon(g_addon_module, g_reshade_module);
}
