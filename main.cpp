#include <reshade.hpp>
#include <windows.h>
#include <fstream>

static std::ofstream g_log;

static void on_init_swapchain(reshade::api::swapchain* swapchain) {
    if (g_log.is_open()) {
        g_log << "[NeuralShader] Swapchain Initialized successfully!\n";
        g_log.flush();
    }
}

static void on_present(reshade::api::command_queue*, reshade::api::swapchain*, const reshade::api::rect*, const reshade::api::rect*, uint32_t, const reshade::api::rect*) {
    static uint64_t count = 0;
    count++;
    if (count % 60 == 0 && g_log.is_open()) {
        g_log << "[NeuralShader] Rendered frames: " << count << "\n";
        g_log.flush();
    }
}

extern "C" __declspec(dllexport) const char* NAME = "Neural Shader Diagnostic";
extern "C" __declspec(dllexport) const char* DESCRIPTION = "Diagnostic test build";

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID) {
    switch (fdwReason) {
    case DLL_PROCESS_ATTACH:
        g_log.open("neural_shader_debug.txt", std::ios::out);
        g_log << "[NeuralShader] Addon DLL attached to process.\n";
        g_log.flush();
        if (!reshade::register_addon(hinstDLL)) return FALSE;
        reshade::register_event<reshade::addon_event::init_swapchain>(on_init_swapchain);
        reshade::register_event<reshade::addon_event::present>(on_present);
        break;
    case DLL_PROCESS_DETACH:
        reshade::unregister_event<reshade::addon_event::present>(on_present);
        reshade::unregister_event<reshade::addon_event::init_swapchain>(on_init_swapchain);
        reshade::unregister_addon(hinstDLL);
        if (g_log.is_open()) {
            g_log << "[NeuralShader] Addon detached.\n";
            g_log.close();
        }
        break;
    }
    return TRUE;
}
