#define RESHADE_ADDON_IMPL 1
#include <reshade.hpp>
#include <Windows.h>

static void on_present(reshade::api::command_queue* queue, reshade::api::swapchain* swapchain, const reshade::api::rect*, const reshade::api::rect*, uint32_t, const reshade::api::rect*) {
    if (!queue || !swapchain) return;

    // Beep once when tapping F10
    if (GetAsyncKeyState(VK_F10) & 0x0001) {
        MessageBeep(MB_OK);
    }

    // While HOLDING F10 down: clear the active swapchain buffer directly on the GPU
    if (GetAsyncKeyState(VK_F10) & 0x8000) {
        reshade::api::device* device = queue->get_device();
        if (!device) return;

        reshade::api::resource backbuffer = swapchain->get_current_back_buffer();
        if (backbuffer.handle == 0) return;

        reshade::api::command_list* cmd_list = queue->get_immediate_command_list();
        if (!cmd_list) return;

        reshade::api::resource_desc desc = device->get_resource_desc(backbuffer);
        reshade::api::resource_view rtv = { 0 };

        if (device->create_resource_view(backbuffer, reshade::api::resource_usage::render_target, reshade::api::resource_view_desc(desc.texture.format), &rtv)) {
            // Bright neon magenta (R=1, G=0, B=1, A=1)
            const float magenta[4] = { 1.0f, 0.0f, 1.0f, 1.0f };
            cmd_list->clear_render_target_view(rtv, magenta, 0, nullptr);
            device->destroy_resource_view(rtv);
        }
    }
}

extern "C" __declspec(dllexport) const char* NAME = "Direct Swapchain Test";
extern "C" __declspec(dllexport) const char* DESCRIPTION = "Tests direct GPU clear on active swapchain buffer";

BOOL APIENTRY DllMain(HMODULE hModule, DWORD fdwReason, LPVOID) {
    switch (fdwReason) {
        case DLL_PROCESS_ATTACH:
            if (!reshade::register_addon(hModule)) return FALSE;
            reshade::register_event<reshade::addon_event::present>(on_present);
            break;
        case DLL_PROCESS_DETACH:
            reshade::unregister_event<reshade::addon_event::present>(on_present);
            reshade::unregister_addon(hModule);
            break;
    }
    return TRUE;
}
