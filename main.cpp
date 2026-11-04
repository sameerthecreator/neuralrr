#define RESHADE_ADDON_IMPL 1
#include <reshade.hpp>
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>
#include <d3d12.h>
#include <Windows.h>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>

static Ort::Env* g_ort_env = nullptr;
static Ort::Session* g_ort_session = nullptr;
static Ort::MemoryInfo* g_memory_info = nullptr;

static reshade::api::resource g_gpu_input_tex = { 0 };
static reshade::api::resource g_gpu_output_tex = { 0 };
static reshade::api::resource g_staging_read = { 0 };
static reshade::api::resource g_staging_write = { 0 };

static bool g_ai_enabled = true;
static float g_ai_multiplier = 3.0f;
static HMODULE g_module_handle = nullptr;

// Internal inference resolution (prevents GTX 1650 TDR timeout while executing full model)
constexpr int INFER_WIDTH = 512;
constexpr int INFER_HEIGHT = 512;

static std::wstring get_addon_directory() {
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(g_module_handle, path, MAX_PATH);
    std::wstring str_path(path);
    size_t pos = str_path.find_last_of(L"\\/");
    return (pos != std::wstring::npos) ? str_path.substr(0, pos + 1) : L"";
}

static void on_init_swapchain(reshade::api::swapchain* swapchain, bool) {
    if (!swapchain || g_ort_session) return;

    try {
        if (!g_ort_env) {
            g_ort_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "HDR2SDR_DirectML");
        }

        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(4);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        OrtSessionOptionsAppendExecutionProvider_DML(options, 0);

        std::wstring model_path = get_addon_directory() + L"neural_shader_gtx1650_sharp.onnx";
        g_ort_session = new Ort::Session(*g_ort_env, model_path.c_str(), options);
        g_memory_info = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
    } catch (...) {}
}

static void on_destroy_swapchain(reshade::api::swapchain* swapchain, bool) {
    if (!swapchain) return;
    reshade::api::device* device = swapchain->get_device();
    if (!device) return;

    if (g_staging_read.handle != 0) { device->destroy_resource(g_staging_read); g_staging_read = { 0 }; }
    if (g_staging_write.handle != 0) { device->destroy_resource(g_staging_write); g_staging_write = { 0 }; }
}

static void on_present(reshade::api::command_queue* queue, reshade::api::swapchain* swapchain, const reshade::api::rect*, const reshade::api::rect*, uint32_t, const reshade::api::rect*) {
    if (!queue || !swapchain) return;

    if (GetAsyncKeyState(VK_F10) & 0x0001) {
        g_ai_enabled = !g_ai_enabled;
        MessageBeep(g_ai_enabled ? MB_OK : MB_ICONEXCLAMATION);
    }

    if (!g_ai_enabled) return;

    if (!g_ort_session) {
        on_init_swapchain(swapchain, false);
        if (!g_ort_session) return;
    }

    reshade::api::device* device = queue->get_device();
    reshade::api::resource backbuffer = swapchain->get_current_back_buffer();
    if (backbuffer.handle == 0) return;

    reshade::api::resource_desc bb_desc = device->get_resource_desc(backbuffer);
    int64_t bb_w = bb_desc.texture.width;
    int64_t bb_h = bb_desc.texture.height;

    // Allocate fixed-size staging buffers for the 512x512 neural pass
    if (g_staging_read.handle == 0) {
        reshade::api::resource_desc desc = {};
        desc.type = reshade::api::resource_type::texture_2d;
        desc.texture.width = INFER_WIDTH;
        desc.texture.height = INFER_HEIGHT;
        desc.texture.depth_or_layers = 1;
        desc.texture.levels = 1;
        desc.texture.format = reshade::api::format::r8g8b8a8_unorm;
        desc.heap = reshade::api::memory_heap::gpu_to_cpu;
        desc.usage = reshade::api::resource_usage::copy_dest;
        device->create_resource(desc, nullptr, reshade::api::resource_usage::copy_dest, &g_staging_read);

        desc.heap = reshade::api::memory_heap::cpu_to_gpu;
        desc.usage = reshade::api::resource_usage::copy_source;
        device->create_resource(desc, nullptr, reshade::api::resource_usage::copy_source, &g_staging_write);
    }

    reshade::api::command_list* cmd_list = queue->get_immediate_command_list();
    if (!cmd_list) return;

    // Step 1: Copy from full backbuffer to staging texture (handles downscale resolve)
    cmd_list->barrier(backbuffer, reshade::api::resource_usage::present, reshade::api::resource_usage::copy_source);
    
    reshade::api::subresource_box src_box = { 0, 0, 0, static_cast<uint32_t>(bb_w), static_cast<uint32_t>(bb_h), 1 };
    reshade::api::subresource_box dst_box = { 0, 0, 0, INFER_WIDTH, INFER_HEIGHT, 1 };
    cmd_list->copy_texture_region(backbuffer, 0, &src_box, g_staging_read, 0, &dst_box);

    queue->wait_idle();

    // Step 2: Map and pack the 7-channel planar tensor for the complete ONNX model
    reshade::api::subresource_data mapped_read;
    if (!device->map_texture_region(g_staging_read, 0, nullptr, reshade::api::map_access::read_only, &mapped_read)) {
        cmd_list->barrier(backbuffer, reshade::api::resource_usage::copy_source, reshade::api::resource_usage::present);
        return;
    }

    size_t plane_size = INFER_WIDTH * INFER_HEIGHT;
    std::vector<float> input_tensor(plane_size * 7, 0.0f);
    uint8_t* raw_src = static_cast<uint8_t*>(mapped_read.data);

    for (int y = 0; y < INFER_HEIGHT; ++y) {
        uint8_t* row = raw_src + (y * mapped_read.row_pitch);
        for (int x = 0; x < INFER_WIDTH; ++x) {
            int idx = y * INFER_WIDTH + x;
            input_tensor[0 * plane_size + idx] = row[x * 4 + 0] / 255.0f; // R
            input_tensor[1 * plane_size + idx] = row[x * 4 + 1] / 255.0f; // G
            input_tensor[2 * plane_size + idx] = row[x * 4 + 2] / 255.0f; // B
            input_tensor[3 * plane_size + idx] = 0.5f;                    // Depth
            input_tensor[4 * plane_size + idx] = 0.5f;                    // Normal X
            input_tensor[5 * plane_size + idx] = 0.5f;                    // Normal Y
            input_tensor[6 * plane_size + idx] = 1.0f;                    // Normal Z
        }
    }
    device->unmap_texture_region(g_staging_read, 0);

    // Step 3: Run the COMPLETE trained ONNX model
    std::vector<int64_t> input_dims = { 1, 7, INFER_HEIGHT, INFER_WIDTH };
    Ort::Value in_val = Ort::Value::CreateTensor<float>(static_cast<const OrtMemoryInfo*>(*g_memory_info), input_tensor.data(), input_tensor.size(), input_dims.data(), input_dims.size());

    const char* in_names[] = { "g_buffer" };
    const char* out_names[] = { "rendered_frame" };

    std::vector<Ort::Value> outputs;
    try {
        outputs = g_ort_session->Run(Ort::RunOptions{ nullptr }, in_names, &in_val, 1, out_names, 1);
    } catch (...) {
        cmd_list->barrier(backbuffer, reshade::api::resource_usage::copy_source, reshade::api::resource_usage::present);
        return;
    }

    if (outputs.empty()) {
        cmd_list->barrier(backbuffer, reshade::api::resource_usage::copy_source, reshade::api::resource_usage::present);
        return;
    }

    float* out_data = outputs[0].GetTensorMutableData<float>();

    // Step 4: Write output tensor back to the staging write texture
    reshade::api::subresource_data mapped_write;
    if (device->map_texture_region(g_staging_write, 0, nullptr, reshade::api::map_access::write_only, &mapped_write)) {
        uint8_t* raw_dst = static_cast<uint8_t*>(mapped_write.data);
        for (int y = 0; y < INFER_HEIGHT; ++y) {
            uint8_t* row = raw_dst + (y * mapped_write.row_pitch);
            for (int x = 0; x < INFER_WIDTH; ++x) {
                int idx = y * INFER_WIDTH + x;
                float r = std::clamp(out_data[0 * plane_size + idx] * g_ai_multiplier, 0.0f, 1.0f);
                float g = std::clamp(out_data[1 * plane_size + idx] * g_ai_multiplier, 0.0f, 1.0f);
                float b = std::clamp(out_data[2 * plane_size + idx] * g_ai_multiplier, 0.0f, 1.0f);

                row[x * 4 + 0] = static_cast<uint8_t>(r * 255.0f);
                row[x * 4 + 1] = static_cast<uint8_t>(g * 255.0f);
                row[x * 4 + 2] = static_cast<uint8_t>(b * 255.0f);
                row[x * 4 + 3] = 255;
            }
        }
        device->unmap_texture_region(g_staging_write, 0);

        // Step 5: Upscale/Blit neural output directly over the active backbuffer
        cmd_list->barrier(backbuffer, reshade::api::resource_usage::copy_source, reshade::api::resource_usage::copy_dest);
        cmd_list->copy_texture_region(g_staging_write, 0, &dst_box, backbuffer, 0, &src_box);
        cmd_list->barrier(backbuffer, reshade::api::resource_usage::copy_dest, reshade::api::resource_usage::present);
    } else {
        cmd_list->barrier(backbuffer, reshade::api::resource_usage::copy_source, reshade::api::resource_usage::present);
    }
}

extern "C" __declspec(dllexport) const char* NAME = "Neural Shader (HDR2SDR)";
extern "C" __declspec(dllexport) const char* DESCRIPTION = "Full Model DirectML Blit Hook";

BOOL APIENTRY DllMain(HMODULE hModule, DWORD fdwReason, LPVOID) {
    switch (fdwReason) {
        case DLL_PROCESS_ATTACH:
            g_module_handle = hModule;
            if (!reshade::register_addon(hModule)) return FALSE;
            reshade::register_event<reshade::addon_event::init_swapchain>(on_init_swapchain);
            reshade::register_event<reshade::addon_event::destroy_swapchain>(on_destroy_swapchain);
            reshade::register_event<reshade::addon_event::present>(on_present);
            break;
        case DLL_PROCESS_DETACH:
            reshade::unregister_event<reshade::addon_event::init_swapchain>(on_init_swapchain);
            reshade::unregister_event<reshade::addon_event::destroy_swapchain>(on_destroy_swapchain);
            reshade::unregister_event<reshade::addon_event::present>(on_present);
            reshade::unregister_addon(hModule);
            break;
    }
    return TRUE;
}
