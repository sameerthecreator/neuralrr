#define RESHADE_ADDON_IMPL 1
#include <reshade.hpp>
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>
#include <Windows.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <fstream>

static Ort::Env* g_ort_env = nullptr;
static Ort::Session* g_ort_session = nullptr;
static Ort::MemoryInfo* g_memory_info = nullptr;

static reshade::api::resource g_staging_color_read = {0};
static reshade::api::resource g_staging_color_write = {0};

static bool g_ai_enabled = true;
static float g_ai_multiplier = 2.0f;
static HMODULE g_module_handle = nullptr;

static void log_debug(const std::string& msg) {
    std::ofstream ofs("neural_debug.log", std::ios::app);
    if (ofs.is_open()) {
        ofs << msg << std::endl;
    }
}

static std::wstring get_addon_directory() {
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(g_module_handle, path, MAX_PATH);
    std::wstring str_path(path);
    size_t pos = str_path.find_last_of(L"\\/");
    return (pos != std::wstring::npos) ? str_path.substr(0, pos + 1) : L"";
}

static void on_init_effect_runtime(reshade::api::effect_runtime* runtime) {
    if (!runtime) return;
    try {
        if (!g_ort_env) {
            g_ort_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "HDR2SDR_NeuralShader");
        }
        if (g_ort_session) {
            delete g_ort_session;
            g_ort_session = nullptr;
        }

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Attach DirectML execution provider
        OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0);

        std::wstring model_full_path = get_addon_directory() + L"neural_shader_gtx1650_sharp.onnx";
        
        g_ort_session = new Ort::Session(*g_ort_env, model_full_path.c_str(), session_options);
        
        if (!g_memory_info) {
            g_memory_info = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
        }

        log_debug("ONNX Model and DirectML loaded successfully!");
        MessageBeep(MB_ICONASTERISK);
    } catch (const std::exception& e) {
        log_debug(std::string("Init Exception: ") + e.what());
        MessageBeep(MB_ICONHAND);
    } catch (...) {
        log_debug("Unknown Init Exception.");
        MessageBeep(MB_ICONHAND);
    }
}

static void on_destroy_effect_runtime(reshade::api::effect_runtime* runtime) {
    if (!runtime) return;
    reshade::api::device* device = runtime->get_device();
    if (!device) return;

    if (g_staging_color_read.handle != 0) { 
        device->destroy_resource(g_staging_color_read); 
        g_staging_color_read = {0}; 
    }
    if (g_staging_color_write.handle != 0) { 
        device->destroy_resource(g_staging_color_write); 
        g_staging_color_write = {0}; 
    }
}

static void on_reshade_finish_effects(reshade::api::effect_runtime* runtime, reshade::api::command_list* cmd_list, reshade::api::resource_view rtv, reshade::api::resource_view) {
    // Null safety guard against 0x0 access violation
    if (!runtime || !cmd_list || rtv.handle == 0 || !g_ort_session || !g_memory_info) {
        return;
    }

    if (GetAsyncKeyState(VK_F10) & 0x0001) {
        g_ai_enabled = !g_ai_enabled;
        MessageBeep(g_ai_enabled ? MB_OK : MB_ICONEXCLAMATION);
    }
    if (GetAsyncKeyState(VK_F11) & 0x0001) {
        g_ai_multiplier = (std::min)(g_ai_multiplier + 0.5f, 5.0f);
        MessageBeep(MB_OK);
    }
    if (GetAsyncKeyState(VK_F9) & 0x0001) {
        g_ai_multiplier = (std::max)(g_ai_multiplier - 0.5f, 0.1f);
        MessageBeep(MB_OK);
    }

    if (!g_ai_enabled) return;

    reshade::api::device* device = runtime->get_device();
    if (!device) return;

    reshade::api::resource target_resource = device->get_resource_from_view(rtv);
    if (target_resource.handle == 0) return;

    reshade::api::resource_desc color_desc = device->get_resource_desc(target_resource);
    int64_t width = color_desc.texture.width;
    int64_t height = color_desc.texture.height;

    // Validate resource bounds
    if (width <= 0 || height <= 0 || width > 3840 || height > 2160) return;

    // Allocate or reallocate staging buffers
    if (g_staging_color_read.handle == 0 || device->get_resource_desc(g_staging_color_read).texture.width != width) {
        if (g_staging_color_read.handle != 0) {
            device->destroy_resource(g_staging_color_read);
            device->destroy_resource(g_staging_color_write);
            g_staging_color_read = {0};
            g_staging_color_write = {0};
        }
        reshade::api::resource_desc stage_desc = color_desc;
        stage_desc.heap = reshade::api::memory_heap::gpu_to_cpu;
        stage_desc.usage = reshade::api::resource_usage::copy_dest;
        device->create_resource(stage_desc, nullptr, reshade::api::resource_usage::copy_dest, &g_staging_color_read);

        stage_desc.heap = reshade::api::memory_heap::cpu_to_gpu;
        stage_desc.usage = reshade::api::resource_usage::copy_source;
        device->create_resource(stage_desc, nullptr, reshade::api::resource_usage::copy_source, &g_staging_color_write);
    }

    if (g_staging_color_read.handle == 0 || g_staging_color_write.handle == 0) return;

    cmd_list->copy_resource(target_resource, g_staging_color_read);
    
    reshade::api::command_queue* queue = runtime->get_command_queue();
    if (!queue) return;
    queue->wait_idle();

    reshade::api::subresource_data mapped_color;
    if (!device->map_texture_region(g_staging_color_read, 0, nullptr, reshade::api::map_access::read_only, &mapped_color)) {
        return;
    }

    int64_t input_channels = 7;
    int64_t output_channels = 3;
    size_t plane_size = height * width;

    std::vector<float> input_tensor(plane_size * input_channels, 0.0f);
    std::vector<float> output_tensor(plane_size * output_channels, 0.0f);

    uint8_t* color_data = static_cast<uint8_t*>(mapped_color.data);

    // Unpack R10G10B10A2 format
    for (int y = 0; y < height; ++y) {
        uint32_t* row = reinterpret_cast<uint32_t*>(color_data + y * mapped_color.row_pitch);
        for (int x = 0; x < width; ++x) {
            uint32_t p = row[x];
            int idx = y * width + x;
            input_tensor[0 * plane_size + idx] = (p & 0x3FF) / 1023.0f;
            input_tensor[1 * plane_size + idx] = ((p >> 10) & 0x3FF) / 1023.0f;
            input_tensor[2 * plane_size + idx] = ((p >> 20) & 0x3FF) / 1023.0f;
            input_tensor[3 * plane_size + idx] = 0.5f;
            input_tensor[4 * plane_size + idx] = 0.5f;
            input_tensor[5 * plane_size + idx] = 0.5f;
            input_tensor[6 * plane_size + idx] = 1.0f;
        }
    }
    device->unmap_texture_region(g_staging_color_read, 0);

    std::vector<int64_t> input_dims = {1, input_channels, height, width};
    std::vector<int64_t> output_dims = {1, output_channels, height, width};

    try {
        Ort::Value input_tensor_val = Ort::Value::CreateTensor<float>(static_cast<const OrtMemoryInfo*>(*g_memory_info), input_tensor.data(), input_tensor.size(), input_dims.data(), input_dims.size());
        Ort::Value output_tensor_val = Ort::Value::CreateTensor<float>(static_cast<const OrtMemoryInfo*>(*g_memory_info), output_tensor.data(), output_tensor.size(), output_dims.data(), output_dims.size());

        const char* input_names[] = {"g_buffer"};
        const char* output_names[] = {"rendered_frame"};

        g_ort_session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_val, 1, output_names, &output_tensor_val, 1);
    } catch (...) {
        return;
    }

    reshade::api::subresource_data mapped_write;
    if (!device->map_texture_region(g_staging_color_write, 0, nullptr, reshade::api::map_access::write_only, &mapped_write)) return;

    uint8_t* write_data = static_cast<uint8_t*>(mapped_write.data);
    for (int y = 0; y < height; ++y) {
        uint32_t* row = reinterpret_cast<uint32_t*>(write_data + y * mapped_write.row_pitch);
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;

            float r_orig = input_tensor[0 * plane_size + idx];
            float g_orig = input_tensor[1 * plane_size + idx];
            float b_orig = input_tensor[2 * plane_size + idx];

            float r_ai = std::clamp(output_tensor[0 * plane_size + idx], 0.0f, 1.0f);
            float g_ai = std::clamp(output_tensor[1 * plane_size + idx], 0.0f, 1.0f);
            float b_ai = std::clamp(output_tensor[2 * plane_size + idx], 0.0f, 1.0f);

            float r_final = std::clamp(r_orig + (r_ai - r_orig) * g_ai_multiplier, 0.0f, 1.0f);
            float g_final = std::clamp(g_orig + (g_ai - g_orig) * g_ai_multiplier, 0.0f, 1.0f);
            float b_final = std::clamp(b_orig + (b_ai - b_orig) * g_ai_multiplier, 0.0f, 1.0f);

            uint32_t r_int = static_cast<uint32_t>(r_final * 1023.0f);
            uint32_t g_int = static_cast<uint32_t>(g_final * 1023.0f);
            uint32_t b_int = static_cast<uint32_t>(b_final * 1023.0f);
            row[x] = r_int | (g_int << 10) | (b_int << 20) | (3u << 30);
        }
    }
    device->unmap_texture_region(g_staging_color_write, 0);

    cmd_list->copy_resource(g_staging_color_write, target_resource);
}

extern "C" __declspec(dllexport) const char* NAME = "Neural Shader (HDR2SDR)";
extern "C" __declspec(dllexport) const char* DESCRIPTION = "DirectML 7-Channel HDR2SDR";

BOOL APIENTRY DllMain(HMODULE hModule, DWORD fdwReason, LPVOID) {
    switch (fdwReason) {
        case DLL_PROCESS_ATTACH:
            g_module_handle = hModule;
            if (!reshade::register_addon(hModule)) return FALSE;
            reshade::register_event<reshade::addon_event::init_effect_runtime>(on_init_effect_runtime);
            reshade::register_event<reshade::addon_event::destroy_effect_runtime>(on_destroy_effect_runtime);
            reshade::register_event<reshade::addon_event::reshade_finish_effects>(on_reshade_finish_effects);
            break;
        case DLL_PROCESS_DETACH:
            reshade::unregister_event<reshade::addon_event::init_effect_runtime>(on_init_effect_runtime);
            reshade::unregister_event<reshade::addon_event::destroy_effect_runtime>(on_destroy_effect_runtime);
            reshade::unregister_event<reshade::addon_event::reshade_finish_effects>(on_reshade_finish_effects);
            reshade::unregister_addon(hModule);
            break;
    }
    return TRUE;
}
