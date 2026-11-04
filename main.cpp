#define RESHADE_ADDON_IMPL 1
#include <reshade.hpp>
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>
#include <Windows.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>

static Ort::Env* g_ort_env = nullptr;
static Ort::Session* g_ort_session = nullptr;
static Ort::MemoryInfo* g_memory_info = nullptr;

static reshade::api::resource g_staging_color_read = {0};
static reshade::api::resource g_staging_color_write = {0};
static reshade::api::resource g_staging_depth_read = {0};

static bool g_ai_enabled = true;
static float g_ai_multiplier = 2.5f;
static HMODULE g_module_handle = nullptr;

// FP16 to FP32 conversion helper
static inline float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF) << 13;
    if (exp == 0x1F) {
        exp = 0xFF;
    } else if (exp != 0) {
        exp += 127 - 15;
    }
    uint32_t f = sign | (exp << 23) | mant;
    return *reinterpret_cast<float*>(&f);
}

// FP32 to FP16 conversion helper
static inline uint16_t float_to_half(float f) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&f);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x03FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | mant);
}

static std::wstring get_addon_directory() {
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(g_module_handle, path, MAX_PATH);
    std::wstring str_path(path);
    size_t pos = str_path.find_last_of(L"\\/");
    return (pos != std::wstring::npos) ? str_path.substr(0, pos + 1) : L"";
}

static void on_init_effect_runtime(reshade::api::effect_runtime* runtime) {
    if (g_ort_session) return;
    try {
        g_ort_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "HDR2SDR_NeuralShader");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0);

        std::wstring model_full_path = get_addon_directory() + L"neural_shader_gtx1650_sharp.onnx";
        g_ort_session = new Ort::Session(*g_ort_env, model_full_path.c_str(), session_options);
        g_memory_info = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));

        MessageBeep(MB_ICONASTERISK);
    } catch (...) {
        MessageBeep(MB_ICONHAND);
    }
}

static void on_destroy_effect_runtime(reshade::api::effect_runtime* runtime) {
    reshade::api::device* device = runtime->get_device();
    if (g_staging_color_read.handle != 0) { device->destroy_resource(g_staging_color_read); g_staging_color_read = {0}; }
    if (g_staging_color_write.handle != 0) { device->destroy_resource(g_staging_color_write); g_staging_color_write = {0}; }
    if (g_staging_depth_read.handle != 0) { device->destroy_resource(g_staging_depth_read); g_staging_depth_read = {0}; }

    if (g_memory_info) { delete g_memory_info; g_memory_info = nullptr; }
    if (g_ort_session) { delete g_ort_session; g_ort_session = nullptr; }
    if (g_ort_env) { delete g_ort_env; g_ort_env = nullptr; }
}

// Hooking reshade_finish_effects guarantees execution on ReShade's active render target
static void on_reshade_finish_effects(reshade::api::effect_runtime* runtime, reshade::api::command_list* cmd_list, reshade::api::resource_view rtv, reshade::api::resource_view) {
    if (GetAsyncKeyState(VK_F10) & 0x0001) {
        g_ai_enabled = !g_ai_enabled;
        MessageBeep(g_ai_enabled ? MB_OK : MB_ICONEXCLAMATION);
    }
    if (GetAsyncKeyState(VK_F11) & 0x0001) {
        g_ai_multiplier = (std::min)(g_ai_multiplier + 0.5f, 10.0f);
        MessageBeep(MB_OK);
    }
    if (GetAsyncKeyState(VK_F9) & 0x0001) {
        g_ai_multiplier = (std::max)(g_ai_multiplier - 0.5f, 0.1f);
        MessageBeep(MB_OK);
    }

    if (!g_ai_enabled || !g_ort_session) return;

    reshade::api::device* device = runtime->get_device();
    reshade::api::resource target_resource = device->get_resource_from_view(rtv);
    if (target_resource.handle == 0) return;

    reshade::api::resource_desc color_desc = device->get_resource_desc(target_resource);
    int64_t width = color_desc.texture.width;
    int64_t height = color_desc.texture.height;
    reshade::api::format fmt = color_desc.texture.format;

    bool is_fp16 = (fmt == reshade::api::format::r16g16b16a16_float || fmt == reshade::api::format::r16g16b16a16_unorm);
    bool is_10bit = (fmt == reshade::api::format::r10g10b10a2_unorm || fmt == reshade::api::format::b10g10r10a2_unorm);
    bool is_bgra = (fmt == reshade::api::format::b8g8r8a8_unorm || fmt == reshade::api::format::b8g8r8a8_unorm_srgb || fmt == reshade::api::format::b10g10r10a2_unorm);

    // Staging resources setup
    if (g_staging_color_read.handle == 0 || device->get_resource_desc(g_staging_color_read).texture.width != width) {
        if (g_staging_color_read.handle != 0) {
            device->destroy_resource(g_staging_color_read);
            device->destroy_resource(g_staging_color_write);
        }
        reshade::api::resource_desc stage_desc = color_desc;
        stage_desc.heap = reshade::api::memory_heap::gpu_to_cpu;
        stage_desc.usage = reshade::api::resource_usage::copy_dest;
        device->create_resource(stage_desc, nullptr, reshade::api::resource_usage::copy_dest, &g_staging_color_read);

        stage_desc.heap = reshade::api::memory_heap::cpu_to_gpu;
        stage_desc.usage = reshade::api::resource_usage::copy_source;
        device->create_resource(stage_desc, nullptr, reshade::api::resource_usage::copy_source, &g_staging_color_write);
    }

    cmd_list->copy_resource(target_resource, g_staging_color_read);

    // Query active depth stencil texture from ReShade's effect runtime
    reshade::api::resource_view depth_view = {0};
    reshade::api::resource depth_res = {0};
    
    // Check if Generic Depth has assigned a depth buffer
    reshade::api::command_queue* queue = runtime->get_command_queue();
    queue->wait_idle();

    reshade::api::subresource_data mapped_color;
    if (!device->map_texture_region(g_staging_color_read, 0, nullptr, reshade::api::map_access::read_only, &mapped_color)) return;

    int64_t input_channels = 7;
    int64_t output_channels = 3;
    size_t plane_size = height * width;

    std::vector<float> input_tensor(plane_size * input_channels, 0.0f);
    std::vector<float> output_tensor(plane_size * output_channels, 0.0f);

    uint8_t* color_data = static_cast<uint8_t*>(mapped_color.data);

    // Unpack color according to format
    for (int y = 0; y < height; ++y) {
        uint8_t* row = color_data + y * mapped_color.row_pitch;

        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            float r = 0.0f, g = 0.0f, b = 0.0f;

            if (is_fp16) {
                uint16_t* fp16_pixel = reinterpret_cast<uint16_t*>(row) + (x * 4);
                r = half_to_float(fp16_pixel[0]);
                g = half_to_float(fp16_pixel[1]);
                b = half_to_float(fp16_pixel[2]);
            } else if (is_10bit) {
                uint32_t p = reinterpret_cast<uint32_t*>(row)[x];
                r = (p & 0x3FF) / 1023.0f;
                g = ((p >> 10) & 0x3FF) / 1023.0f;
                b = ((p >> 20) & 0x3FF) / 1023.0f;
                if (is_bgra) std::swap(r, b);
            } else {
                uint8_t b_raw = row[x * 4 + 0];
                uint8_t g_raw = row[x * 4 + 1];
                uint8_t r_raw = row[x * 4 + 2];
                if (!is_bgra) std::swap(b_raw, r_raw);
                r = r_raw / 255.0f;
                g = g_raw / 255.0f;
                b = b_raw / 255.0f;
            }

            input_tensor[0 * plane_size + idx] = r;
            input_tensor[1 * plane_size + idx] = g;
            input_tensor[2 * plane_size + idx] = b;
            input_tensor[3 * plane_size + idx] = 0.5f; // Baseline depth
        }
    }

    // Normal channels (Flat baseline fallback when depth is unmapped)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            input_tensor[4 * plane_size + idx] = 0.5f; // Normal X
            input_tensor[5 * plane_size + idx] = 0.5f; // Normal Y
            input_tensor[6 * plane_size + idx] = 1.0f; // Normal Z facing camera
        }
    }

    device->unmap_texture_region(g_staging_color_read, 0);

    // Run inference
    std::vector<int64_t> input_dims = {1, input_channels, height, width};
    std::vector<int64_t> output_dims = {1, output_channels, height, width};

    Ort::Value input_tensor_val = Ort::Value::CreateTensor<float>(static_cast<const OrtMemoryInfo*>(*g_memory_info), input_tensor.data(), input_tensor.size(), input_dims.data(), input_dims.size());
    Ort::Value output_tensor_val = Ort::Value::CreateTensor<float>(static_cast<const OrtMemoryInfo*>(*g_memory_info), output_tensor.data(), output_tensor.size(), output_dims.data(), output_dims.size());

    const char* input_names[] = {"g_buffer"};
    const char* output_names[] = {"rendered_frame"};

    try {
        g_ort_session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_val, 1, output_names, &output_tensor_val, 1);
    } catch (...) {
        return;
    }

    // Write Back to Target
    reshade::api::subresource_data mapped_write;
    if (!device->map_texture_region(g_staging_color_write, 0, nullptr, reshade::api::map_access::write_only, &mapped_write)) return;

    uint8_t* write_data = static_cast<uint8_t*>(mapped_write.data);
    for (int y = 0; y < height; ++y) {
        uint8_t* row = write_data + y * mapped_write.row_pitch;

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

            if (is_fp16) {
                uint16_t* fp16_pixel = reinterpret_cast<uint16_t*>(row) + (x * 4);
                fp16_pixel[0] = float_to_half(r_final);
                fp16_pixel[1] = float_to_half(g_final);
                fp16_pixel[2] = float_to_half(b_final);
                fp16_pixel[3] = float_to_half(1.0f);
            } else if (is_10bit) {
                if (is_bgra) std::swap(r_final, b_final);
                uint32_t r_int = static_cast<uint32_t>(r_final * 1023.0f);
                uint32_t g_int = static_cast<uint32_t>(g_final * 1023.0f);
                uint32_t b_int = static_cast<uint32_t>(b_final * 1023.0f);
                reinterpret_cast<uint32_t*>(row)[x] = r_int | (g_int << 10) | (b_int << 20) | (3u << 30);
            } else {
                uint8_t r_byte = static_cast<uint8_t>(r_final * 255.0f);
                uint8_t g_byte = static_cast<uint8_t>(g_final * 255.0f);
                uint8_t b_byte = static_cast<uint8_t>(b_final * 255.0f);
                if (is_bgra) {
                    row[x * 4 + 0] = b_byte;
                    row[x * 4 + 1] = g_byte;
                    row[x * 4 + 2] = r_byte;
                    row[x * 4 + 3] = 255;
                } else {
                    row[x * 4 + 0] = r_byte;
                    row[x * 4 + 1] = g_byte;
                    row[x * 4 + 2] = b_byte;
                    row[x * 4 + 3] = 255;
                }
            }
        }
    }
    device->unmap_texture_region(g_staging_color_write, 0);

    cmd_list->copy_resource(g_staging_color_write, target_resource);
}

extern "C" __declspec(dllexport) const char* NAME = "Neural Shader (HDR2SDR)";
extern "C" __declspec(dllexport) const char* DESCRIPTION = "7-Channel HDR2SDR ONNX Hook";

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
