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

static reshade::api::resource g_staging_color_read = { 0 };
static reshade::api::resource g_staging_color_write = { 0 };

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

static void on_init_swapchain(reshade::api::swapchain* swapchain) {
    if (!swapchain || g_ort_session) return;
    try {
        if (!g_ort_env) {
            g_ort_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "HDR2SDR_NeuralShader");
        }

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Bind DirectML GPU Execution Provider
        OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0);

        std::wstring model_full_path = get_addon_directory() + L"neural_shader_gtx1650_sharp.onnx";
        g_ort_session = new Ort::Session(*g_ort_env, model_full_path.c_str(), session_options);

        if (!g_memory_info) {
            g_memory_info = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
        }

        log_debug("ONNX Model initialized for swapchain.");
        MessageBeep(MB_ICONASTERISK);
    } catch (const std::exception& e) {
        log_debug(std::string("Init Exception: ") + e.what());
        MessageBeep(MB_ICONHAND);
    } catch (...) {
        log_debug("Unknown Init Exception.");
        MessageBeep(MB_ICONHAND);
    }
}

static void on_destroy_swapchain(reshade::api::swapchain* swapchain) {
    if (!swapchain) return;
    reshade::api::device* device = swapchain->get_device();
    if (!device) return;

    if (g_staging_color_read.handle != 0) {
        device->destroy_resource(g_staging_color_read);
        g_staging_color_read = { 0 };
    }
    if (g_staging_color_write.handle != 0) {
        device->destroy_resource(g_staging_color_write);
        g_staging_color_write = { 0 };
    }
}

static void on_present(reshade::api::command_queue* queue, reshade::api::swapchain* swapchain, const reshade::api::rect*, const reshade::api::rect*, uint32_t, const reshade::api::rect*) {
    if (!queue || !swapchain) return;

    // Hotkey Controls
    if (GetAsyncKeyState(VK_F10) & 0x0001) {
        g_ai_enabled = !g_ai_enabled;
        MessageBeep(g_ai_enabled ? MB_OK : MB_ICONEXCLAMATION);
        log_debug(std::string("F10 Pressed: AI is ") + (g_ai_enabled ? "ON" : "OFF"));
    }
    if (GetAsyncKeyState(VK_F11) & 0x0001) {
        g_ai_multiplier = (std::min)(g_ai_multiplier + 0.5f, 6.0f);
        MessageBeep(MB_OK);
    }
    if (GetAsyncKeyState(VK_F9) & 0x0001) {
        g_ai_multiplier = (std::max)(g_ai_multiplier - 0.5f, 0.1f);
        MessageBeep(MB_OK);
    }

    if (!g_ai_enabled) return;

    // Lazy load session if not created during init
    if (!g_ort_session) {
        on_init_swapchain(swapchain);
        if (!g_ort_session) return;
    }

    reshade::api::device* device = queue->get_device();
    if (!device) return;

    reshade::api::resource backbuffer = swapchain->get_current_back_buffer();
    if (backbuffer.handle == 0) return;

    reshade::api::resource_desc color_desc = device->get_resource_desc(backbuffer);
    int64_t width = color_desc.texture.width;
    int64_t height = color_desc.texture.height;
    reshade::api::format fmt = color_desc.texture.format;

    if (width <= 0 || height <= 0 || width > 3840 || height > 2160) return;

    bool is_10bit = (fmt == reshade::api::format::r10g10b10a2_unorm || fmt == reshade::api::format::b10g10r10a2_unorm);
    bool is_bgra = (fmt == reshade::api::format::b8g8r8a8_unorm || fmt == reshade::api::format::b8g8r8a8_unorm_srgb || fmt == reshade::api::format::b10g10r10a2_unorm);

    // Staging resources allocation
    if (g_staging_color_read.handle == 0 || device->get_resource_desc(g_staging_color_read).texture.width != width) {
        if (g_staging_color_read.handle != 0) {
            device->destroy_resource(g_staging_color_read);
            device->destroy_resource(g_staging_color_write);
            g_staging_color_read = { 0 };
            g_staging_color_write = { 0 };
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

    reshade::api::command_list* cmd_list = queue->get_immediate_command_list();
    if (!cmd_list) return;

    // Copy active backbuffer to read staging buffer
    cmd_list->copy_resource(backbuffer, g_staging_color_read);
    queue->wait_idle();

    reshade::api::subresource_data mapped_color;
    if (!device->map_texture_region(g_staging_color_read, 0, nullptr, reshade::api::map_access::read_only, &mapped_color)) {
        return;
    }

    int64_t input_channels = 7;
    size_t plane_size = height * width;
    std::vector<float> input_tensor(plane_size * input_channels, 0.0f);
    uint8_t* color_data = static_cast<uint8_t*>(mapped_color.data);

    // Unpack color channels according to format
    for (int y = 0; y < height; ++y) {
        uint8_t* row = color_data + y * mapped_color.row_pitch;
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            float r = 0.0f, g = 0.0f, b = 0.0f;

            if (is_10bit) {
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
            input_tensor[3 * plane_size + idx] = 0.5f; // depth baseline
            input_tensor[4 * plane_size + idx] = 0.5f; // normal x
            input_tensor[5 * plane_size + idx] = 0.5f; // normal y
            input_tensor[6 * plane_size + idx] = 1.0f; // normal z
        }
    }
    device->unmap_texture_region(g_staging_color_read, 0);

    // Build tensor and execute DirectML session
    std::vector<int64_t> input_dims = { 1, input_channels, height, width };
    Ort::Value input_tensor_val = Ort::Value::CreateTensor<float>(static_cast<const OrtMemoryInfo*>(*g_memory_info), input_tensor.data(), input_tensor.size(), input_dims.data(), input_dims.size());

    const char* input_names[] = { "g_buffer" };
    const char* output_names[] = { "rendered_frame" };

    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = g_ort_session->Run(
            Ort::RunOptions{ nullptr },
            input_names, &input_tensor_val, 1,
            output_names, 1
        );
    } catch (...) {
        return;
    }

    if (output_tensors.empty()) return;
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    if (!output_data) return;

    // Map output staging buffer and write blended pixels
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

            float r_ai = std::clamp(output_data[0 * plane_size + idx], 0.0f, 1.0f);
            float g_ai = std::clamp(output_data[1 * plane_size + idx], 0.0f, 1.0f);
            float b_ai = std::clamp(output_data[2 * plane_size + idx], 0.0f, 1.0f);

            float r_final = std::clamp(r_orig + (r_ai - r_orig) * g_ai_multiplier, 0.0f, 1.0f);
            float g_final = std::clamp(g_orig + (g_ai - g_orig) * g_ai_multiplier, 0.0f, 1.0f);
            float b_final = std::clamp(b_orig + (b_ai - b_orig) * g_ai_multiplier, 0.0f, 1.0f);

            if (is_10bit) {
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

    // Copy modified frame directly onto active swapchain backbuffer
    cmd_list->copy_resource(g_staging_color_write, backbuffer);
}

extern "C" __declspec(dllexport) const char* NAME = "Neural Shader (HDR2SDR)";
extern "C" __declspec(dllexport) const char* DESCRIPTION = "DirectML HDR2SDR Present Hook";

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
