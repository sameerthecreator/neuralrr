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

        OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0);

        std::wstring model_full_path = get_addon_directory() + L"neural_shader_gtx1650_sharp.onnx";
        
        char path_a[MAX_PATH];
        WideCharToMultiByte(CP_UTF8, 0, model_full_path.c_str(), -1, path_a, MAX_PATH, NULL, NULL);
        log_debug(std::string("Attempting to load model from: ") + path_a);

        g_ort_session = new Ort::Session(*g_ort_env, model_full_path.c_str(), session_options);
        
        if (!g_memory_info) {
            g_memory_info = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
        }

        // Query model expected input shapes
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_inputs = g_ort_session->GetInputCount();
        log_debug("Model loaded successfully! Num inputs: " + std::to_string(num_inputs));

        for (size_t i = 0; i < num_inputs; ++i) {
            auto input_name = g_ort_session->GetInputNameAllocated(i, allocator);
            auto type_info = g_ort_session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            std::string shape_str = "Input " + std::string(input_name.get()) + " shape: [";
            for (auto dim : shape) shape_str += std::to_string(dim) + " ";
            shape_str += "]";
            log_debug(shape_str);
        }

        MessageBeep(MB_ICONASTERISK);
    } catch (const Ort::Exception& oe) {
        log_debug(std::string("ONNX Exception in init: ") + oe.what());
        MessageBeep(MB_ICONHAND);
    } catch (const std::exception& e) {
        log_debug(std::string("Standard Exception in init: ") + e.what());
        MessageBeep(MB_ICONHAND);
    } catch (...) {
        log_debug("Unknown Exception in init");
        MessageBeep(MB_ICONHAND);
    }
}

static void on_destroy_effect_runtime(reshade::api::effect_runtime* runtime) {
    reshade::api::device* device = runtime->get_device();
    if (g_staging_color_read.handle != 0) { device->destroy_resource(g_staging_color_read); g_staging_color_read = {0}; }
    if (g_staging_color_write.handle != 0) { device->destroy_resource(g_staging_color_write); g_staging_color_write = {0}; }
}

static void on_reshade_finish_effects(reshade::api::effect_runtime* runtime, reshade::api::command_list* cmd_list, reshade::api::resource_view rtv, reshade::api::resource_view) {
    if (GetAsyncKeyState(VK_F10) & 0x0001) {
        g_ai_enabled = !g_ai_enabled;
        MessageBeep(g_ai_enabled ? MB_OK : MB_ICONEXCLAMATION);
        log_debug(std::string("F10 Pressed: AI is now ") + (g_ai_enabled ? "ON" : "OFF"));
    }

    if (!g_ai_enabled || !g_ort_session) return;

    reshade::api::device* device = runtime->get_device();
    reshade::api::resource target_resource = device->get_resource_from_view(rtv);
    if (target_resource.handle == 0) return;

    reshade::api::resource_desc color_desc = device->get_resource_desc(target_resource);
    int64_t width = color_desc.texture.width;
    int64_t height = color_desc.texture.height;

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
    runtime->get_command_queue()->wait_idle();

    reshade::api::subresource_data mapped_color;
    if (!device->map_texture_region(g_staging_color_read, 0, nullptr, reshade::api::map_access::read_only, &mapped_color)) {
        log_debug("Failed to map staging color read texture!");
        return;
    }

    int64_t input_channels = 7;
    int64_t output_channels = 3;
    size_t plane_size = height * width;

    std::vector<float> input_tensor(plane_size * input_channels, 0.0f);
    std::vector<float> output_tensor(plane_size * output_channels, 0.0f);

    uint8_t* color_data = static_cast<uint8_t*>(mapped_color.data);

    // Unpack R10G10B10A2
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

    Ort::Value input_tensor_val = Ort::Value::CreateTensor<float>(static_cast<const OrtMemoryInfo*>(*g_memory_info), input_tensor.data(), input_tensor.size(), input_dims.data(), input_dims.size());
    Ort::Value output_tensor_val = Ort::Value::CreateTensor<float>(static_cast<const OrtMemoryInfo*>(*g_memory_info), output_tensor.data(), output_tensor.size(), output_dims.data(), output_dims.size());

    const char* input_names[] = {"g_buffer"};
    const char* output_names[] = {"rendered_frame"};

    try {
        g_ort_session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_val, 1, output_names, &output_tensor_val, 1);
    } catch (const Ort::Exception& oe) {
        static int error_count = 0;
        if (error_count++ < 3) {
            log_debug(std::string("ONNX Run Error: ") + oe.what());
        }
        return;
    } catch (...) {
        static int error_count2 = 0;
        if (error_count2++ < 3) {
            log_debug("ONNX Run Error: Unknown exception");
        }
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
