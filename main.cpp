#define RESHADE_ADDON_IMPL 1
#include <reshade.hpp>
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>
#include <Windows.h>
#include <vector>
#include <algorithm>

static Ort::Env* g_ort_env = nullptr;
static Ort::Session* g_ort_session = nullptr;
static Ort::MemoryInfo* g_memory_info = nullptr;

static reshade::api::resource g_staging_read = {0};
static reshade::api::resource g_staging_write = {0};

static bool g_ai_enabled = true;
static float g_ai_multiplier = 2.0f;

static void on_init_effect_runtime(reshade::api::effect_runtime* runtime) {
    if (g_ort_session) return;
    try {
        g_ort_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "NeuralShader");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // DirectML GPU acceleration
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0));

        const wchar_t* model_path = L"neural_shader_gtx1650_sharp.onnx";
        g_ort_session = new Ort::Session(*g_ort_env, model_path, session_options);
        g_memory_info = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
    }
    catch (...) {}
}

static void on_destroy_effect_runtime(reshade::api::effect_runtime* runtime) {
    reshade::api::device* device = runtime->get_device();
    if (g_staging_read.handle != 0) { device->destroy_resource(g_staging_read); g_staging_read = {0}; }
    if (g_staging_write.handle != 0) { device->destroy_resource(g_staging_write); g_staging_write = {0}; }
    
    if (g_memory_info) { delete g_memory_info; g_memory_info = nullptr; }
    if (g_ort_session) { delete g_ort_session; g_ort_session = nullptr; }
    if (g_ort_env) { delete g_ort_env; g_ort_env = nullptr; }
}

static void on_present(reshade::api::command_queue* queue, reshade::api::swapchain* swapchain, const reshade::api::rect*, const reshade::api::rect*, uint32_t, const reshade::api::rect*) {
    if (!g_ort_session) return;

    // Controls: F10 = Toggle ON/OFF | F11 = Multiplier Up | F9 = Multiplier Down
    if (GetAsyncKeyState(VK_F10) & 0x0001) {
        g_ai_enabled = !g_ai_enabled;
    }
    if (GetAsyncKeyState(VK_F11) & 0x0001) {
        g_ai_multiplier = (std::min)(g_ai_multiplier + 0.5f, 10.0f);
    }
    if (GetAsyncKeyState(VK_F9) & 0x0001) {
        g_ai_multiplier = (std::max)(g_ai_multiplier - 0.5f, 0.1f);
    }

    if (!g_ai_enabled) return;

    reshade::api::device* device = queue->get_device();
    reshade::api::command_list* cmd_list = queue->get_immediate_command_list();
    
    reshade::api::resource backbuffer = swapchain->get_current_back_buffer();
    reshade::api::resource_desc desc = device->get_resource_desc(backbuffer);
    
    int64_t width = desc.texture.width;
    int64_t height = desc.texture.height;

    if (g_staging_read.handle == 0) {
        reshade::api::resource_desc stage_desc = desc;
        stage_desc.heap = reshade::api::memory_heap::gpu_to_cpu;
        stage_desc.usage = reshade::api::resource_usage::copy_dest;
        device->create_resource(stage_desc, nullptr, reshade::api::resource_usage::copy_dest, &g_staging_read);
        
        stage_desc.heap = reshade::api::memory_heap::cpu_to_gpu;
        stage_desc.usage = reshade::api::resource_usage::copy_source;
        device->create_resource(stage_desc, nullptr, reshade::api::resource_usage::copy_source, &g_staging_write);
    }

    cmd_list->copy_resource(backbuffer, g_staging_read);
    queue->wait_idle();

    reshade::api::subresource_data mapped_read;
    if (!device->map_texture_region(g_staging_read, 0, nullptr, reshade::api::map_access::read_only, &mapped_read)) return;

    int64_t input_channels = 7;
    int64_t output_channels = 3;
    size_t plane_size = height * width;
    
    std::vector<float> input_tensor_values(plane_size * input_channels, 0.0f);
    std::vector<float> output_tensor_values(plane_size * output_channels, 0.0f);

    uint8_t* read_data = static_cast<uint8_t*>(mapped_read.data);
    for (int y = 0; y < height; ++y) {
        uint32_t* row = reinterpret_cast<uint32_t*>(read_data + y * mapped_read.row_pitch);
        for (int x = 0; x < width; ++x) {
            uint32_t pixel = row[x];
            int idx = y * width + x;
            
            input_tensor_values[0 * plane_size + idx] = (pixel & 0x3FF) / 1023.0f;
            input_tensor_values[1 * plane_size + idx] = ((pixel >> 10) & 0x3FF) / 1023.0f;
            input_tensor_values[2 * plane_size + idx] = ((pixel >> 20) & 0x3FF) / 1023.0f;
        }
    }
    device->unmap_texture_region(g_staging_read, 0);

    std::vector<int64_t> input_dims = {1, input_channels, height, width};
    std::vector<int64_t> output_dims = {1, output_channels, height, width};

    Ort::Value input_tensor = Ort::Value::CreateTensor
