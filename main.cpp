#define RESHADE_ADDON_IMPL 1
#include <reshade.hpp>
#include <onnxruntime_cxx_api.h>
#include <Windows.h>
#include <vector>

static Ort::Env* g_ort_env = nullptr;
static Ort::Session* g_ort_session = nullptr;
static Ort::MemoryInfo* g_memory_info = nullptr;

static void on_init_effect_runtime(reshade::api::effect_runtime* runtime) {
    if (g_ort_session) return;
    try {
        g_ort_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "NeuralShader");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4); 
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        const wchar_t* model_path = L"neural_shader_gtx1650_sharp.onnx";
        g_ort_session = new Ort::Session(*g_ort_env, model_path, session_options);
        g_memory_info = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
    }
    catch (...) {}
}

static void on_destroy_effect_runtime(reshade::api::effect_runtime* runtime) {
    if (g_memory_info) { delete g_memory_info; g_memory_info = nullptr; }
    if (g_ort_session) { delete g_ort_session; g_ort_session = nullptr; }
    if (g_ort_env) { delete g_ort_env; g_ort_env = nullptr; }
}

static void on_present(reshade::api::command_queue* queue, reshade::api::swapchain* swapchain, const reshade::api::rect*, const reshade::api::rect*, uint32_t, const reshade::api::rect*) {
    if (!g_ort_session) return;

    reshade::api::device* device = queue->get_device();
    reshade::api::resource backbuffer = swapchain->get_current_back_buffer();
    reshade::api::resource_desc desc = device->get_resource_desc(backbuffer);
    
    int64_t width = desc.texture.width;
    int64_t height = desc.texture.height;
    int64_t batch_size = 1;
    
    // Model expects 7 channels (RGB + padded G-buffer data)
    int64_t input_channels = 7; 
    // Model outputs 3 channels (RGB)
    int64_t output_channels = 3; 

    // Allocate continuous memory for NCHW tensors
    size_t input_tensor_size = batch_size * input_channels * height * width;
    size_t output_tensor_size = batch_size * output_channels * height * width;
    
    std::vector<float> input_tensor_values(input_tensor_size, 0.0f); // Pre-filled with 0.0f to pad missing channels
    std::vector<float> output_tensor_values(output_tensor_size, 0.0f);

    // --- GPU TO CPU COPY WOULD GO HERE ---
    // (In a full implementation, you map a staging texture here, extract the RGB pixels, 
    // normalize them to 0.0f-1.0f, and write them into input_tensor_values[0], [1], and [2] for R, G, and B).
    
    // Define the exact shapes from the Netron graph
    std::vector<int64_t> input_node_dims = {batch_size, input_channels, height, width};
    std::vector<int64_t> output_node_dims = {batch_size, output_channels, height, width};

    // Create ONNX Tensors wrapping our C++ vectors
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        *g_memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), input_node_dims.size());
        
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
        *g_memory_info, output_tensor_values.data(), output_tensor_size, output_node_dims.data(), output_node_dims.size());

    // Exact node names from Netron
    const char* input_node_names[] = {"g_buffer"};
    const char* output_node_names[] = {"rendered_frame"};

    // Execute the neural network frame prediction
    try {
        g_ort_session->Run(Ort::RunOptions{nullptr}, 
                           input_node_names, &input_tensor, 1, 
                           output_node_names, &output_tensor, 1);
    } 
    catch (...) {
        // Suppress inference crashes mid-frame
    }

    // --- CPU TO GPU COPY WOULD GO HERE ---
    // (Map the output staging texture, convert output_tensor_values from 0.0f-1.0f back to 0-255 RGBA, 
    // and queue a texture copy back to the active swapchain backbuffer).
}

// Export basic addon metadata
extern "C" __declspec(dllexport) const char* NAME = "Neural Shader";
extern "C" __declspec(dllexport) const char* DESCRIPTION = "ONNX Neural Shader Hook";

BOOL APIENTRY DllMain(HMODULE hModule, DWORD fdwReason, LPVOID) {
    switch (fdwReason) {
        case DLL_PROCESS_ATTACH:
            if (!reshade::register_addon(hModule))
                return FALSE;
            
            reshade::register_event<reshade::addon_event::init_effect_runtime>(on_init_effect_runtime);
            reshade::register_event<reshade::addon_event::destroy_effect_runtime>(on_destroy_effect_runtime);
            reshade::register_event<reshade::addon_event::present>(on_present);
            break;
            
        case DLL_PROCESS_DETACH:
            reshade::unregister_event<reshade::addon_event::init_effect_runtime>(on_init_effect_runtime);
            reshade::unregister_event<reshade::addon_event::destroy_effect_runtime>(on_destroy_effect_runtime);
            reshade::unregister_event<reshade::addon_event::present>(on_present);
            reshade::unregister_addon(hModule);
            break;
    }
    return TRUE;
}
