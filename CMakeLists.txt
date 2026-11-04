cmake_minimum_required(VERSION 3.20)
project(NeuralShaderDirectML LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# -------------------------------------------------------------
# 1. Fetch Official ReShade API Headers
# -------------------------------------------------------------
FetchContent_Declare(
    reshade
    GIT_REPOSITORY https://github.com/crosire/reshade.git
    GIT_TAG        v6.3.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(reshade)

# -------------------------------------------------------------
# 2. Download ONNX Runtime with DirectML GPU Support
# -------------------------------------------------------------
set(ORT_VERSION "1.19.2")
set(ORT_ZIP_NAME "Microsoft.ML.OnnxRuntime.DirectML.${ORT_VERSION}.zip")
set(ORT_URL "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/${ORT_VERSION}")
set(ORT_EXTRACT_DIR "${CMAKE_BINARY_DIR}/onnxruntime_dml")

if(NOT EXISTS "${ORT_EXTRACT_DIR}")
    file(DOWNLOAD "${ORT_URL}" "${CMAKE_BINARY_DIR}/${ORT_ZIP_NAME}" SHOW_PROGRESS)
    file(ARCHIVE_EXTRACT INPUT "${CMAKE_BINARY_DIR}/${ORT_ZIP_NAME}" DESTINATION "${ORT_EXTRACT_DIR}")
endif()

set(ORT_INCLUDE_DIR "${ORT_EXTRACT_DIR}/build/native/include")
set(ORT_LIB_DIR "${ORT_EXTRACT_DIR}/runtimes/win-x64/native")

add_library(onnxruntime_dml_lib INTERFACE)
target_include_directories(onnxruntime_dml_lib INTERFACE ${ORT_INCLUDE_DIR})
target_link_libraries(onnxruntime_dml_lib INTERFACE "${ORT_LIB_DIR}/onnxruntime.lib")

# -------------------------------------------------------------
# 3. Target 1: ReShade Addon (neural_shader.addon64)
# -------------------------------------------------------------
add_library(neural_shader SHARED main.cpp)

target_include_directories(neural_shader PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${reshade_SOURCE_DIR}/include
)

target_link_libraries(neural_shader PRIVATE
    onnxruntime_dml_lib
    d3d12.lib
    dxgi.lib
)

set_target_properties(neural_shader PROPERTIES
    OUTPUT_NAME "neural_shader"
    SUFFIX ".addon64"
    PREFIX ""
)

# -------------------------------------------------------------
# 4. Target 2: Standalone D3D12 GPU Benchmark (d3d12_benchmark.exe)
# -------------------------------------------------------------
add_executable(d3d12_benchmark d3d12_benchmark.cpp)

target_include_directories(d3d12_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(d3d12_benchmark PRIVATE
    onnxruntime_dml_lib
    "${ORT_LIB_DIR}/DirectML.lib"
    d3d12.lib
    dxgi.lib
)
