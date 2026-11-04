cmake_minimum_required(VERSION 3.20)
project(neural_shader LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DUNICODE -D_UNICODE -DNOMINMAX -DWIN32_LEAN_AND_MEAN)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/reshade/include
    ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/include
)

add_library(neural_shader SHARED main.cpp)

set_target_properties(neural_shader PROPERTIES
    OUTPUT_NAME "neural_shader"
    SUFFIX ".addon64"
    PREFIX ""
)

target_link_libraries(neural_shader PRIVATE
    d3d12.lib
    dxgi.lib
)
