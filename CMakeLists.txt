# Standalone DirectML D3D12 Benchmark Executable
add_executable(d3d12_benchmark d3d12_benchmark.cpp)

target_include_directories(d3d12_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(d3d12_benchmark PRIVATE
    ${ONNXRUNTIME_LIB}
    d3d12.lib
    dxgi.lib
)