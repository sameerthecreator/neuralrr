name: Build Neural Shader and Benchmark

on:
  push:
    branches: [ "main", "master" ]
  pull_request:
    branches: [ "main", "master" ]
  workflow_dispatch:

jobs:
  build-windows:
    runs-on: windows-latest

    steps:
    - name: Checkout Repository
      uses: actions/checkout@v4

    - name: Setup MSVC Environment
      uses: ilammy/msvc-dev-cmd@v1

    - name: Configure CMake
      run: >
        cmake -B build
        -DCMAKE_BUILD_TYPE=Release
        -A x64

    - name: Build Targets
      run: cmake --build build --config Release --parallel

    - name: Prepare Artifact Folder
      shell: bash
      run: |
        mkdir -p dist
        find build/ -type f \( -name "*.addon64" -o -name "d3d12_benchmark.exe" -o -name "*.dll" \) -exec cp {} dist/ \; || true

    - name: Upload Build Artifacts
      uses: actions/upload-artifact@v4
      with:
        name: neural-shader-d3d12-benchmark
        path: dist/*
        if-no-files-found: error
