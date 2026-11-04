name: Build Neural Shader Addon

on:
  push:
    branches: [ main, master ]
  workflow_dispatch:

jobs:
  build:
    runs-on: windows-latest

    steps:
    - name: Checkout Code
      uses: actions/checkout@v4
      with:
        submodules: recursive

    - name: Setup MSVC
      uses: ilammy/msvc-dev-cmd@v1

    - name: Download ONNX Runtime & DirectML Headers/Libs
      shell: pwsh
      run: |
        Invoke-WebRequest -Uri "https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-win-x64-1.17.1.zip" -OutFile "ort.zip"
        Expand-Archive -Path "ort.zip" -DestinationPath "ort_extract"
        New-Item -ItemType Directory -Force -Path "onnxruntime/include"
        New-Item -ItemType Directory -Force -Path "onnxruntime/lib"
        Copy-Item "ort_extract/onnxruntime-win-x64-1.17.1/include/*" -Destination "onnxruntime/include" -Recurse
        Copy-Item "ort_extract/onnxruntime-win-x64-1.17.1/lib/*" -Destination "onnxruntime/lib" -Recurse

        Invoke-WebRequest -Uri "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.14.0" -OutFile "dml.zip"
        Expand-Archive -Path "dml.zip" -DestinationPath "dml_extract"
        New-Item -ItemType Directory -Force -Path "directml/include"
        New-Item -ItemType Directory -Force -Path "directml/lib"
        Copy-Item "dml_extract/include/*" -Destination "directml/include" -Recurse
        Copy-Item "dml_extract/bin/x64-win/DirectML.lib" -Destination "directml/lib"
        Copy-Item "dml_extract/bin/x64-win/DirectML.dll" -Destination "directml/lib"

    - name: Configure CMake
      run: cmake -B build -DCMAKE_BUILD_TYPE=Release

    - name: Build Project
      run: cmake --build build --config Release

    - name: Upload Addon Artifact
      uses: actions/upload-artifact@v4
      with:
        name: neural_shader_addon
        path: |
          build/Release/neural_shader.addon64
          onnxruntime/lib/onnxruntime.dll
          directml/lib/DirectML.dll
