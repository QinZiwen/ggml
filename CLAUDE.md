# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release -j 8
```

启用 GPU 后端（可选）：
```bash
cmake .. -DGGML_CUDA=ON       # NVIDIA CUDA
cmake .. -DGGML_METAL=ON      # Apple Metal（macOS 默认启用）
cmake .. -DGGML_VULKAN=ON     # Vulkan
cmake .. -DGGML_HIP=ON        # AMD ROCm/HIP
```

动态加载后端（每个后端编译为独立 .so/.dll）：
```bash
cmake .. -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON
```

## 运行测试

```bash
# 在 build 目录中运行所有测试
ctest --test-dir build

# 运行单个测试
./build/bin/test-backend-ops
./build/bin/test-quantize-fns

# 针对特定后端运行 test-backend-ops
./build/bin/test-backend-ops -b CUDA0
```

## 代码架构

### 核心层次结构

**`ggml-base`**（静态基础库）：
- [src/ggml.c](src/ggml.c) / [src/ggml.cpp](src/ggml.cpp)：张量操作、计算图构建与执行调度
- [src/ggml-backend.cpp](src/ggml-backend.cpp)：后端抽象接口（buffer 分配、张量 I/O、图执行）
- [src/ggml-alloc.c](src/ggml-alloc.c)：张量分配器（`ggml_tallocr`）和图分配器（`ggml_gallocr`）
- [src/ggml-quants.c](src/ggml-quants.c)：量化/反量化实现（Q4_0, Q8_0 等格式）
- [src/gguf.cpp](src/gguf.cpp)：GGUF 二进制模型文件格式的读写
- [src/ggml-opt.cpp](src/ggml-opt.cpp)：ADAM / L-BFGS 优化器
- [src/ggml-threading.cpp](src/ggml-threading.cpp)：线程池

**`ggml`**（主库，链接 ggml-base 和所有后端）：
- [src/ggml-backend-dl.cpp](src/ggml-backend-dl.cpp)：动态加载后端（`GGML_BACKEND_DL` 模式）
- [src/ggml-backend-reg.cpp](src/ggml-backend-reg.cpp)：后端注册表（静态链接模式）

### 后端插件（各自位于 `src/ggml-<name>/`）

每个后端实现 [src/ggml-backend-impl.h](src/ggml-backend-impl.h) 中定义的接口（vtable 结构体）：
- `ggml-cpu`：CPU 后端，含 SIMD（AVX2/AVX512/NEON/SVE/AMX）、KleidiAI、llamafile 优化路径
- `ggml-cuda`：NVIDIA CUDA，同时支持 MUSA（摩尔线程）
- `ggml-metal`：Apple Metal（macOS/iOS）
- `ggml-vulkan`：跨平台 Vulkan
- `ggml-hip`：AMD ROCm/HIP
- `ggml-opencl`：OpenCL（含 Adreno GPU 专用 kernel）
- `ggml-sycl`：Intel oneAPI SYCL
- `ggml-rpc`：通过网络将计算分发到远程 ggml 服务器
- `ggml-blas`：BLAS 矩阵乘加速（OpenBLAS / Apple Accelerate）
- `ggml-cann`：华为 CANN（昇腾 NPU）
- `ggml-hexagon`：Qualcomm Hexagon DSP/HTP
- `ggml-openvino`：Intel OpenVINO
- `ggml-virtgpu`：VirtGPU/Virglrenderer API Remoting
- `ggml-webgpu`：WebGPU（Emscripten）

### 关键头文件（公共 API）

- [include/ggml.h](include/ggml.h)：张量类型、操作（ggml_add/mul/matmul 等）、计算图（ggml_cgraph）
- [include/ggml-backend.h](include/ggml-backend.h)：后端/设备/buffer 的公共接口
- [include/ggml-alloc.h](include/ggml-alloc.h)：`ggml_tallocr`（张量分配器）与 `ggml_gallocr`（图分配器）
- [include/gguf.h](include/gguf.h)：GGUF 文件读写 API
- [include/ggml-opt.h](include/ggml-opt.h)：优化器 API
- [src/ggml-backend-impl.h](src/ggml-backend-impl.h)：**后端开发者**实现新后端所需的内部 vtable 接口

### 核心编程模型

1. 用 `ggml_init()` 创建上下文，分配内存池（运行时零分配）
2. 用张量操作函数构建计算图（懒求值，只建图不计算）
3. 用 `ggml_gallocr` 为图中所有张量分配 buffer（可复用 buffer）
4. 调用 `ggml_backend_graph_compute()` 在指定后端执行图

### GGUF 文件格式

GGUF（GGerganov Unified Format）是 ggml 的二进制模型文件格式，包含：KV 元数据 + 张量元信息 + 对齐的张量数据 blob。版本当前为 v3，默认 32 字节对齐。

## 标准规范

- C 标准：C11；C++ 标准：C++17
- 编译时默认开启全部警告（`-Wall -Wextra -Wpedantic`）
- 量化类型命名约定：`Q<bits>_<variant>`（如 `Q4_0`、`Q8_0`）
- 新后端须实现 [src/ggml-backend-impl.h](src/ggml-backend-impl.h) 中的 vtable，并通过 `ggml_add_backend()` CMake 函数注册
