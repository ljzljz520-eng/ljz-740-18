# Stable Diffusion Go Binding (PureGo)

使用 [purego](https://github.com/ebitengine/purego) 实现的 `stable-diffusion.cpp` 纯 Go 语言绑定。  
**无需 CGO**，支持 macOS、Linux 和 Windows 三大平台。

## ✨ 核心亮点

| 特性 | 说明 |
|------|------|
| **零 CGO** | 纯 Go 实现，通过 `purego` 直接调用 C 动态库 |
| **三平台** | macOS `.dylib` / Linux `.so` / Windows `.dll` 均支持 |
| **功能完整** | 1:1 覆盖 `stable-diffusion.h` 中全部 38 个导出函数 |
| **生产就绪** | 示例服务含并发控制、超时处理、PNG 编码、优雅退出 |

---

## 📦 模块路径

```
github.com/example/stablediffusion           # 高层 API
github.com/example/stablediffusion/bindings  # 底层 purego 绑定
```

---

## 🚀 快速上手

### 1. 运行单元测试（无需 C++ 环境）

测试会自动使用 Mock 实现，无需动态库：

```bash
go test -v ./...
```

### 2. macOS / Linux 原生构建

**A. 编译动态库**
```bash
cd stable-diffusion.cpp
mkdir -p build && cd build
cmake .. -DSD_BUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j
```

**B. 启动服务**
```bash
cd ../..
# macOS
export SD_LIB_PATH=$(pwd)/stable-diffusion.cpp/build/bin/libstable-diffusion.dylib
# Linux
# export SD_LIB_PATH=$(pwd)/stable-diffusion.cpp/build/bin/libstable-diffusion.so

go run examples/server/server.go
```

### 3. Windows 原生构建

**A. 前置条件**
- [CMake](https://cmake.org/download/)
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (含 MSVC)

**B. 编译 DLL**
```powershell
cd stable-diffusion.cpp
mkdir build; cd build
cmake .. -DSD_BUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```
编译产物路径：`stable-diffusion.cpp/build/bin/Release/stable-diffusion.dll`

> **purego 在 Windows 上的工作原理**：`purego` 底层在 Windows 上使用 `syscall.LoadLibrary` / `GetProcAddress` 加载 DLL，与 Unix 上的 `dlopen`/`dlsym` 等价，完全兼容。

**C. 启动服务**
```powershell
cd ../..
$env:SD_LIB_PATH="$(Get-Location)\stable-diffusion.cpp\build\bin\Release\stable-diffusion.dll"
go run examples/server/server.go
```

### 4. Docker 部署（Linux 环境）

```bash
docker compose up --build -d
curl http://localhost:8080/health
```

> **macOS Docker 说明**：Dockerfile 使用 `debian:bookworm-slim` 基础镜像，面向 Linux 服务器部署。在 macOS 上 Docker Desktop 通过虚拟机运行，无法透传 Metal GPU，性能大幅下降。macOS 推荐使用上述 **原生构建** 方式。

---

## 📋 API 概览

### 基础用法

```go
package main

import (
    "log"
    "github.com/example/stablediffusion"
    "github.com/example/stablediffusion/bindings"
)

func main() {
    // 创建上下文
    options := stablediffusion.DefaultContextOptions("models/sd1.5.gguf")
    options.NThreads = 8

    ctx, err := stablediffusion.NewContext(options)
    if err != nil {
        log.Fatalf("Failed to create context: %v", err)
    }
    defer ctx.Close() // Close 之后不得再生成，否则返回 stablediffusion.ErrClosed

    // 图像生成
    cfg := stablediffusion.GenerationConfig{
        Prompt:     "A cyberpunk city, 8k wallpaper",
        Width:      512,
        Height:     512,
        BatchCount: 1,
        Sampler: stablediffusion.SamplerConfig{
            Steps:     20,
            TxtCfg:    7.0,
            Method:    bindings.EULER_A_SAMPLE_METHOD,
            Scheduler: bindings.KARRAS_SCHEDULER,
        },
    }

    images, err := ctx.GenerateImage(cfg)
    if err != nil {
        log.Fatalf("Failed: %v", err)
    }
    log.Printf("Generated %d images", len(images))
}
```

### 内存与生命周期约定（C 指针 / Go 内存交互）

绑定底层是 C 动态库，所有跨边界资源遵循以下规则：

**1. 传给底层的字符串、图像缓冲、结构体数组**

- Go 字符串会转成带 NUL 的临时缓冲（Go 堆内存），**只在同步 C 调用期间**
  借给底层；高层 API 内部用 `runtime.KeepAlive` 保证调用期间不被 GC。
  调用返回后底层不再持有这些指针，由 Go GC 回收，**切勿**对它们调用
  `FreeCPtr`。
- 输入图像（`InitImage`/`MaskImage`/`ControlImage`/`RefImages`/控制帧等）
  的 `Data` 在生成调用返回前必须有效，且**不能从其他 goroutine 并发改写**；
  `PreprocessCanny` 会**原地写回**缓冲。空缓冲/尺寸不匹配会在进入 C 之前报错，
  不会对空切片取指针。
- LoRA、参考帧等 Go 结构体切片同样在调用期间整体保活。

**2. 底层返回的资源由谁释放**

| C 资源 | Go 侧释放方式 |
|--------|--------------|
| `generate_image` 返回的数组 + 每帧 `data` | `GenerateImage` 内部拷贝后自动释放 |
| `generate_video` 返回的数组 + 每帧 `data` | `GenerateVideo` 内部拷贝后自动释放 |
| `upscale` 返回的 `data` | `Upscale` 内部拷贝后自动释放 |
| `*_params_to_str` 返回的字符串 | `SdCtxParamsToStr` 等转成 Go 字符串后自动释放 |
| `new_sd_ctx` / `new_upscaler_ctx` | `Context.Close()` / `Upscaler.Close()` |

即使**生成失败**（返回 nil、帧数为 0 或部分帧为空），已从底层拿到的资源
也会被释放，不会泄漏。底层 `bindings` 层额外导出 `FreeCPtr` /
`FreeImageData` / `FreeImageArray`，仅用于直接使用底层 API 的场景——
它们只能接收 **C malloc 的内存**，不能传 Go 指针。

**3. Close 之后禁止继续生成**

`Close()` 会释放底层上下文（C 侧为 `free`，之后是悬空指针）。Close 后再调用
`GenerateImage`/`GenerateVideo`/`Upscale` 等会返回
`stablediffusion.ErrClosed`，而不会进入 C 层造成 use-after-free。同一
Context/Upscaler 的生成调用由内部互斥锁串行化；`Close` 可重复调用。
旧的 `Free()` 仍保留为 `Close()` 的弃用别名。

**4. 回调及上下文（data）生命周期**

回调是**进程级全局**的（C 侧静态变量，与具体 Context 无关）：

- 推荐使用 `stablediffusion.OnLog` / `OnProgress` / `OnPreview`
  或 `bindings.RegisterLogCallback` 等，它们用 `runtime/cgo.Handle`
  管理任意 Go 上下文并保活，返回的 `CallbackRegistration` 调用 `Release()`
  即注销并释放上下文；
- 预览回调里的 `frames` 只在**本次回调执行期间**有效，需保留请用
  `CopyPreviewFrame(s)` 深拷贝，不能自行 free；
- 直接使用 `SetLogCallback(cb, data unsafe.Pointer)` 时，调用方必须自行
  保证 `data` 指向的对象在注销前存活，且不要在热路径反复注册
  （回调蹦床数量有上限且不释放）。注销用 `UnsetLogCallback` 等。

### 完整 API 清单

对比 `stable-diffusion.h` 全部 38 个导出函数，以下是绑定覆盖状态：

| C 函数 | Go 绑定 (bindings) | 高层 API (stablediffusion) |
|--------|-------|-----------|
| `sd_set_log_callback` | `SetLogCallback` / `RegisterLogCallback` / `UnsetLogCallback` | `SetLogCallback`(弃用) / `OnLog` |
| `sd_set_progress_callback` | `SetProgressCallback` / `RegisterProgressCallback` / `UnsetProgressCallback` | `SetProgressCallback`(弃用) / `OnProgress` |
| `sd_set_preview_callback` | `SetPreviewCallback` / `RegisterPreviewCallback` / `UnsetPreviewCallback` | `SetPreviewCallback`(弃用) / `OnPreview` |
| `sd_get_num_physical_cores` | `GetNumPhysicalCores` | `GetNumPhysicalCores` |
| `sd_get_system_info` | `GetSystemInfo` | `GetSystemInfo` |
| `sd_type_name` | `GetTypeName` | `GetTypeName` |
| `str_to_sd_type` | `StrToSdType` | `StrToSdType` |
| `sd_rng_type_name` | `GetRngTypeName` | `GetRngTypeName` |
| `str_to_rng_type` | `StrToRngType` | `StrToRngType` |
| `sd_sample_method_name` | `GetSampleMethodName` | `GetSampleMethodName` |
| `str_to_sample_method` | `StrToSampleMethod` | `StrToSampleMethod` |
| `sd_scheduler_name` | `GetSchedulerName` | `GetSchedulerName` |
| `str_to_scheduler` | `StrToScheduler` | `StrToScheduler` |
| `sd_prediction_name` | `GetPredictionName` | `GetPredictionName` |
| `str_to_prediction` | `StrToPrediction` | `StrToPrediction` |
| `sd_preview_name` | `GetPreviewName` | `GetPreviewName` |
| `str_to_preview` | `StrToPreview` | `StrToPreview` |
| `sd_lora_apply_mode_name` | `GetLoraApplyModeName` | `GetLoraApplyModeName` |
| `str_to_lora_apply_mode` | `StrToLoraApplyMode` | `StrToLoraApplyMode` |
| `sd_cache_params_init` | `SdCacheParamsInit` | ✅ 内部使用 |
| `sd_ctx_params_init` | `SdCtxParamsInit` | ✅ 内部使用 |
| `sd_ctx_params_to_str` | `SdCtxParamsToStr` | ✅ 调试用 |
| `new_sd_ctx` | `CreateSdCtx` | `NewContext` |
| `free_sd_ctx` | `FreeSdCtx` | `Context.Close`（`Free` 为弃用别名） |
| `sd_sample_params_init` | `SdSampleParamsInit` | ✅ 内部使用 |
| `sd_sample_params_to_str` | `SdSampleParamsToStr` | ✅ 调试用 |
| `sd_get_default_sample_method` | `GetDefaultSampleMethod` | `Context.GetDefaultSampleMethod` |
| `sd_get_default_scheduler` | `GetDefaultScheduler` | `Context.GetDefaultScheduler` |
| `sd_img_gen_params_init` | `SdImgGenParamsInit` | ✅ 内部使用 |
| `sd_img_gen_params_to_str` | `SdImgGenParamsToStr` | ✅ 调试用 |
| `generate_image` | `GenerateImage` | `Context.GenerateImage` |
| `sd_vid_gen_params_init` | `SdVidGenParamsInit` | ✅ 内部使用 |
| `generate_video` | `GenerateVideo` | `Context.GenerateVideo` |
| `new_upscaler_ctx` | `CreateUpscalerCtx` | `NewUpscaler` |
| `free_upscaler_ctx` | `FreeUpscalerCtx` | `Upscaler.Close`（`Free` 为弃用别名） |
| `upscale` | `Upscale` | `Upscaler.Upscale` |
| `get_upscale_factor` | `GetUpscaleFactor` | `Upscaler.GetUpscaleFactor` |
| `convert` | `Convert` | `ConvertModel` |
| `preprocess_canny` | `PreprocessCanny` | `PreprocessCanny` |
| `sd_commit` | `GetCommit` | `GetCommit` |
| `sd_version` | `GetVersion` | `GetVersion` |

---

## 🛠 目录结构

```
├── bindings/                     # 底层 purego 绑定（1:1 映射 C API）
│   ├── stablediffusion.go        # 绑定实现 + Mock
│   └── stablediffusion_test.go   # 字符串/回调/Mock 测试
├── stablediffusion.go            # 高层 Go API 封装
├── test/
│   └── stablediffusion_test.go   # 高层 API 集成测试
├── examples/
│   └── server/server.go          # 生产级 HTTP 服务示例
├── stable-diffusion.cpp/         # 核心 C++ 库（子模块）
├── Dockerfile                    # 多阶段 Linux Docker 构建
└── docker-compose.yml            # 一键部署配置
```

---

## 🧪 测试用例详情

### 底层绑定层 (`bindings/`)

| # | 测试名称 | 验证内容 | 预期结果 |
|---|----------|----------|----------|
| 1.1 | `TestCStringGoString` | Go↔C 字符串转换含 NULL 终止符 | 字符串相等，NULL 存在 |
| 1.2 | `TestMockImplementation` | 无动态库时自动降级到 Mock | `GetSystemInfo()`/`GetVersion()` 返回非空 |
| 1.3 | `TestCallbackWrapper` | `purego.NewCallback` 包装不被 GC 回收 | `currentLogCallback != 0` |
| 1.4 | `TestBytePtr` / `TestFreeInMockModeIsNoop` | 空切片返回 nil；mock 下释放 C 资源为空操作 | 不 panic、不崩溃 |
| 1.5 | `TestParamsToStrFreesCBuffer` | `*_ToStr` 返回 Go 字符串且内部释放 C 缓冲 | 内容正确 |
| 1.6 | `TestRegisterCallbackWithContext` / `TestUnsetCallbacks` | `cgo.Handle` 上下文保活、`Release` 幂等、可注销 | 不 panic |

### 高层 API 层 (`test/` 及包内生命周期测试)

| # | 测试名称 | 验证内容 | 预期结果 |
|---|----------|----------|----------|
| 2.1 | `TestSystemInfo` | `stablediffusion.GetSystemInfo()` | 返回非空字符串 |
| 2.2 | `TestVersionInfo` | `GetVersion()` + `GetCommit()` | 均返回非空 |
| 2.3 | `TestCreateContext` | `NewContext` 无模型文件时 | 返回 `error` |
| 2.4 | `TestCreateUpscaler` | `NewUpscaler` 无模型文件时 | 返回 `error` |
| 2.5 | `TestConvertModel` | `ConvertModel` 无文件时 | 返回 `error` |
| 2.6 | `TestImageGenerationConfig` | `GenerationConfig` 结构体初始化 | `Width=512`, `Height=512` |
| 2.7 | `TestClosedContextRejectsGeneration` | Close/零值上下文再生成 | 返回 `ErrClosed`，不进 C 层 |
| 2.8 | `TestClosedUpscalerRejectsUpscale` | Close/零值 Upscaler 再超分 | 返回 `ErrClosed` |
| 2.9 | `TestImageValidation` / `TestCannyValidation` | nil/尺寸与缓冲不符的输入 | 进入 C 之前报错 |

### 运行所有测试

```bash
go test -v ./...
```

预期输出：
```
=== RUN   TestCStringGoString
--- PASS: TestCStringGoString
=== RUN   TestMockImplementation
--- PASS: TestMockImplementation
=== RUN   TestCallbackWrapper
--- PASS: TestCallbackWrapper
PASS
ok  github.com/example/stablediffusion/bindings

=== RUN   TestSystemInfo
--- PASS: TestSystemInfo
=== RUN   TestVersionInfo
--- PASS: TestVersionInfo
=== RUN   TestCreateContext
--- PASS: TestCreateContext
=== RUN   TestCreateUpscaler
--- PASS: TestCreateUpscaler
=== RUN   TestConvertModel
--- PASS: TestConvertModel
=== RUN   TestImageGenerationConfig
--- PASS: TestImageGenerationConfig
PASS
ok  github.com/example/stablediffusion/test
```

---

## 🔧 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `SD_LIB_PATH` | 动态库完整路径 | 自动搜索平台默认名 |
| `MODEL_PATH` | 模型文件路径 | `models/model.gguf` |
| `PORT` | HTTP 服务端口 | `8080` |
| `HOST` | HTTP 监听地址 | `0.0.0.0` |
| `GENERATE_TIMEOUT` | 生成超时时间 | `5m` |
