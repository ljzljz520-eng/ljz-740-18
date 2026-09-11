package bindings

import (
	"fmt"
	"os"
	"runtime"
	"runtime/cgo"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// 平台相关的库名
var libNames = map[string]string{
	"darwin":  "libstable-diffusion.dylib",
	"linux":   "libstable-diffusion.so",
	"windows": "stable-diffusion.dll", // 与 CMakeLists.txt 一致
}

// 全局变量，存储加载的库
var (
	lib uintptr
	// cFree 指向动态库解析出来的 C 标准库 free。
	// stable-diffusion.cpp 对所有返回给调用方的资源（生成的图像缓冲、
	// 结果数组、*_to_str 字符串等）统一使用 malloc/calloc 分配，
	// 必须由同一 libc 的 free 释放，因此不能用 Go 的内存管理回收。
	// 库加载失败（mock 模式）时为 nil，此时不存在真正的 C 分配，释放操作为空操作。
	cFree uintptr
)

// 保持回调函数的引用，防止被GC
var (
	currentLogCallback      uintptr
	currentProgressCallback uintptr
	currentPreviewCallback  uintptr

	// C 侧只保存 data 的裸指针，不参与 Go GC。这里保活最近一次注册时
	// 传入的上下文，保证回调被 C 侧触发期间其指向的 Go 对象不会被回收。
	// 重新注册会替换保活对象，注销回调时清空。
	callbackMu      sync.Mutex
	currentLogData  unsafe.Pointer
	currentProgData unsafe.Pointer
	currentPrevData unsafe.Pointer
)

// 初始化函数，加载对应平台的库
func init() {
	// 优先检查环境变量
	libName := os.Getenv("SD_LIB_PATH")
	if libName == "" {
		name, ok := libNames[runtime.GOOS]
		if ok {
			// 尝试在几个可能的路径搜索
			searchPaths := []string{
				name,
				"./" + name,
				"./stable-diffusion.cpp/build/bin/" + name,
				"./stable-diffusion.cpp/build/bin/Release/" + name,
				"/usr/local/lib/" + name,
				"/usr/lib/" + name,
			}
			for _, p := range searchPaths {
				if _, err := os.Stat(p); err == nil {
					libName = p
					break
				}
			}
			if libName == "" {
				libName = name // 回退到默认库名，让 Dlopen 尝试系统路径
			}
		} else {
			fmt.Printf("Warning: Unsupported platform: %s, using mock implementation\n", runtime.GOOS)
			InitFuncs()
			return
		}
	}

	// 尝试加载库
	var err error
	lib, err = purego.Dlopen(libName, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		fmt.Printf("Warning: Failed to load library: %v, using mock implementation\n", err)
	}

	// 初始化函数指针
	InitFuncs()
}

// 枚举类型定义

type RngType int
type SampleMethod int
type Scheduler int
type Prediction int
type SdType int
type SdLogLevel int
type Preview int
type LoraApplyMode int
type SdCacheMode int

// 常量定义
const (
	// RngType
	STD_DEFAULT_RNG RngType = iota
	CUDA_RNG
	CPU_RNG
	RNG_TYPE_COUNT

	// SampleMethod
	EULER_SAMPLE_METHOD SampleMethod = iota
	EULER_A_SAMPLE_METHOD
	HEUN_SAMPLE_METHOD
	DPM2_SAMPLE_METHOD
	DPMPP2S_A_SAMPLE_METHOD
	DPMPP2M_SAMPLE_METHOD
	DPMPP2Mv2_SAMPLE_METHOD
	IPNDM_SAMPLE_METHOD
	IPNDM_V_SAMPLE_METHOD
	LCM_SAMPLE_METHOD
	DDIM_TRAILING_SAMPLE_METHOD
	TCD_SAMPLE_METHOD
	RES_MULTISTEP_SAMPLE_METHOD
	RES_2S_SAMPLE_METHOD
	SAMPLE_METHOD       = iota // Added to match header logic if needed, but SAMPLE_METHOD_COUNT is usually last
	SAMPLE_METHOD_COUNT = RES_2S_SAMPLE_METHOD + 1

	// Scheduler
	DISCRETE_SCHEDULER Scheduler = iota
	KARRAS_SCHEDULER
	EXPONENTIAL_SCHEDULER
	AYS_SCHEDULER
	GITS_SCHEDULER
	SGM_UNIFORM_SCHEDULER
	SIMPLE_SCHEDULER
	SMOOTHSTEP_SCHEDULER
	KL_OPTIMAL_SCHEDULER
	LCM_SCHEDULER
	BONG_TANGENT_SCHEDULER
	SCHEDULER_COUNT

	// Prediction
	EPS_PRED Prediction = iota
	V_PRED
	EDM_V_PRED
	FLOW_PRED
	FLUX_FLOW_PRED
	FLUX2_FLOW_PRED
	PREDICTION_COUNT

	// SdType
	SD_TYPE_F32     SdType = 0
	SD_TYPE_F16     SdType = 1
	SD_TYPE_Q4_0    SdType = 2
	SD_TYPE_Q4_1    SdType = 3
	SD_TYPE_Q5_0    SdType = 6
	SD_TYPE_Q5_1    SdType = 7
	SD_TYPE_Q8_0    SdType = 8
	SD_TYPE_Q8_1    SdType = 9
	SD_TYPE_Q2_K    SdType = 10
	SD_TYPE_Q3_K    SdType = 11
	SD_TYPE_Q4_K    SdType = 12
	SD_TYPE_Q5_K    SdType = 13
	SD_TYPE_Q6_K    SdType = 14
	SD_TYPE_Q8_K    SdType = 15
	SD_TYPE_IQ2_XXS SdType = 16
	SD_TYPE_IQ2_XS  SdType = 17
	SD_TYPE_IQ3_XXS SdType = 18
	SD_TYPE_IQ1_S   SdType = 19
	SD_TYPE_IQ4_NL  SdType = 20
	SD_TYPE_IQ3_S   SdType = 21
	SD_TYPE_IQ2_S   SdType = 22
	SD_TYPE_IQ4_XS  SdType = 23
	SD_TYPE_I8      SdType = 24
	SD_TYPE_I16     SdType = 25
	SD_TYPE_I32     SdType = 26
	SD_TYPE_I64     SdType = 27
	SD_TYPE_F64     SdType = 28
	SD_TYPE_IQ1_M   SdType = 29
	SD_TYPE_BF16    SdType = 30
	SD_TYPE_TQ1_0   SdType = 34
	SD_TYPE_TQ2_0   SdType = 35
	SD_TYPE_MXFP4   SdType = 39
	SD_TYPE_COUNT   SdType = 40

	// SdLogLevel
	SD_LOG_DEBUG SdLogLevel = iota
	SD_LOG_INFO
	SD_LOG_WARN
	SD_LOG_ERROR

	// Preview
	PREVIEW_NONE Preview = iota
	PREVIEW_PROJ
	PREVIEW_TAE
	PREVIEW_VAE
	PREVIEW_COUNT

	// LoraApplyMode
	LORA_APPLY_AUTO LoraApplyMode = iota
	LORA_APPLY_IMMEDIATELY
	LORA_APPLY_AT_RUNTIME
	LORA_APPLY_MODE_COUNT

	// SdCacheMode
	SD_CACHE_DISABLED SdCacheMode = iota
	SD_CACHE_EASYCACHE
	SD_CACHE_UCACHE
	SD_CACHE_DBCACHE
	SD_CACHE_TAYLORSEER
	SD_CACHE_CACHE_DIT
)

// 结构体定义
type SdTilingParams struct {
	Enabled       bool
	TileSizeX     int
	TileSizeY     int
	TargetOverlap float32
	RelSizeX      float32
	RelSizeY      float32
}

type SdEmbedding struct {
	Name *byte
	Path *byte
}

type SdCtxParams struct {
	ModelPath                   *byte
	ClipLPath                   *byte
	ClipGPath                   *byte
	ClipVisionPath              *byte
	T5xxlPath                   *byte
	LlmPath                     *byte
	LlmVisionPath               *byte
	DiffusionModelPath          *byte
	HighNoiseDiffusionModelPath *byte
	VaePath                     *byte
	TaesdPath                   *byte
	ControlNetPath              *byte
	Embeddings                  *SdEmbedding
	EmbeddingCount              uint32
	PhotoMakerPath              *byte
	TensorTypeRules             *byte
	VaeDecodeOnly               bool
	FreeParamsImmediately       bool
	NThreads                    int
	Wtype                       SdType
	RngType                     RngType
	SamplerRngType              RngType
	Prediction                  Prediction
	LoraApplyMode               LoraApplyMode
	OffloadParamsToCpu          bool
	EnableMmap                  bool
	KeepClipOnCpu               bool
	KeepControlNetOnCpu         bool
	KeepVaeOnCpu                bool
	FlashAttn                   bool
	DiffusionFlashAttn          bool
	TaePreviewOnly              bool
	DiffusionConvDirect         bool
	VaeConvDirect               bool
	CircularX                   bool
	CircularY                   bool
	ForceSdxlVaeConvScale       bool
	ChromaUseDitMask            bool
	ChromaUseT5Mask             bool
	ChromaT5MaskPad             int
	QwenImageZeroCondT          bool
	FlowShift                   float32
}

type SdImage struct {
	Width   uint32
	Height  uint32
	Channel uint32
	Data    *uint8
}

type SdSlgParams struct {
	Layers     *int
	LayerCount uintptr
	LayerStart float32
	LayerEnd   float32
	Scale      float32
}

type SdGuidanceParams struct {
	TxtCfg            float32
	ImgCfg            float32
	DistilledGuidance float32
	Slg               SdSlgParams
}

type SdSampleParams struct {
	Guidance          SdGuidanceParams
	Scheduler         Scheduler
	SampleMethod      SampleMethod
	SampleSteps       int
	Eta               float32
	ShiftedTimestep   int
	CustomSigmas      *float32
	CustomSigmasCount int
}

type SdPmParams struct {
	IdImages      *SdImage
	IdImagesCount int
	IdEmbedPath   *byte
	StyleStrength float32
}

type SdCacheParams struct {
	Mode                     SdCacheMode
	ReuseThreshold           float32
	StartPercent             float32
	EndPercent               float32
	ErrorDecayRate           float32
	UseRelativeThreshold     bool
	ResetErrorOnCompute      bool
	FnComputeBlocks          int
	BnComputeBlocks          int
	ResidualDiffThreshold    float32
	MaxWarmupSteps           int
	MaxCachedSteps           int
	MaxContinuousCachedSteps int
	TaylorseerNDerivatives   int
	TaylorseerSkipInterval   int
	ScmMask                  *byte
	ScmPolicyDynamic         bool
}

type SdLora struct {
	IsHighNoise bool
	Multiplier  float32
	Path        *byte
}

type SdImgGenParams struct {
	Loras              *SdLora
	LoraCount          uint32
	Prompt             *byte
	NegativePrompt     *byte
	ClipSkip           int
	InitImage          SdImage
	RefImages          *SdImage
	RefImagesCount     int
	AutoResizeRefImage bool
	IncreaseRefIndex   bool
	MaskImage          SdImage
	Width              int
	Height             int
	SampleParams       SdSampleParams
	Strength           float32
	Seed               int64
	BatchCount         int
	ControlImage       SdImage
	ControlStrength    float32
	PmParams           SdPmParams
	VaeTilingParams    SdTilingParams
	Cache              SdCacheParams
}

type SdVidGenParams struct {
	Loras                 *SdLora
	LoraCount             uint32
	Prompt                *byte
	NegativePrompt        *byte
	ClipSkip              int
	InitImage             SdImage
	EndImage              SdImage
	ControlFrames         *SdImage
	ControlFramesSize     int
	Width                 int
	Height                int
	SampleParams          SdSampleParams
	HighNoiseSampleParams SdSampleParams
	MoeBoundary           float32
	Strength              float32
	Seed                  int64
	VideoFrames           int
	VaceStrength          float32
	VaeTilingParams       SdTilingParams
	Cache                 SdCacheParams
}

type SdCtx struct{}
type UpscalerCtx struct{}

// 回调函数类型
type SdLogCb func(level SdLogLevel, text *byte, data unsafe.Pointer)
type SdProgressCb func(step, steps int, time float32, data unsafe.Pointer)
type SdPreviewCb func(step, frameCount int, frames *SdImage, isNoisy bool, data unsafe.Pointer)

// 函数原型定义
var (
	// 设置日志回调
	sdSetLogCallback func(cb uintptr, data unsafe.Pointer)

	// 设置进度回调
	sdSetProgressCallback func(cb uintptr, data unsafe.Pointer)

	// 设置预览回调
	sdSetPreviewCallback func(cb uintptr, mode Preview, interval int, denoised, noisy bool, data unsafe.Pointer)

	// 获取物理核心数
	sdGetNumPhysicalCores func() int32

	// 获取系统信息
	sdGetSystemInfo func() *byte

	// 获取类型名称
	sdTypeName          func(t SdType) *byte
	strToSdType         func(s *byte) SdType
	sdRngTypeName       func(t RngType) *byte
	strToRngType        func(s *byte) RngType
	sdSampleMethodName  func(t SampleMethod) *byte
	strToSampleMethod   func(s *byte) SampleMethod
	sdSchedulerName     func(t Scheduler) *byte
	strToScheduler      func(s *byte) Scheduler
	sdPredictionName    func(t Prediction) *byte
	strToPrediction     func(s *byte) Prediction
	sdPreviewName       func(t Preview) *byte
	strToPreview        func(s *byte) Preview
	sdLoraApplyModeName func(t LoraApplyMode) *byte
	strToLoraApplyMode  func(s *byte) LoraApplyMode

	// 缓存参数初始化
	sdCacheParamsInit func(params *SdCacheParams)

	// 上下文参数初始化
	sdCtxParamsInit  func(params *SdCtxParams)
	sdCtxParamsToStr func(params *SdCtxParams) *byte

	// 创建和释放上下文
	newSdCtx  func(params *SdCtxParams) *SdCtx
	freeSdCtx func(ctx *SdCtx)

	// 采样参数初始化
	sdSampleParamsInit  func(params *SdSampleParams)
	sdSampleParamsToStr func(params *SdSampleParams) *byte

	// 获取默认采样方法和调度器
	sdGetDefaultSampleMethod func(ctx *SdCtx) SampleMethod
	sdGetDefaultScheduler    func(ctx *SdCtx, method SampleMethod) Scheduler

	// 图像生成参数初始化
	sdImgGenParamsInit  func(params *SdImgGenParams)
	sdImgGenParamsToStr func(params *SdImgGenParams) *byte
	generateImage       func(ctx *SdCtx, params *SdImgGenParams) *SdImage

	// 视频生成参数初始化
	sdVidGenParamsInit func(params *SdVidGenParams)
	generateVideo      func(ctx *SdCtx, params *SdVidGenParams, numFramesOut *int) *SdImage

	// 超分辨率
	newUpscalerCtx   func(esrganPath *byte, offloadParamsToCpu, direct bool, nThreads, tileSize int) *UpscalerCtx
	freeUpscalerCtx  func(ctx *UpscalerCtx)
	upscale          func(ctx *UpscalerCtx, input SdImage, factor uint32) SdImage
	getUpscaleFactor func(ctx *UpscalerCtx) int

	// 转换
	convert func(inputPath, vaePath, outputPath *byte, outputType SdType, tensorTypeRules *byte, convertName bool) bool

	// 预处理
	preprocessCanny func(image SdImage, highThreshold, lowThreshold, weak, strong float32, inverse bool) bool

	// 获取版本信息
	sdCommit  func() *byte
	sdVersion func() *byte
)

// 初始化所有函数指针
func InitFuncs() {
	if lib == 0 {
		// 设置模拟实现
		setMockImplementations()
		return
	}

	// C 侧返回的所有内存都由 libc 分配，解析同一个 libc 的 free 用于释放。
	// stable-diffusion 动态库必然链接 libc/libSystem，因此可以从其句柄解析到 free。
	if p, err := purego.Dlsym(lib, "free"); err == nil {
		cFree = p
	}

	purego.RegisterLibFunc(&sdSetLogCallback, lib, "sd_set_log_callback")
	purego.RegisterLibFunc(&sdSetProgressCallback, lib, "sd_set_progress_callback")
	purego.RegisterLibFunc(&sdSetPreviewCallback, lib, "sd_set_preview_callback")
	purego.RegisterLibFunc(&sdGetNumPhysicalCores, lib, "sd_get_num_physical_cores")
	purego.RegisterLibFunc(&sdGetSystemInfo, lib, "sd_get_system_info")
	purego.RegisterLibFunc(&sdTypeName, lib, "sd_type_name")
	purego.RegisterLibFunc(&strToSdType, lib, "str_to_sd_type")
	purego.RegisterLibFunc(&sdRngTypeName, lib, "sd_rng_type_name")
	purego.RegisterLibFunc(&strToRngType, lib, "str_to_rng_type")
	purego.RegisterLibFunc(&sdSampleMethodName, lib, "sd_sample_method_name")
	purego.RegisterLibFunc(&strToSampleMethod, lib, "str_to_sample_method")
	purego.RegisterLibFunc(&sdSchedulerName, lib, "sd_scheduler_name")
	purego.RegisterLibFunc(&strToScheduler, lib, "str_to_scheduler")
	purego.RegisterLibFunc(&sdPredictionName, lib, "sd_prediction_name")
	purego.RegisterLibFunc(&strToPrediction, lib, "str_to_prediction")
	purego.RegisterLibFunc(&sdPreviewName, lib, "sd_preview_name")
	purego.RegisterLibFunc(&strToPreview, lib, "str_to_preview")
	purego.RegisterLibFunc(&sdLoraApplyModeName, lib, "sd_lora_apply_mode_name")
	purego.RegisterLibFunc(&strToLoraApplyMode, lib, "str_to_lora_apply_mode")
	purego.RegisterLibFunc(&sdCacheParamsInit, lib, "sd_cache_params_init")
	purego.RegisterLibFunc(&sdCtxParamsInit, lib, "sd_ctx_params_init")
	purego.RegisterLibFunc(&sdCtxParamsToStr, lib, "sd_ctx_params_to_str")
	purego.RegisterLibFunc(&newSdCtx, lib, "new_sd_ctx")
	purego.RegisterLibFunc(&freeSdCtx, lib, "free_sd_ctx")
	purego.RegisterLibFunc(&sdSampleParamsInit, lib, "sd_sample_params_init")
	purego.RegisterLibFunc(&sdSampleParamsToStr, lib, "sd_sample_params_to_str")
	purego.RegisterLibFunc(&sdGetDefaultSampleMethod, lib, "sd_get_default_sample_method")
	purego.RegisterLibFunc(&sdGetDefaultScheduler, lib, "sd_get_default_scheduler")
	purego.RegisterLibFunc(&sdImgGenParamsInit, lib, "sd_img_gen_params_init")
	purego.RegisterLibFunc(&sdImgGenParamsToStr, lib, "sd_img_gen_params_to_str")
	purego.RegisterLibFunc(&generateImage, lib, "generate_image")
	purego.RegisterLibFunc(&sdVidGenParamsInit, lib, "sd_vid_gen_params_init")
	purego.RegisterLibFunc(&generateVideo, lib, "generate_video")
	purego.RegisterLibFunc(&newUpscalerCtx, lib, "new_upscaler_ctx")
	purego.RegisterLibFunc(&freeUpscalerCtx, lib, "free_upscaler_ctx")
	purego.RegisterLibFunc(&upscale, lib, "upscale")
	purego.RegisterLibFunc(&getUpscaleFactor, lib, "get_upscale_factor")
	purego.RegisterLibFunc(&convert, lib, "convert")
	purego.RegisterLibFunc(&preprocessCanny, lib, "preprocess_canny")
	purego.RegisterLibFunc(&sdCommit, lib, "sd_commit")
	purego.RegisterLibFunc(&sdVersion, lib, "sd_version")
}

// 设置模拟实现
func setMockImplementations() {
	// 模拟系统信息函数
	sdGetSystemInfo = func() *byte {
		return CString("CPU: Mock CPU, Cores: 10, Version: 1.0.0")
	}

	// 模拟版本信息函数
	sdVersion = func() *byte {
		return CString("1.0.0")
	}

	sdCommit = func() *byte {
		return CString("mock-commit-hash")
	}

	// 模拟核心数函数
	sdGetNumPhysicalCores = func() int32 {
		return 10
	}

	// 模拟上下文创建函数
	newSdCtx = func(params *SdCtxParams) *SdCtx {
		return nil
	}

	// 模拟其他函数
	freeSdCtx = func(ctx *SdCtx) {}
	sdGetDefaultSampleMethod = func(ctx *SdCtx) SampleMethod {
		return EULER_A_SAMPLE_METHOD
	}
	sdGetDefaultScheduler = func(ctx *SdCtx, method SampleMethod) Scheduler {
		return KARRAS_SCHEDULER
	}
	generateImage = func(ctx *SdCtx, params *SdImgGenParams) *SdImage {
		return nil
	}
	generateVideo = func(ctx *SdCtx, params *SdVidGenParams, numFramesOut *int) *SdImage {
		return nil
	}
	newUpscalerCtx = func(esrganPath *byte, offloadParamsToCpu, direct bool, nThreads, tileSize int) *UpscalerCtx {
		return nil
	}
	freeUpscalerCtx = func(ctx *UpscalerCtx) {}
	upscale = func(ctx *UpscalerCtx, input SdImage, factor uint32) SdImage {
		return SdImage{}
	}
	getUpscaleFactor = func(ctx *UpscalerCtx) int {
		return 2
	}
	convert = func(inputPath, vaePath, outputPath *byte, outputType SdType, tensorTypeRules *byte, convertName bool) bool {
		return false
	}
	preprocessCanny = func(image SdImage, highThreshold, lowThreshold, weak, strong float32, inverse bool) bool {
		return false
	}

	// 初始化参数函数
	sdCacheParamsInit = func(params *SdCacheParams) {}
	sdCtxParamsInit = func(params *SdCtxParams) {}
	sdCtxParamsToStr = func(params *SdCtxParams) *byte {
		return CString("mock ctx params")
	}
	sdSampleParamsInit = func(params *SdSampleParams) {}
	sdSampleParamsToStr = func(params *SdSampleParams) *byte {
		return CString("mock sample params")
	}
	sdImgGenParamsInit = func(params *SdImgGenParams) {}
	sdImgGenParamsToStr = func(params *SdImgGenParams) *byte {
		return CString("mock img gen params")
	}
	sdVidGenParamsInit = func(params *SdVidGenParams) {}

	// 模拟类型转换函数
	sdTypeName = func(t SdType) *byte {
		return CString("mock type")
	}
	strToSdType = func(s *byte) SdType {
		return SD_TYPE_F16
	}
	sdRngTypeName = func(t RngType) *byte {
		return CString("mock rng type")
	}
	strToRngType = func(s *byte) RngType {
		return CPU_RNG
	}
	sdSampleMethodName = func(t SampleMethod) *byte {
		return CString("mock sample method")
	}
	strToSampleMethod = func(s *byte) SampleMethod {
		return EULER_A_SAMPLE_METHOD
	}
	sdSchedulerName = func(t Scheduler) *byte {
		return CString("mock scheduler")
	}
	strToScheduler = func(s *byte) Scheduler {
		return KARRAS_SCHEDULER
	}
	sdPredictionName = func(t Prediction) *byte {
		return CString("mock prediction")
	}
	strToPrediction = func(s *byte) Prediction {
		return EPS_PRED
	}
	sdPreviewName = func(t Preview) *byte {
		return CString("mock preview")
	}
	strToPreview = func(s *byte) Preview {
		return PREVIEW_NONE
	}
	sdLoraApplyModeName = func(t LoraApplyMode) *byte {
		return CString("mock lora apply mode")
	}
	strToLoraApplyMode = func(s *byte) LoraApplyMode {
		return LORA_APPLY_AUTO
	}

	// 模拟回调设置函数
	sdSetLogCallback = func(cb uintptr, data unsafe.Pointer) {}
	sdSetProgressCallback = func(cb uintptr, data unsafe.Pointer) {}
	sdSetPreviewCallback = func(cb uintptr, mode Preview, interval int, denoised, noisy bool, data unsafe.Pointer) {}
}

// CString 将 Go 字符串转换为以 NUL 结尾的 C 字符串。
//
// 生命周期说明：返回的 *byte 指向 Go 堆内存（非 C 分配），只能作为
// “同步调用”参数传给底层函数——即该 C 调用返回前指针必须一直有效，
// 且 C 侧不会在调用返回后继续持有它。stable-diffusion.cpp 的所有
// 字符串参数（路径、prompt、tensor_type_rules 等）都在调用内部拷贝，
// 因此满足该条件。调用方必须在 C 调用返回后用 runtime.KeepAlive
// 保活其所属分配，禁止把该指针保存到长期存活的结构中留给后续调用。
func CString(s string) *byte {
	// 关键修复：添加 NULL 结尾
	b := append([]byte(s), 0)
	return &b[0]
}

// BytePtr 返回字节切片底层数组的指针；空切片返回 nil。
//
// 与 CString 相同，返回的指针只在当前 Go 切片存活期间有效，
// 仅可用于同步 C 调用，并在调用后对切片执行 runtime.KeepAlive。
// 调用方还必须保证在 C 调用返回前不从 Go 侧并发改写该切片
// （preprocess_canny 会原地写回缓冲）。
func BytePtr(b []byte) *byte {
	if len(b) == 0 {
		return nil
	}
	return &b[0]
}

// FreeCPtr 释放底层库通过 malloc/calloc 返回的内存。
//
// 适用对象：generate_image / generate_video 返回的图像数组本身及每个
// SdImage.Data、upscale 返回的 SdImage.Data、*_params_to_str 返回的
// 字符串。Go 分配的内存（CString、BytePtr 的结果）绝不能传入本函数。
// p 为 nil 或是 mock 模式（无真实 C 分配）时为空操作。
func FreeCPtr(p unsafe.Pointer) {
	if p == nil || cFree == 0 {
		return
	}
	purego.SyscallN(cFree, uintptr(p))
}

// FreeImageData 释放单张底层返回图像的数据缓冲。
// 不会释放 SdImage 结构体本身（按值嵌入时由数组持有）。
func FreeImageData(img *SdImage) {
	if img == nil || img.Data == nil {
		return
	}
	FreeCPtr(unsafe.Pointer(img.Data))
	img.Data = nil
}

// FreeImageArray 释放底层返回的图像数组：先释放每帧的 data，
// 再释放数组本身，与 C 示例（cli/main.cpp）的释放方式一致。
//
// count 应使用 generate_image 实际的 batch_count 或
// generate_video 返回的 num_frames_out。任何一帧 data 为 nil
// （生成失败的槽位）都会被跳过。
func FreeImageArray(images *SdImage, count int) {
	if images == nil || count <= 0 {
		return
	}
	// C 数组是连续的 SdImage 结构体
	slice := unsafe.Slice(images, count)
	for i := range slice {
		if slice[i].Data != nil {
			FreeCPtr(unsafe.Pointer(slice[i].Data))
			slice[i].Data = nil
		}
	}
	FreeCPtr(unsafe.Pointer(images))
}

// 辅助函数：将C字符串转换为Go字符串
//
// 入参必须指向 C 侧内存（静态字符串或 malloc 缓冲）。
// 本函数只读取并拷贝内容，不释放入参；对于 malloc 返回的
// 字符串（如 *_params_to_str），调用方需随后调用 FreeCPtr。
func GoString(c *byte) string {
	if c == nil {
		return ""
	}
	// 关键修复：正确计算字符串长度直到 NULL。
	// 使用 unsafe.Add 做指针步进，避免 uintptr -> Pointer 的非法中间转换。
	n := 0
	for *(*byte)(unsafe.Add(unsafe.Pointer(c), n)) != 0 {
		n++
	}
	return unsafe.String(c, n)
}

// 公开的API函数

// SetLogCallback 设置日志回调。
//
// 生命周期：
//   - 回调是进程级全局状态（C 侧保存在静态变量中），替换/注销前会被
//     任何 goroutine 上的 C 日志触发，因此 cb 与 data 必须在注销前
//     一直有效。本包会保活最近一次注册的 data，但不会阻止调用方
//     自行让其底层对象失效（例如 data 指向的结构被其他逻辑回收/复用）。
//   - data 为裸指针，调用方负责其指向内存的有效性；推荐改用
//     RegisterLogCallback（以 cgo.Handle 管理任意 Go 上下文）。
//   - purego 回调蹦床在进程内数量有限且不释放，不要在热路径反复注册。
func SetLogCallback(cb SdLogCb, data unsafe.Pointer) {
	// 使用 purego.NewCallback 包装 Go 函数
	currentLogCallback = purego.NewCallback(func(level SdLogLevel, text *byte, d unsafe.Pointer) {
		cb(level, text, d)
	})
	callbackMu.Lock()
	currentLogData = data
	callbackMu.Unlock()
	sdSetLogCallback(currentLogCallback, data)
}

// SetProgressCallback 设置进度回调，生命周期约束同 SetLogCallback。
func SetProgressCallback(cb SdProgressCb, data unsafe.Pointer) {
	currentProgressCallback = purego.NewCallback(func(step, steps int, time float32, d unsafe.Pointer) {
		cb(step, steps, time, d)
	})
	callbackMu.Lock()
	currentProgData = data
	callbackMu.Unlock()
	sdSetProgressCallback(currentProgressCallback, data)
}

// SetPreviewCallback 设置预览回调，生命周期约束同 SetLogCallback。
//
// 额外说明：回调收到的 frames 是 C 侧内存，仅在本次回调执行期间有效，
// 回调返回后 C 可能立即复用/释放；如需保留必须自行拷贝。绝不能对
// frames 或其 Data 调用 FreeCPtr——其所有权属于底层库。
func SetPreviewCallback(cb SdPreviewCb, mode Preview, interval int, denoised, noisy bool, data unsafe.Pointer) {
	currentPreviewCallback = purego.NewCallback(func(step, frameCount int, frames *SdImage, isNoisy bool, d unsafe.Pointer) {
		cb(step, frameCount, frames, isNoisy, d)
	})
	callbackMu.Lock()
	currentPrevData = data
	callbackMu.Unlock()
	sdSetPreviewCallback(currentPreviewCallback, mode, interval, denoised, noisy, data)
}

// UnsetLogCallback 注销日志回调并清空调用方上下文。
// 注意：这只阻止后续 Go 回调被触发；正在另一个 OS 线程执行的回调
// 不会被中断，注销后仍应保证 data 存活到没有并发生成为止。
func UnsetLogCallback() {
	callbackMu.Lock()
	currentLogData = nil
	callbackMu.Unlock()
	sdSetLogCallback(0, nil)
}

// UnsetProgressCallback 注销进度回调，语义同 UnsetLogCallback。
func UnsetProgressCallback() {
	callbackMu.Lock()
	currentProgData = nil
	callbackMu.Unlock()
	sdSetProgressCallback(0, nil)
}

// UnsetPreviewCallback 注销预览回调，语义同 UnsetLogCallback。
func UnsetPreviewCallback() {
	callbackMu.Lock()
	currentPrevData = nil
	callbackMu.Unlock()
	sdSetPreviewCallback(0, PREVIEW_NONE, 0, false, false, nil)
}

// CallbackRegistration 表示一次带 Go 上下文的回调注册。
// Release 前 context 会被 cgo.Handle 保活；Release 注销 C 回调
// 并释放 Handle，此后不得再依赖回调触发。
type CallbackRegistration struct {
	release func()
}

// Release 注销回调并释放上下文句柄。可重复调用。
//
// 注意：C 回调是全局的，Release 仅在当前注册仍是“最后一次注册”时
// 才真正注销底层回调；若之后又注册了别的回调，则只释放本注册的
// 上下文句柄（避免误注销新回调）。
func (r *CallbackRegistration) Release() {
	if r != nil && r.release != nil {
		r.release()
		r.release = nil
	}
}

// RegisterLogCallback 注册携带任意 Go 上下文的日志回调。
// context 可为 nil；非 nil 时其生命周期由返回的 Registration 管理，
// 无需调用方手动保活裸指针。
func RegisterLogCallback(cb SdLogCb, context any) *CallbackRegistration {
	// 官方推荐：把 cgo.Handle 的地址（真实 Go 指针）作为 data，
	// 而不是把 handle 整数强转成指针。h 被返回的闭包保活，
	// Release 前一直有效。
	var h cgo.Handle
	var p unsafe.Pointer
	if context != nil {
		h = cgo.NewHandle(context)
		p = unsafe.Pointer(&h)
	}
	SetLogCallback(cb, p)
	cbPtr := currentLogCallback
	return &CallbackRegistration{release: func() {
		// 只有底层仍指向本次注册的蹦床时才注销，避免误删更新的注册
		if currentLogCallback == cbPtr {
			UnsetLogCallback()
		}
		if h != 0 {
			h.Delete()
		}
	}}
}

// RegisterProgressCallback 注册携带任意 Go 上下文的进度回调。
func RegisterProgressCallback(cb SdProgressCb, context any) *CallbackRegistration {
	var h cgo.Handle
	var p unsafe.Pointer
	if context != nil {
		h = cgo.NewHandle(context)
		p = unsafe.Pointer(&h)
	}
	SetProgressCallback(cb, p)
	cbPtr := currentProgressCallback
	return &CallbackRegistration{release: func() {
		if currentProgressCallback == cbPtr {
			UnsetProgressCallback()
		}
		if h != 0 {
			h.Delete()
		}
	}}
}

// RegisterPreviewCallback 注册携带任意 Go 上下文的预览回调。
// frames 的内存约束见 SetPreviewCallback 文档。
func RegisterPreviewCallback(cb SdPreviewCb, mode Preview, interval int, denoised, noisy bool, context any) *CallbackRegistration {
	var h cgo.Handle
	var p unsafe.Pointer
	if context != nil {
		h = cgo.NewHandle(context)
		p = unsafe.Pointer(&h)
	}
	SetPreviewCallback(cb, mode, interval, denoised, noisy, p)
	cbPtr := currentPreviewCallback
	return &CallbackRegistration{release: func() {
		if currentPreviewCallback == cbPtr {
			UnsetPreviewCallback()
		}
		if h != 0 {
			h.Delete()
		}
	}}
}

// CallbackContext 从回调的 data 参数取回 Register*Callback 注册时
// 传入的 Go 上下文；没有上下文或类型不符时返回 nil。
// CallbackContext 从回调的 data 参数取回 Register*Callback 注册时
// 传入的 Go 上下文；没有上下文或类型不符时返回 nil。
//
// data 指向的是注册时保存的 cgo.Handle（按官方约定传其地址），
// 这里解引用后再取 Value。该值仅在回调执行期间有效。
func CallbackContext(data unsafe.Pointer) any {
	if data == nil {
		return nil
	}
	h := *(*cgo.Handle)(data)
	if h == 0 {
		return nil
	}
	return h.Value()
}

// GetNumPhysicalCores 获取物理核心数
func GetNumPhysicalCores() int32 {
	return sdGetNumPhysicalCores()
}

// GetSystemInfo 获取系统信息
func GetSystemInfo() string {
	return GoString(sdGetSystemInfo())
}

// GetCommit 获取提交信息
func GetCommit() string {
	return GoString(sdCommit())
}

// GetVersion 获取版本信息
func GetVersion() string {
	return GoString(sdVersion())
}

// CreateSdCtx 创建上下文
func CreateSdCtx(params *SdCtxParams) *SdCtx {
	return newSdCtx(params)
}

// FreeSdCtx 释放上下文
func FreeSdCtx(ctx *SdCtx) {
	freeSdCtx(ctx)
}

// GenerateImage 生成图像
func GenerateImage(ctx *SdCtx, params *SdImgGenParams) *SdImage {
	return generateImage(ctx, params)
}

// GenerateVideo 生成视频
func GenerateVideo(ctx *SdCtx, params *SdVidGenParams, numFramesOut *int) *SdImage {
	return generateVideo(ctx, params, numFramesOut)
}

// CreateUpscalerCtx 创建超分辨率上下文
func CreateUpscalerCtx(esrganPath string, offloadParamsToCpu, direct bool, nThreads, tileSize int) *UpscalerCtx {
	return newUpscalerCtx(CString(esrganPath), offloadParamsToCpu, direct, nThreads, tileSize)
}

// FreeUpscalerCtx 释放超分辨率上下文
func FreeUpscalerCtx(ctx *UpscalerCtx) {
	freeUpscalerCtx(ctx)
}

// Upscale 执行超分辨率
func Upscale(ctx *UpscalerCtx, input SdImage, factor uint32) SdImage {
	return upscale(ctx, input, factor)
}

// GetUpscaleFactor 获取超分辨率因子
func GetUpscaleFactor(ctx *UpscalerCtx) int {
	return getUpscaleFactor(ctx)
}

// Convert 转换模型
func Convert(inputPath, vaePath, outputPath string, outputType SdType, tensorTypeRules string, convertName bool) bool {
	return convert(CString(inputPath), CString(vaePath), CString(outputPath), outputType, CString(tensorTypeRules), convertName)
}

// PreprocessCanny 预处理Canny边缘检测
func PreprocessCanny(image SdImage, highThreshold, lowThreshold, weak, strong float32, inverse bool) bool {
	return preprocessCanny(image, highThreshold, lowThreshold, weak, strong, inverse)
}

// SdImgGenParamsInit 初始化图像生成参数
func SdImgGenParamsInit(params *SdImgGenParams) {
	sdImgGenParamsInit(params)
}

// SdCtxParamsInit 初始化上下文参数
func SdCtxParamsInit(params *SdCtxParams) {
	sdCtxParamsInit(params)
}

// SdSampleParamsInit 初始化采样参数
func SdSampleParamsInit(params *SdSampleParams) {
	sdSampleParamsInit(params)
}

// SdVidGenParamsInit 初始化视频生成参数
func SdVidGenParamsInit(params *SdVidGenParams) {
	sdVidGenParamsInit(params)
}

// SdCacheParamsInit 初始化缓存参数
func SdCacheParamsInit(params *SdCacheParams) {
	sdCacheParamsInit(params)
}

// GetDefaultSampleMethod 获取默认采样方法
func GetDefaultSampleMethod(ctx *SdCtx) SampleMethod {
	return sdGetDefaultSampleMethod(ctx)
}

// GetDefaultScheduler 获取默认调度器
func GetDefaultScheduler(ctx *SdCtx, method SampleMethod) Scheduler {
	return sdGetDefaultScheduler(ctx, method)
}

// GetTypeName 获取类型名称
func GetTypeName(t SdType) string {
	return GoString(sdTypeName(t))
}

// StrToSdType 字符串转 SdType
func StrToSdType(s string) SdType {
	return strToSdType(CString(s))
}

// GetRngTypeName 获取 RNG 类型名称
func GetRngTypeName(t RngType) string {
	return GoString(sdRngTypeName(t))
}

// StrToRngType 字符串转 RngType
func StrToRngType(s string) RngType {
	return strToRngType(CString(s))
}

// GetSampleMethodName 获取采样方法名称
func GetSampleMethodName(t SampleMethod) string {
	return GoString(sdSampleMethodName(t))
}

// StrToSampleMethod 字符串转 SampleMethod
func StrToSampleMethod(s string) SampleMethod {
	return strToSampleMethod(CString(s))
}

// GetSchedulerName 获取调度器名称
func GetSchedulerName(t Scheduler) string {
	return GoString(sdSchedulerName(t))
}

// StrToScheduler 字符串转 Scheduler
func StrToScheduler(s string) Scheduler {
	return strToScheduler(CString(s))
}

// GetPredictionName 获取预测类型名称
func GetPredictionName(t Prediction) string {
	return GoString(sdPredictionName(t))
}

// StrToPrediction 字符串转 Prediction
func StrToPrediction(s string) Prediction {
	return strToPrediction(CString(s))
}

// GetPreviewName 获取预览类型名称
func GetPreviewName(t Preview) string {
	return GoString(sdPreviewName(t))
}

// StrToPreview 字符串转 Preview
func StrToPreview(s string) Preview {
	return strToPreview(CString(s))
}

// GetLoraApplyModeName 获取 LoRA 应用模式名称
func GetLoraApplyModeName(t LoraApplyMode) string {
	return GoString(sdLoraApplyModeName(t))
}

// StrToLoraApplyMode 字符串转 LoraApplyMode
func StrToLoraApplyMode(s string) LoraApplyMode {
	return strToLoraApplyMode(CString(s))
}

// SdCtxParamsToStr 上下文参数转字符串（调试用）。
//
// C 侧返回 malloc(4096) 的字符串缓冲，这里在拷贝成 Go 字符串后
// 立即通过 FreeCPtr 归还，调用方拿到的是独立的 Go 内存。
func SdCtxParamsToStr(params *SdCtxParams) string {
	cstr := sdCtxParamsToStr(params)
	if cstr == nil {
		return ""
	}
	defer FreeCPtr(unsafe.Pointer(cstr))
	return GoString(cstr)
}

// SdSampleParamsToStr 采样参数转字符串（调试用），同样负责释放 C 缓冲。
func SdSampleParamsToStr(params *SdSampleParams) string {
	cstr := sdSampleParamsToStr(params)
	if cstr == nil {
		return ""
	}
	defer FreeCPtr(unsafe.Pointer(cstr))
	return GoString(cstr)
}

// SdImgGenParamsToStr 图像生成参数转字符串（调试用），同样负责释放 C 缓冲。
func SdImgGenParamsToStr(params *SdImgGenParams) string {
	cstr := sdImgGenParamsToStr(params)
	if cstr == nil {
		return ""
	}
	defer FreeCPtr(unsafe.Pointer(cstr))
	return GoString(cstr)
}
