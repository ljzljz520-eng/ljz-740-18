package stablediffusion

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/example/stablediffusion/bindings"
)

// ErrClosed 在 Close/Free 之后继续调用生成等方法时返回。
var ErrClosed = errors.New("stablediffusion: context already closed")

// Context 表示stable-diffusion的上下文。
//
// 并发与生命周期：
//   - Context 持有 C 堆资源，不是线程安全的，同一时刻只允许一次
//     GenerateImage / GenerateVideo 调用（内部用 mu 串行化）。
//   - Close（或 Free）释放底层 C 上下文，之后任何生成/查询调用
//     都会返回 ErrClosed。Close 可安全重复调用。
//   - 回调（日志/进度/预览）是进程级全局的，即使 Context 关闭后
//     仍可能被其他上下文触发；它们的生命周期独立于 Context 管理。
type Context struct {
	ctx *bindings.SdCtx

	mu     sync.Mutex
	closed bool
}

// Image 表示生成的图像。
//
// 进入 Go 层的 Image 始终持有自己拥有的 Go 字节切片（Data），
// 与底层 C 缓冲无关，可自由传递、长期保存并由 GC 回收。
// 作为*输入*传给生成/超分/预处理函数时，Data 的底层内存在该次
// 同步调用期间不得被改写或释放（Go GC 会保证不被移动/回收，
// 调用方只需避免并发写入）。
type Image struct {
	Width   uint32
	Height  uint32
	Channel uint32
	Data    []byte
}

// Lora 表示LoRA模型
type Lora struct {
	Path        string
	Multiplier  float32
	IsHighNoise bool
}

// SamplerConfig 表示采样器配置
type SamplerConfig struct {
	Scheduler    bindings.Scheduler
	Method       bindings.SampleMethod
	Steps        int
	Eta          float32
	TxtCfg       float32
	ImgCfg       float32
	DistilledCfg float32
}

// TilingParams 表示平铺渲染参数
type TilingParams struct {
	Enabled       bool
	TileSizeX     int
	TileSizeY     int
	TargetOverlap float32
	RelSizeX      float32
	RelSizeY      float32
}

// CacheParams 表示缓存配置参数
type CacheParams struct {
	Mode                     bindings.SdCacheMode
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
	ScmMask                  string
	ScmPolicyDynamic         bool
}

// GenerationConfig 表示图像生成配置
type GenerationConfig struct {
	Prompt             string
	NegativePrompt     string
	Width              int
	Height             int
	Seed               int64
	Strength           float32
	BatchCount         int
	ClipSkip           int
	Loras              []Lora
	ControlImage       *Image
	ControlStrength    float32
	InitImage          *Image
	MaskImage          *Image
	RefImages          []Image
	AutoResizeRefImage bool
	IncreaseRefIndex   bool
	Sampler            SamplerConfig
	VaeTilingParams    TilingParams
	Cache              CacheParams
}

// Upscaler 表示超分辨率器。
//
// 与 Context 相同：内部持有 C 堆资源，调用需串行，Close 后
// 不得再调用 Upscale。
type Upscaler struct {
	ctx *bindings.UpscalerCtx

	mu     sync.Mutex
	closed bool
}

// NewUpscaler 创建一个新的超分辨率器。
//
// 路径字符串在 bindings 层转成临时 NUL 缓冲并同步传入，底层构造时
// 会拷贝（std::string），调用返回后由 Go GC 回收，无需手动释放。
func NewUpscaler(modelPath string) (*Upscaler, error) {
	ctx := bindings.CreateUpscalerCtx(modelPath, false, true, -1, 0)
	if ctx == nil {
		return nil, fmt.Errorf("failed to create upscaler")
	}

	return &Upscaler{ctx: ctx}, nil
}

// Close 释放底层超分辨率上下文，可重复调用。Close 后不得再 Upscale。
func (u *Upscaler) Close() error {
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.closed {
		return nil
	}
	u.closed = true
	if u.ctx != nil {
		bindings.FreeUpscalerCtx(u.ctx)
		u.ctx = nil
	}
	return nil
}

// Free 是 Close 的旧名称。
//
// Deprecated: 请使用 Close；调用后不得继续超分。
func (u *Upscaler) Free() {
	_ = u.Close()
}

// imageDataPtr 返回图像像素缓冲的 C 指针，并校验尺寸/缓冲一致性。
func imageDataPtr(img *Image) (*byte, int, error) {
	if img == nil {
		return nil, 0, nil
	}
	size := int(img.Width) * int(img.Height) * int(img.Channel)
	if size == 0 {
		return nil, 0, fmt.Errorf("image has zero dimensions")
	}
	if len(img.Data) < size {
		return nil, 0, fmt.Errorf("image data too short: have %d bytes, need %d (%dx%dx%d)",
			len(img.Data), size, img.Width, img.Height, img.Channel)
	}
	return bindings.BytePtr(img.Data), size, nil
}

// toSdImage 把 Go Image 转成按值传递给 C 的 SdImage。
// 返回的 keepAlive 必须在 C 调用结束后执行，以保活 Go 像素缓冲。
func toSdImage(img *Image) (bindings.SdImage, func(), error) {
	var zero bindings.SdImage
	if img == nil {
		// 空图像：data 保持 nil，由 C 侧按“未提供”处理
		return zero, func() {}, nil
	}
	p, _, err := imageDataPtr(img)
	if err != nil {
		return zero, func() {}, err
	}
	return bindings.SdImage{
			Width:   img.Width,
			Height:  img.Height,
			Channel: img.Channel,
			Data:    p,
		}, func() {
			runtime.KeepAlive(img.Data)
		},
		nil
}

// Upscale 执行超分辨率。
//
// 输入 img.Data 在同步调用期间必须有效且不被并发改写。
// 底层返回的图像缓冲是 malloc 分配的，拷贝进 Go Image 后立即
// 用 libc free 归还；生成失败（返回 data == nil）时同样安全返回，
// 不会泄漏。
func (u *Upscaler) Upscale(img *Image, factor uint32) (*Image, error) {
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.closed || u.ctx == nil {
		return nil, ErrClosed
	}
	if img == nil {
		return nil, fmt.Errorf("input image is nil")
	}

	input, keepAlive, err := toSdImage(img)
	if err != nil {
		return nil, err
	}

	// 执行超分辨率（同步）
	result := bindings.Upscale(u.ctx, input, factor)
	keepAlive()

	// 底层失败时返回零值 SdImage（Data == nil）；结构体按值返回无需释放
	if result.Data == nil {
		return nil, fmt.Errorf("upscale failed (empty result from backend)")
	}
	size := int(result.Width) * int(result.Height) * int(result.Channel)
	if size == 0 {
		bindings.FreeImageData(&result)
		return nil, fmt.Errorf("upscale returned image with zero dimensions")
	}

	// 拷贝进 Go 内存，随后释放底层缓冲
	data := make([]byte, size)
	copy(data, unsafe.Slice(result.Data, size))
	bindings.FreeImageData(&result)

	return &Image{
		Width:   result.Width,
		Height:  result.Height,
		Channel: result.Channel,
		Data:    data,
	}, nil
}

// ContextOptions 定义了stable-diffusion上下文的完整配置参数
type ContextOptions struct {
	ModelPath                   string
	ClipLPath                   string
	ClipGPath                   string
	ClipVisionPath              string
	T5xxlPath                   string
	LlmPath                     string
	LlmVisionPath               string
	DiffusionModelPath          string
	HighNoiseDiffusionModelPath string
	VaePath                     string
	TaesdPath                   string
	ControlNetPath              string
	PhotoMakerPath              string
	TensorTypeRules             string
	VaeDecodeOnly               bool
	FreeParamsImmediately       bool
	NThreads                    int
	Wtype                       bindings.SdType
	RngType                     bindings.RngType
	SamplerRngType              bindings.RngType
	Prediction                  bindings.Prediction
	LoraApplyMode               bindings.LoraApplyMode
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
	TilingParams                TilingParams
	CacheParams                 CacheParams
}

// DefaultContextOptions 返回具有默认参数的上下文选项
func DefaultContextOptions(modelPath string) ContextOptions {
	return ContextOptions{
		ModelPath:  modelPath,
		NThreads:   -1, // 自动
		Wtype:      bindings.SD_TYPE_F16,
		EnableMmap: true,
	}
}

// NewContext 创建一个新的stable-diffusion上下文。
//
// 路径字符串以 CString（Go 堆 + NUL）形式同步传入；底层在构造时
// 会拷贝为 std::string，因此调用返回后这些临时缓冲即可被 GC，
// 无需手动释放（也绝不能用 C free 释放）。
func NewContext(options ContextOptions) (*Context, error) {
	// 初始化上下文参数
	params := &bindings.SdCtxParams{}
	bindings.SdCtxParamsInit(params)

	// 所有 CString 分配都由 Go GC 管理。alloc 收集其所属的字节分配，
	// 在本次同步 C 调用结束后统一 KeepAlive，保证调用期间不被回收。
	var alloc [][]byte
	cstr := func(s string) *byte {
		if s == "" {
			return nil
		}
		b := append([]byte(s), 0)
		alloc = append(alloc, b)
		return &b[0]
	}

	if options.ModelPath != "" {
		params.ModelPath = cstr(options.ModelPath)
	}
	if options.ClipLPath != "" {
		params.ClipLPath = cstr(options.ClipLPath)
	}
	if options.ClipGPath != "" {
		params.ClipGPath = cstr(options.ClipGPath)
	}
	if options.ClipVisionPath != "" {
		params.ClipVisionPath = cstr(options.ClipVisionPath)
	}
	if options.T5xxlPath != "" {
		params.T5xxlPath = cstr(options.T5xxlPath)
	}
	if options.LlmPath != "" {
		params.LlmPath = cstr(options.LlmPath)
	}
	if options.LlmVisionPath != "" {
		params.LlmVisionPath = cstr(options.LlmVisionPath)
	}
	if options.DiffusionModelPath != "" {
		params.DiffusionModelPath = cstr(options.DiffusionModelPath)
	}
	if options.HighNoiseDiffusionModelPath != "" {
		params.HighNoiseDiffusionModelPath = cstr(options.HighNoiseDiffusionModelPath)
	}
	if options.VaePath != "" {
		params.VaePath = cstr(options.VaePath)
	}
	if options.TaesdPath != "" {
		params.TaesdPath = cstr(options.TaesdPath)
	}
	if options.ControlNetPath != "" {
		params.ControlNetPath = cstr(options.ControlNetPath)
	}
	if options.PhotoMakerPath != "" {
		params.PhotoMakerPath = cstr(options.PhotoMakerPath)
	}
	if options.TensorTypeRules != "" {
		params.TensorTypeRules = cstr(options.TensorTypeRules)
	}

	params.VaeDecodeOnly = options.VaeDecodeOnly
	params.FreeParamsImmediately = options.FreeParamsImmediately
	params.NThreads = options.NThreads
	params.Wtype = options.Wtype
	params.RngType = options.RngType
	params.SamplerRngType = options.SamplerRngType
	params.Prediction = options.Prediction
	params.LoraApplyMode = options.LoraApplyMode
	params.OffloadParamsToCpu = options.OffloadParamsToCpu
	params.EnableMmap = options.EnableMmap
	params.KeepClipOnCpu = options.KeepClipOnCpu
	params.KeepControlNetOnCpu = options.KeepControlNetOnCpu
	params.KeepVaeOnCpu = options.KeepVaeOnCpu
	params.FlashAttn = options.FlashAttn
	params.DiffusionFlashAttn = options.DiffusionFlashAttn
	params.TaePreviewOnly = options.TaePreviewOnly
	params.DiffusionConvDirect = options.DiffusionConvDirect
	params.VaeConvDirect = options.VaeConvDirect
	params.CircularX = options.CircularX
	params.CircularY = options.CircularY
	params.ForceSdxlVaeConvScale = options.ForceSdxlVaeConvScale
	params.ChromaUseDitMask = options.ChromaUseDitMask
	params.ChromaUseT5Mask = options.ChromaUseT5Mask
	params.ChromaT5MaskPad = options.ChromaT5MaskPad
	params.QwenImageZeroCondT = options.QwenImageZeroCondT
	params.FlowShift = options.FlowShift

	// 设置 Tiling 和 Cache
	params.CircularX = options.CircularX
	params.CircularY = options.CircularY

	// 注意：bindings 层的某些复杂结构体初始化可能需要显式处理

	// 创建上下文（同步调用，返回后底层不再持有上述字符串指针）
	ctx := bindings.CreateSdCtx(params)
	// 确保所有 CString 临时缓冲在整个 C 调用期间存活
	for i := range alloc {
		runtime.KeepAlive(alloc[i])
	}
	if ctx == nil {
		return nil, fmt.Errorf("failed to create context")
	}

	return &Context{ctx: ctx}, nil
}

// Close 释放底层 C 上下文。Close 之后不得再调用 GenerateImage、
// GenerateVideo 或任何依赖该上下文的方法，否则返回 ErrClosed
// （C 侧此时已是悬空指针，继续生成会导致 use-after-free 崩溃）。
// 可安全重复调用。
func (c *Context) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	if c.ctx != nil {
		bindings.FreeSdCtx(c.ctx)
		c.ctx = nil
	}
	return nil
}

// Free 是 Close 的旧名称，语义相同。
//
// Deprecated: 请使用 Close；调用后不得继续生成。
func (c *Context) Free() {
	_ = c.Close()
}

// GetUpscaleFactor 获取超分辨率因子
func (u *Upscaler) GetUpscaleFactor() int {
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.closed || u.ctx == nil {
		return 0
	}
	return bindings.GetUpscaleFactor(u.ctx)
}

// GenerateImage 生成图像。
//
// 生命周期约定：
//   - cfg 中的 prompt、LoRA 路径等字符串以及各输入图像缓冲都只在
//     本次同步调用期间借给 C 使用，调用返回后底层不再持有它们，
//     由 Go GC 回收；调用期间不得并发改写输入图像数据。
//   - 底层返回的图像数组（calloc）与每帧缓冲（malloc）在拷贝成
//     Go Image 后立即释放。即使底层返回失败、部分帧为空，也保证
//     不泄漏已返回的资源。
//   - Context 已 Close 时返回 ErrClosed；调用全程持有内部锁，
//     与同一 Context 上的其他生成调用串行执行。
func (c *Context) GenerateImage(cfg GenerationConfig) ([]*Image, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed || c.ctx == nil {
		return nil, ErrClosed
	}

	// 初始化图像生成参数
	params := &bindings.SdImgGenParams{}
	bindings.SdImgGenParamsInit(params)

	// keep 收集所有“借给 C”的 Go 堆分配（字符串字节、结构体切片、
	// 输入图像缓冲），C 调用结束后统一 KeepAlive。
	var keep []any
	keepAlive := func() {
		for _, x := range keep {
			runtime.KeepAlive(x)
		}
	}
	cstr := func(s string) *byte {
		if s == "" {
			return nil
		}
		b := append([]byte(s), 0)
		keep = append(keep, b)
		return &b[0]
	}

	// 设置基本参数
	params.Prompt = cstr(cfg.Prompt)
	params.NegativePrompt = cstr(cfg.NegativePrompt)
	params.Width = cfg.Width
	params.Height = cfg.Height
	params.Seed = cfg.Seed
	params.Strength = cfg.Strength
	params.BatchCount = cfg.BatchCount
	params.ClipSkip = cfg.ClipSkip

	// 设置采样器参数
	params.SampleParams.Scheduler = cfg.Sampler.Scheduler
	params.SampleParams.SampleMethod = cfg.Sampler.Method
	params.SampleParams.SampleSteps = cfg.Sampler.Steps
	params.SampleParams.Eta = cfg.Sampler.Eta
	params.SampleParams.Guidance.TxtCfg = cfg.Sampler.TxtCfg
	params.SampleParams.Guidance.ImgCfg = cfg.Sampler.ImgCfg
	params.SampleParams.Guidance.DistilledGuidance = cfg.Sampler.DistilledCfg

	// 设置LoRA：结构体数组连同其中的路径字符串在调用期间必须连续存活
	if len(cfg.Loras) > 0 {
		loras := make([]bindings.SdLora, len(cfg.Loras))
		for i, lora := range cfg.Loras {
			loras[i] = bindings.SdLora{
				Path:        cstr(lora.Path),
				Multiplier:  lora.Multiplier,
				IsHighNoise: lora.IsHighNoise,
			}
		}
		params.Loras = &loras[0]
		params.LoraCount = uint32(len(loras))
		keep = append(keep, loras)
	}

	// 把 Go Image 转成按值传给 C 的 SdImage；返回的保活闭包记录其像素缓冲
	bindImage := func(img *Image, what string) (bindings.SdImage, error) {
		var zero bindings.SdImage
		sdImg, ka, err := toSdImage(img)
		if err != nil {
			return zero, fmt.Errorf("%s: %w", what, err)
		}
		keep = append(keep, ka)
		return sdImg, nil
	}

	// 设置控制图像
	if cfg.ControlImage != nil {
		sdImg, err := bindImage(cfg.ControlImage, "control image")
		if err != nil {
			keepAlive()
			return nil, err
		}
		params.ControlImage = sdImg
		params.ControlStrength = cfg.ControlStrength
	}

	// 设置初始图像
	if cfg.InitImage != nil {
		sdImg, err := bindImage(cfg.InitImage, "init image")
		if err != nil {
			keepAlive()
			return nil, err
		}
		params.InitImage = sdImg
	}

	// 设置掩码图像
	if cfg.MaskImage != nil {
		sdImg, err := bindImage(cfg.MaskImage, "mask image")
		if err != nil {
			keepAlive()
			return nil, err
		}
		params.MaskImage = sdImg
	}

	// 设置参考图像
	if len(cfg.RefImages) > 0 {
		refImages := make([]bindings.SdImage, len(cfg.RefImages))
		for i := range cfg.RefImages {
			sdImg, err := bindImage(&cfg.RefImages[i], fmt.Sprintf("ref image %d", i))
			if err != nil {
				keepAlive()
				return nil, err
			}
			refImages[i] = sdImg
		}
		params.RefImages = &refImages[0]
		params.RefImagesCount = len(refImages)
		params.AutoResizeRefImage = cfg.AutoResizeRefImage
		params.IncreaseRefIndex = cfg.IncreaseRefIndex
		keep = append(keep, refImages)
	}

	// 设置 Tiling 和 Cache 参数
	params.VaeTilingParams = bindings.SdTilingParams{
		Enabled:       cfg.VaeTilingParams.Enabled,
		TileSizeX:     cfg.VaeTilingParams.TileSizeX,
		TileSizeY:     cfg.VaeTilingParams.TileSizeY,
		TargetOverlap: cfg.VaeTilingParams.TargetOverlap,
		RelSizeX:      cfg.VaeTilingParams.RelSizeX,
		RelSizeY:      cfg.VaeTilingParams.RelSizeY,
	}

	params.Cache = bindings.SdCacheParams{
		Mode:                     cfg.Cache.Mode,
		ReuseThreshold:           cfg.Cache.ReuseThreshold,
		StartPercent:             cfg.Cache.StartPercent,
		EndPercent:               cfg.Cache.EndPercent,
		ErrorDecayRate:           cfg.Cache.ErrorDecayRate,
		UseRelativeThreshold:     cfg.Cache.UseRelativeThreshold,
		ResetErrorOnCompute:      cfg.Cache.ResetErrorOnCompute,
		FnComputeBlocks:          cfg.Cache.FnComputeBlocks,
		BnComputeBlocks:          cfg.Cache.BnComputeBlocks,
		ResidualDiffThreshold:    cfg.Cache.ResidualDiffThreshold,
		MaxWarmupSteps:           cfg.Cache.MaxWarmupSteps,
		MaxCachedSteps:           cfg.Cache.MaxCachedSteps,
		MaxContinuousCachedSteps: cfg.Cache.MaxContinuousCachedSteps,
		TaylorseerNDerivatives:   cfg.Cache.TaylorseerNDerivatives,
		TaylorseerSkipInterval:   cfg.Cache.TaylorseerSkipInterval,
		ScmPolicyDynamic:         cfg.Cache.ScmPolicyDynamic,
	}
	if cfg.Cache.ScmMask != "" {
		params.Cache.ScmMask = cstr(cfg.Cache.ScmMask)
	}

	// 生成图像（同步调用）。无论成功失败，返回后先保活所有借出的
	// Go 分配，再处理 C 返回的资源。
	result := bindings.GenerateImage(c.ctx, params)
	keepAlive()
	if result == nil {
		return nil, fmt.Errorf("failed to generate image")
	}

	// 数组长度即请求的 batch_count；逐槽拷贝，data == nil 的失败槽位
	// 跳过。数组本身及所有非空缓冲最后统一释放。
	count := cfg.BatchCount
	cImages := unsafe.Slice(result, count)

	images := make([]*Image, 0, count)
	for i := 0; i < count; i++ {
		img := &cImages[i]
		if img.Data == nil {
			continue
		}
		size := int(img.Width) * int(img.Height) * int(img.Channel)
		if size == 0 {
			continue
		}
		data := make([]byte, size)
		copy(data, unsafe.Slice(img.Data, size))
		images = append(images, &Image{
			Width:   img.Width,
			Height:  img.Height,
			Channel: img.Channel,
			Data:    data,
		})
	}

	// 释放整个 C 结果（内部逐帧 free(data) 再 free(数组)）
	bindings.FreeImageArray(result, count)

	if len(images) == 0 {
		return nil, fmt.Errorf("failed to generate image: backend returned no frames")
	}
	return images, nil
}

// ConvertModel 转换模型格式。
//
// 路径字符串同步传入，底层打开/转换期间使用，返回后即由 Go GC 回收。
func ConvertModel(inputPath, vaePath, outputPath string, outputType bindings.SdType) error {
	in := append([]byte(inputPath), 0)
	vae := append([]byte(vaePath), 0)
	out := append([]byte(outputPath), 0)
	success := bindings.Convert(inputPath, vaePath, outputPath, outputType, "", true)
	runtime.KeepAlive(in)
	runtime.KeepAlive(vae)
	runtime.KeepAlive(out)
	if !success {
		return fmt.Errorf("failed to convert model")
	}
	return nil
}

// GetDefaultSampleMethod 获取默认采样方法。Context 已 Close 时返回 0。
func (c *Context) GetDefaultSampleMethod() bindings.SampleMethod {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed || c.ctx == nil {
		return 0
	}
	return bindings.GetDefaultSampleMethod(c.ctx)
}

// GetDefaultScheduler 获取默认调度器。Context 已 Close 时返回 0。
func (c *Context) GetDefaultScheduler(method bindings.SampleMethod) bindings.Scheduler {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed || c.ctx == nil {
		return 0
	}
	return bindings.GetDefaultScheduler(c.ctx, method)
}

// GetTypeName 获取类型名称
func GetTypeName(t bindings.SdType) string {
	return bindings.GetTypeName(t)
}

// GetSystemInfo 获取系统信息
func GetSystemInfo() string {
	return bindings.GetSystemInfo()
}

// GetVersion 获取版本信息
func GetVersion() string {
	return bindings.GetVersion()
}

// GetCommit 获取提交信息
func GetCommit() string {
	return bindings.GetCommit()
}

// PreprocessCanny 对 img.Data 原地执行 Canny 边缘检测。
//
// 生命周期：底层按值接收 SdImage 并直接写回其 data 指向的缓冲，
// 不分配/返回新的图像内存，因此本函数没有需要释放的 C 资源。
// 调用期间 img.Data 必须有效且不得从其他 goroutine 并发读写；
// 成功后 img.Data 仍是同一个 Go 切片（内容被改写），无需重新切片。
// img 为 nil、尺寸为 0 或缓冲不足时返回错误，不会触发底层调用。
func PreprocessCanny(img *Image, highThreshold, lowThreshold, weak, strong float32, inverse bool) error {
	input, keepAlive, err := toSdImage(img)
	if err != nil {
		return err
	}
	if input.Data == nil {
		return fmt.Errorf("image has no pixel data")
	}

	success := bindings.PreprocessCanny(input, highThreshold, lowThreshold, weak, strong, inverse)
	// C 只是原地改写了 img.Data 指向的 Go 内存
	keepAlive()
	if !success {
		return fmt.Errorf("failed to preprocess canny")
	}
	return nil
}

// VideoGenerationConfig 表示视频生成配置
type VideoGenerationConfig struct {
	Prompt           string
	NegativePrompt   string
	Width            int
	Height           int
	Seed             int64
	Strength         float32
	ClipSkip         int
	Loras            []Lora
	ControlFrames    []Image
	InitImage        *Image
	EndImage         *Image
	VideoFrames      int
	MoeBoundary      float32
	VaceStrength     float32
	Sampler          SamplerConfig
	HighNoiseSampler SamplerConfig
}

// GenerateVideo 生成视频帧。资源生命周期约定与 GenerateImage 相同：
// 字符串与输入帧缓冲仅在同步调用期间借给 C；底层返回的帧数组与
// 每帧缓冲拷贝进 Go Image 后立即释放，失败路径同样不泄漏；
// Context 已 Close 时返回 ErrClosed。
func (c *Context) GenerateVideo(cfg VideoGenerationConfig) ([]*Image, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed || c.ctx == nil {
		return nil, ErrClosed
	}

	params := &bindings.SdVidGenParams{}
	bindings.SdVidGenParamsInit(params)

	var keep []any
	keepAlive := func() {
		for _, x := range keep {
			runtime.KeepAlive(x)
		}
	}
	cstr := func(s string) *byte {
		if s == "" {
			return nil
		}
		b := append([]byte(s), 0)
		keep = append(keep, b)
		return &b[0]
	}

	params.Prompt = cstr(cfg.Prompt)
	params.NegativePrompt = cstr(cfg.NegativePrompt)
	params.Width = cfg.Width
	params.Height = cfg.Height
	params.Seed = cfg.Seed
	params.Strength = cfg.Strength
	params.ClipSkip = cfg.ClipSkip
	params.VideoFrames = cfg.VideoFrames
	params.MoeBoundary = cfg.MoeBoundary
	params.VaceStrength = cfg.VaceStrength

	// 设置采样器参数
	params.SampleParams.Scheduler = cfg.Sampler.Scheduler
	params.SampleParams.SampleMethod = cfg.Sampler.Method
	params.SampleParams.SampleSteps = cfg.Sampler.Steps
	params.SampleParams.Eta = cfg.Sampler.Eta
	params.SampleParams.Guidance.TxtCfg = cfg.Sampler.TxtCfg
	params.SampleParams.Guidance.ImgCfg = cfg.Sampler.ImgCfg
	params.SampleParams.Guidance.DistilledGuidance = cfg.Sampler.DistilledCfg

	// 设置高噪声采样器参数
	params.HighNoiseSampleParams.Scheduler = cfg.HighNoiseSampler.Scheduler
	params.HighNoiseSampleParams.SampleMethod = cfg.HighNoiseSampler.Method
	params.HighNoiseSampleParams.SampleSteps = cfg.HighNoiseSampler.Steps
	params.HighNoiseSampleParams.Eta = cfg.HighNoiseSampler.Eta
	params.HighNoiseSampleParams.Guidance.TxtCfg = cfg.HighNoiseSampler.TxtCfg
	params.HighNoiseSampleParams.Guidance.ImgCfg = cfg.HighNoiseSampler.ImgCfg
	params.HighNoiseSampleParams.Guidance.DistilledGuidance = cfg.HighNoiseSampler.DistilledCfg

	// 设置LoRA
	if len(cfg.Loras) > 0 {
		loras := make([]bindings.SdLora, len(cfg.Loras))
		for i, lora := range cfg.Loras {
			loras[i] = bindings.SdLora{
				Path:        cstr(lora.Path),
				Multiplier:  lora.Multiplier,
				IsHighNoise: lora.IsHighNoise,
			}
		}
		params.Loras = &loras[0]
		params.LoraCount = uint32(len(loras))
		keep = append(keep, loras)
	}

	// 输入帧缓冲在调用期间必须存活
	bindImage := func(img *Image, what string) (bindings.SdImage, error) {
		var zero bindings.SdImage
		sdImg, ka, err := toSdImage(img)
		if err != nil {
			return zero, fmt.Errorf("%s: %w", what, err)
		}
		keep = append(keep, ka)
		return sdImg, nil
	}

	// 设置初始和结束图像
	if cfg.InitImage != nil {
		sdImg, err := bindImage(cfg.InitImage, "init image")
		if err != nil {
			keepAlive()
			return nil, err
		}
		params.InitImage = sdImg
	}

	if cfg.EndImage != nil {
		sdImg, err := bindImage(cfg.EndImage, "end image")
		if err != nil {
			keepAlive()
			return nil, err
		}
		params.EndImage = sdImg
	}

	// 设置控制帧
	if len(cfg.ControlFrames) > 0 {
		controlFrames := make([]bindings.SdImage, len(cfg.ControlFrames))
		for i := range cfg.ControlFrames {
			sdImg, err := bindImage(&cfg.ControlFrames[i], fmt.Sprintf("control frame %d", i))
			if err != nil {
				keepAlive()
				return nil, err
			}
			controlFrames[i] = sdImg
		}
		params.ControlFrames = &controlFrames[0]
		params.ControlFramesSize = len(controlFrames)
		keep = append(keep, controlFrames)
	}

	// 生成视频（同步调用）
	var numFramesOut int
	result := bindings.GenerateVideo(c.ctx, params, &numFramesOut)
	keepAlive()
	if result == nil || numFramesOut <= 0 {
		return nil, fmt.Errorf("failed to generate video")
	}

	// 按底层实际返回的帧数逐帧拷贝；空槽位跳过
	cImages := unsafe.Slice(result, numFramesOut)
	images := make([]*Image, 0, numFramesOut)
	for i := 0; i < numFramesOut; i++ {
		img := &cImages[i]
		if img.Data == nil {
			continue
		}
		size := int(img.Width) * int(img.Height) * int(img.Channel)
		if size == 0 {
			continue
		}
		data := make([]byte, size)
		copy(data, unsafe.Slice(img.Data, size))
		images = append(images, &Image{
			Width:   img.Width,
			Height:  img.Height,
			Channel: img.Channel,
			Data:    data,
		})
	}

	// 释放整个 C 结果（逐帧 free(data) 再 free(数组)）
	bindings.FreeImageArray(result, numFramesOut)

	if len(images) == 0 {
		return nil, fmt.Errorf("failed to generate video: backend returned no frames")
	}
	return images, nil
}

// GetNumPhysicalCores 获取物理核心数
func GetNumPhysicalCores() int32 {
	return bindings.GetNumPhysicalCores()
}

// LogFunc 是类型安全的日志回调：text 已转成 Go 字符串。
type LogFunc func(level bindings.SdLogLevel, text string)

// ProgressFunc 是类型安全的进度回调。
type ProgressFunc func(step, steps int, time float32)

// PreviewFunc 是类型安全的预览回调。
//
// frames 是本次回调瞬时有效的 C 内存视图，回调返回后即失效；
// 如需保留必须在回调内拷贝（CopyPreviewFrames 可批量深拷贝）。
type PreviewFunc func(step, frameCount int, frames []Image, isNoisy bool)

// CopyPreviewImage 深拷贝一帧预览图（C 内存 -> Go 内存），
// 尺寸异常或 data 为空时返回零值与 false。
func CopyPreviewImage(src *bindings.SdImage) (Image, bool) {
	if src == nil || src.Data == nil {
		return Image{}, false
	}
	size := int(src.Width) * int(src.Height) * int(src.Channel)
	if size == 0 {
		return Image{}, false
	}
	data := make([]byte, size)
	copy(data, unsafe.Slice(src.Data, size))
	return Image{Width: src.Width, Height: src.Height, Channel: src.Channel, Data: data}, true
}

// CopyPreviewFrames 深拷贝整组预览帧。
func CopyPreviewFrames(frames *bindings.SdImage, count int) []Image {
	if frames == nil || count <= 0 {
		return nil
	}
	src := unsafe.Slice(frames, count)
	out := make([]Image, 0, count)
	for i := range src {
		if img, ok := CopyPreviewImage(&src[i]); ok {
			out = append(out, img)
		}
	}
	return out
}

// OnLog 注册进程级日志回调，返回的 Registration 用于注销。
// 回调不携带用户上下文（通常只写日志）；注册前的旧回调会被替换。
//
// 注意：回调是进程全局的，与具体 Context 无关；Context Close 后，
// 只要没有注销，其他上下文生成时仍会触发该回调。
func OnLog(cb LogFunc) *bindings.CallbackRegistration {
	if cb == nil {
		bindings.UnsetLogCallback()
		return &bindings.CallbackRegistration{}
	}
	return bindings.RegisterLogCallback(func(level bindings.SdLogLevel, text *byte, _ unsafe.Pointer) {
		cb(level, bindings.GoString(text))
	}, nil)
}

// OnProgress 注册进程级进度回调，生命周期语义同 OnLog。
func OnProgress(cb ProgressFunc) *bindings.CallbackRegistration {
	if cb == nil {
		bindings.UnsetProgressCallback()
		return &bindings.CallbackRegistration{}
	}
	return bindings.RegisterProgressCallback(func(step, steps int, t float32, _ unsafe.Pointer) {
		cb(step, steps, t)
	}, nil)
}

// OnPreview 注册进程级预览回调，mode/interval/denoised/noisy 透传给
// 底层；frames 仅在回调执行期间有效。生命周期语义同 OnLog。
func OnPreview(cb PreviewFunc, mode bindings.Preview, interval int, denoised, noisy bool) *bindings.CallbackRegistration {
	if cb == nil {
		bindings.UnsetPreviewCallback()
		return &bindings.CallbackRegistration{}
	}
	return bindings.RegisterPreviewCallback(func(step, frameCount int, frames *bindings.SdImage, isNoisy bool, _ unsafe.Pointer) {
		// 只把 C 内存包成切片视图传入；cb 内若需保留必须自行深拷贝
		var view []Image
		if frames != nil && frameCount > 0 {
			src := unsafe.Slice(frames, frameCount)
			view = make([]Image, frameCount)
			for i := range src {
				size := int(src[i].Width) * int(src[i].Height) * int(src[i].Channel)
				if src[i].Data != nil && size > 0 {
					view[i] = Image{
						Width:   src[i].Width,
						Height:  src[i].Height,
						Channel: src[i].Channel,
						// 直接引用 C 内存的只读视图，仅在本次回调内有效
						Data: unsafe.Slice(src[i].Data, size),
					}
				}
			}
		}
		cb(step, frameCount, view, isNoisy)
	}, mode, interval, denoised, noisy, nil)
}

// SetLogCallback 透传裸指针上下文给底层。
//
// Deprecated: 请使用 OnLog 或 bindings.RegisterLogCallback。
// 本函数不管理 data 的生命周期：调用方必须保证 data 指向的 Go 对象
// 在回调被注销前一直存活（可通过 runtime.KeepAlive 或长期持有者）。
func SetLogCallback(cb bindings.SdLogCb, data unsafe.Pointer) {
	bindings.SetLogCallback(cb, data)
}

// SetProgressCallback 透传裸指针上下文，生命周期风险同 SetLogCallback。
//
// Deprecated: 请使用 OnProgress 或 bindings.RegisterProgressCallback。
func SetProgressCallback(cb bindings.SdProgressCb, data unsafe.Pointer) {
	bindings.SetProgressCallback(cb, data)
}

// SetPreviewCallback 透传裸指针上下文，生命周期风险同 SetLogCallback。
// frames 内存仅在回调期间有效，见 PreviewFunc 文档。
//
// Deprecated: 请使用 OnPreview 或 bindings.RegisterPreviewCallback。
func SetPreviewCallback(cb bindings.SdPreviewCb, mode bindings.Preview, interval int, denoised, noisy bool, data unsafe.Pointer) {
	bindings.SetPreviewCallback(cb, mode, interval, denoised, noisy, data)
}

// --- 类型名称 / 字符串互转（完整覆盖 stable-diffusion.h） ---

// StrToSdType 将字符串转换为 SdType
func StrToSdType(s string) bindings.SdType {
	return bindings.StrToSdType(s)
}

// GetRngTypeName 获取 RNG 类型名称
func GetRngTypeName(t bindings.RngType) string {
	return bindings.GetRngTypeName(t)
}

// StrToRngType 将字符串转换为 RngType
func StrToRngType(s string) bindings.RngType {
	return bindings.StrToRngType(s)
}

// GetSampleMethodName 获取采样方法名称
func GetSampleMethodName(t bindings.SampleMethod) string {
	return bindings.GetSampleMethodName(t)
}

// StrToSampleMethod 将字符串转换为 SampleMethod
func StrToSampleMethod(s string) bindings.SampleMethod {
	return bindings.StrToSampleMethod(s)
}

// GetSchedulerName 获取调度器名称
func GetSchedulerName(t bindings.Scheduler) string {
	return bindings.GetSchedulerName(t)
}

// StrToScheduler 将字符串转换为 Scheduler
func StrToScheduler(s string) bindings.Scheduler {
	return bindings.StrToScheduler(s)
}

// GetPredictionName 获取预测类型名称
func GetPredictionName(t bindings.Prediction) string {
	return bindings.GetPredictionName(t)
}

// StrToPrediction 将字符串转换为 Prediction
func StrToPrediction(s string) bindings.Prediction {
	return bindings.StrToPrediction(s)
}

// GetPreviewName 获取预览类型名称
func GetPreviewName(t bindings.Preview) string {
	return bindings.GetPreviewName(t)
}

// StrToPreview 将字符串转换为 Preview
func StrToPreview(s string) bindings.Preview {
	return bindings.StrToPreview(s)
}

// GetLoraApplyModeName 获取 LoRA 应用模式名称
func GetLoraApplyModeName(t bindings.LoraApplyMode) string {
	return bindings.GetLoraApplyModeName(t)
}

// StrToLoraApplyMode 将字符串转换为 LoraApplyMode
func StrToLoraApplyMode(s string) bindings.LoraApplyMode {
	return bindings.StrToLoraApplyMode(s)
}
