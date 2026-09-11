package stablediffusion

import (
	"errors"
	"testing"
)

// 未初始化/已关闭的上下文必须返回 ErrClosed，不能把 nil 传给 C 导致崩溃。
func TestClosedContextRejectsGeneration(t *testing.T) {
	c := &Context{} // ctx == nil，等价于 Close 后的状态

	if _, err := c.GenerateImage(GenerationConfig{BatchCount: 1}); !errors.Is(err, ErrClosed) {
		t.Fatalf("GenerateImage on nil ctx: want ErrClosed, got %v", err)
	}
	if _, err := c.GenerateVideo(VideoGenerationConfig{}); !errors.Is(err, ErrClosed) {
		t.Fatalf("GenerateVideo on nil ctx: want ErrClosed, got %v", err)
	}
	if err := c.Close(); err != nil {
		t.Fatalf("Close on zero context should be safe: %v", err)
	}
	c.Free() // 旧 API 别名，重复调用安全
}

func TestClosedUpscalerRejectsUpscale(t *testing.T) {
	u := &Upscaler{}
	if _, err := u.Upscale(&Image{Width: 1, Height: 1, Channel: 3, Data: make([]byte, 3)}, 2); !errors.Is(err, ErrClosed) {
		t.Fatalf("Upscale on nil ctx: want ErrClosed, got %v", err)
	}
	if u.GetUpscaleFactor() != 0 {
		t.Fatal("GetUpscaleFactor on closed upscaler must be 0")
	}
	if err := u.Close(); err != nil {
		t.Fatalf("Close must be idempotent: %v", err)
	}
}

func TestImageValidation(t *testing.T) {
	u := &Upscaler{} // 用它的 ErrClosed 路径之前先验证输入校验
	if _, err := u.Upscale(nil, 2); !errors.Is(err, ErrClosed) {
		t.Fatalf("want ErrClosed, got %v", err)
	}

	// 缓冲与尺寸不匹配必须在进入 C 之前报错
	bad := &Image{Width: 2, Height: 2, Channel: 3, Data: make([]byte, 1)}
	if _, _, err := toSdImage(bad); err == nil {
		t.Fatal("expected error for short data buffer")
	}
	good := &Image{Width: 2, Height: 2, Channel: 3, Data: make([]byte, 12)}
	sd, ka, err := toSdImage(good)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if sd.Data == nil {
		t.Fatal("expected non-nil data pointer")
	}
	ka()
}

func TestCannyValidation(t *testing.T) {
	if err := PreprocessCanny(nil, 1, 1, 1, 1, false); err == nil {
		t.Fatal("nil image should be rejected")
	}
	bad := &Image{Width: 4, Height: 4, Channel: 3, Data: make([]byte, 5)}
	if err := PreprocessCanny(bad, 1, 1, 1, 1, false); err == nil {
		t.Fatal("short buffer should be rejected")
	}
}
