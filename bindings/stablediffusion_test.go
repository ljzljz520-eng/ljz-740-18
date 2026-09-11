package bindings

import (
	"fmt"
	"testing"
	"unsafe"
)

func TestCStringGoString(t *testing.T) {
	testStr := "Hello, World!"
	cStr := CString(testStr)
	goStr := GoString(cStr)

	if goStr != testStr {
		t.Errorf("Expected %s, got %s", testStr, goStr)
	}

	// Test empty string
	emptyStr := ""
	cEmpty := CString(emptyStr)
	goEmpty := GoString(cEmpty)
	if goEmpty != emptyStr {
		t.Errorf("Expected empty string, got '%s'", goEmpty)
	}

	// Verify null termination using unsafe.Add (no uintptr->Pointer roundtrip)
	if *(*byte)(unsafe.Add(unsafe.Pointer(cEmpty), 0)) != 0 {
		t.Errorf("Expected null terminator")
	}
}

func TestMockImplementation(t *testing.T) {
	// Ensure we rely on mock implementation when lib is not present
	// This assumes the test environment doesn't have the shared library in default paths

	sysInfo := GetSystemInfo()
	if sysInfo == "" {
		t.Error("GetSystemInfo returned empty string")
	}
	fmt.Printf("System Info: %s\n", sysInfo)

	cores := GetNumPhysicalCores()
	if cores <= 0 {
		t.Error("GetNumPhysicalCores returned invalid number")
	}

	version := GetVersion()
	if version == "" {
		t.Error("GetVersion returned empty string")
	}
	fmt.Printf("Version: %s\n", version)
}

func TestCallbackWrapper(t *testing.T) {
	// This test just ensures the wrapper function doesn't panic
	// Actual callback execution would require the mock to call it back,
	// which currently the mock implementations (empty bodies) don't do.
	// However, we can verify the setting logic.

	logCb := func(level SdLogLevel, text *byte, data unsafe.Pointer) {
		fmt.Printf("Log: %s\n", GoString(text))
	}
	SetLogCallback(logCb, nil)

	if currentLogCallback == 0 {
		t.Error("currentLogCallback was not set")
	}
}

func TestBytePtr(t *testing.T) {
	if BytePtr(nil) != nil {
		t.Error("BytePtr(nil) should be nil")
	}
	if BytePtr([]byte{}) != nil {
		t.Error("BytePtr(empty) should be nil")
	}
	b := []byte{1, 2, 3}
	if BytePtr(b) == nil {
		t.Error("BytePtr(non-empty) should not be nil")
	}
}

func TestFreeInMockModeIsNoop(t *testing.T) {
	// 没有真实动态库时 cFree == 0，释放操作必须是安全的空操作
	FreeCPtr(nil)
	FreeImageData(nil)
	FreeImageArray(nil, 3)
	var img SdImage
	FreeImageData(&img) // data 为 nil 的图像也必须安全跳过
}

func TestParamsToStrFreesCBuffer(t *testing.T) {
	// mock 实现返回 Go 内存伪装的 C 字符串，FreeCPtr 在 mock 模式为空操作，
	// 这里至少验证返回值正确且不会重复释放/崩溃
	p := &SdCtxParams{}
	SdCtxParamsInit(p)
	if got := SdCtxParamsToStr(p); got != "mock ctx params" {
		t.Errorf("SdCtxParamsToStr = %q", got)
	}
	sp := &SdSampleParams{}
	if got := SdSampleParamsToStr(sp); got != "mock sample params" {
		t.Errorf("SdSampleParamsToStr = %q", got)
	}
	gp := &SdImgGenParams{}
	if got := SdImgGenParamsToStr(gp); got != "mock img gen params" {
		t.Errorf("SdImgGenParamsToStr = %q", got)
	}
}

type testCallbackCtx struct {
	Value int
}

func TestRegisterCallbackWithContext(t *testing.T) {
	ctx := &testCallbackCtx{Value: 42}
	reg := RegisterProgressCallback(func(step, steps int, tm float32, data unsafe.Pointer) {
		v := CallbackContext(data)
		if v != nil {
			_ = v.(*testCallbackCtx).Value
		}
	}, ctx)
	if reg == nil {
		t.Fatal("expected non-nil registration")
	}
	// 不应 panic；mock 下不会真正回调
	reg.Release()
	reg.Release() // 重复释放必须安全

	// nil 上下文也应安全
	reg2 := RegisterLogCallback(func(level SdLogLevel, text *byte, data unsafe.Pointer) {}, nil)
	if CallbackContext(nil) != nil {
		t.Error("CallbackContext(nil) must be nil")
	}
	reg2.Release()
}

func TestUnsetCallbacks(t *testing.T) {
	SetLogCallback(func(level SdLogLevel, text *byte, data unsafe.Pointer) {}, nil)
	UnsetLogCallback()
	SetProgressCallback(func(step, steps int, tm float32, data unsafe.Pointer) {}, nil)
	UnsetProgressCallback()
	SetPreviewCallback(func(step, n int, f *SdImage, noisy bool, data unsafe.Pointer) {},
		PREVIEW_NONE, 0, false, false, nil)
	UnsetPreviewCallback()
}
