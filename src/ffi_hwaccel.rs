// from https://github.com/oddity-ai/video-rs/blob/main/src/ffi_hwaccel.rs
extern crate ffmpeg_next as ffmpeg;

use crate::hwaccel::HardwareAccelerationDeviceType;
use ffmpeg::ffi::*;

pub struct HardwareDeviceContext {
    ptr: *mut ffmpeg::ffi::AVBufferRef,
}

impl HardwareDeviceContext {
    pub fn new(
        device_type: HardwareAccelerationDeviceType,
    ) -> Result<HardwareDeviceContext, ffmpeg::error::Error> {
        let mut ptr: *mut ffmpeg::ffi::AVBufferRef = std::ptr::null_mut();

        unsafe {
            match ffmpeg::ffi::av_hwdevice_ctx_create(
                (&mut ptr) as *mut *mut ffmpeg::ffi::AVBufferRef,
                device_type.into(),
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
            ) {
                0 => Ok(HardwareDeviceContext { ptr }),
                e => Err(ffmpeg::error::Error::from(e)),
            }
        }
    }

    pub unsafe fn ref_raw(&self) -> *mut ffmpeg::ffi::AVBufferRef {
        ffmpeg::ffi::av_buffer_ref(self.ptr)
    }
}

impl Drop for HardwareDeviceContext {
    fn drop(&mut self) {
        unsafe {
            ffmpeg::ffi::av_buffer_unref(&mut self.ptr);
        }
    }
}

pub fn hwdevice_list_available_device_types() -> Vec<HardwareAccelerationDeviceType> {
    let mut hwdevice_types = Vec::new();
    let mut hwdevice_type = unsafe {
        ffmpeg::ffi::av_hwdevice_iterate_types(ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE)
    };
    while hwdevice_type != ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE {
        hwdevice_types.push(HardwareAccelerationDeviceType::from(hwdevice_type).unwrap());
        hwdevice_type = unsafe { ffmpeg::ffi::av_hwdevice_iterate_types(hwdevice_type) };
    }
    hwdevice_types
}

pub fn codec_find_corresponding_hwaccel_pixfmt(
    codec: &ffmpeg::codec::codec::Codec,
    hwaccel_type: HardwareAccelerationDeviceType,
) -> Option<ffmpeg::format::pixel::Pixel> {
    let mut i = 0;
    loop {
        unsafe {
            let hw_config = ffmpeg::ffi::avcodec_get_hw_config(codec.as_ptr(), i);
            if !hw_config.is_null() {
                let hw_config_supports_codec = (((*hw_config).methods) as i32
                    & ffmpeg::ffi::AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as i32)
                    != 0;
                if hw_config_supports_codec && (*hw_config).device_type == hwaccel_type.into() {
                    break Some((*hw_config).pix_fmt.into());
                }
            } else {
                break None;
            }
        }
        i += 1;
    }
}

pub fn codec_context_hwaccel_set_get_format(
    codec_context: &mut ffmpeg::codec::context::Context,
    hw_pixfmt: ffmpeg::format::pixel::Pixel,
) {
    unsafe {
        (*codec_context.as_mut_ptr()).opaque =
            ffmpeg::ffi::AVPixelFormat::from(hw_pixfmt) as i32 as _;
        (*codec_context.as_mut_ptr()).get_format = Some(hwaccel_get_format);
    }
}

pub fn codec_context_hwaccel_set_hw_device_ctx(
    codec_context: &mut ffmpeg::codec::context::Context,
    hardware_device_context: &HardwareDeviceContext,
) {
    unsafe {
        (*codec_context.as_mut_ptr()).hw_device_ctx = hardware_device_context.ref_raw();
    }
}

pub fn codec_context_get_hw_frames_ctx(
    codec_context: &mut ffmpeg::codec::decoder::Video,
    hw_pixfmt: ffmpeg::util::format::Pixel,
    sw_pixfmt: ffmpeg::util::format::Pixel,
) -> Result<(), ffmpeg::error::Error> {
    unsafe {
        let hw_frame_ref = av_hwframe_ctx_alloc((*codec_context.as_mut_ptr()).hw_device_ctx);
        (*codec_context.as_mut_ptr()).pix_fmt = hw_pixfmt.into();
        let frame_ctx = (*hw_frame_ref).data as *mut AVHWFramesContext;
        (*frame_ctx).format = hw_pixfmt.into();
        (*frame_ctx).sw_format = sw_pixfmt.into();
        (*frame_ctx).width = (*codec_context.as_mut_ptr()).width;
        (*frame_ctx).height = (*codec_context.as_mut_ptr()).height;
        (*frame_ctx).initial_pool_size = 4;
        let ret = av_hwframe_ctx_init(hw_frame_ref);
        if ret < 0 {
            return Err(ffmpeg::error::Error::from(ret));
        }
        (*codec_context.as_mut_ptr()).hw_frames_ctx = av_buffer_ref(hw_frame_ref);
    }
    Ok(())
}

// #[no_mangle]
unsafe extern "C" fn hwaccel_get_format(
    ctx: *mut ffmpeg::ffi::AVCodecContext,
    pix_fmts: *const ffmpeg::ffi::AVPixelFormat,
) -> ffmpeg::ffi::AVPixelFormat {
    let mut p = pix_fmts;
    while *p != ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_NONE {
        if *p == std::mem::transmute::<i32, ffmpeg::ffi::AVPixelFormat>((*ctx).opaque as i32) {
            return *p;
        }
        p = p.add(1);
    }
    ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_NONE
}
