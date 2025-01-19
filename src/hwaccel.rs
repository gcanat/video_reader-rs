// based on: https://github.com/oddity-ai/video-rs/blob/main/src/hwaccel.rs
use ffmpeg_next as ffmpeg;
use log::debug;
use std::str::FromStr;

use crate::ffi_hwaccel;

pub(crate) struct HardwareAccelerationContext {
    pixel_format: ffmpeg::util::format::Pixel,
    _hardware_device_context: ffi_hwaccel::HardwareDeviceContext,
}

impl HardwareAccelerationContext {
    pub(crate) fn new(
        decoder: &mut ffmpeg::codec::Context,
        device_type: HardwareAccelerationDeviceType,
    ) -> Result<Self, ffmpeg::Error> {
        let codec = ffmpeg::codec::decoder::find(decoder.id()).unwrap();
        let pixel_format =
            match ffi_hwaccel::codec_find_corresponding_hwaccel_pixfmt(&codec, device_type) {
                Some(pix_fmt) => pix_fmt,
                None => ffmpeg::util::format::Pixel::None,
            };
        if pixel_format == ffmpeg::util::format::Pixel::None {
            println!("Could not find hwaccel pix_fmt for this device type");
            return Err(ffmpeg::Error::DecoderNotFound);
        }
        debug!("HardwareAccel Pixel format: {:?}", pixel_format);

        ffi_hwaccel::codec_context_hwaccel_set_get_format(decoder, pixel_format);

        let hardware_device_context = ffi_hwaccel::HardwareDeviceContext::new(device_type)?;
        ffi_hwaccel::codec_context_hwaccel_set_hw_device_ctx(decoder, &hardware_device_context);

        Ok(HardwareAccelerationContext {
            pixel_format,
            _hardware_device_context: hardware_device_context,
        })
    }
    pub(crate) fn format(&self) -> ffmpeg::util::format::Pixel {
        self.pixel_format
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum HardwareAccelerationDeviceType {
    /// Video Decode and Presentation API for Unix (VDPAU)
    Vdpau,
    /// NVIDIA CUDA
    Cuda,
    /// Video Acceleration API (VA-API)
    VaApi,
    /// DirectX Video Acceleration 2.0
    Dxva2,
    /// Quick Sync Video
    Qsv,
    /// VideoToolbox
    VideoToolbox,
    /// Direct3D 11 Video Acceleration
    D3D11Va,
    /// Linux Direct Rendering Manager
    Drm,
    /// OpenCL
    OpenCl,
    /// MediaCodec
    MediaCodec,
    /// Direct3D 12 Video Acceleration
    D3D12Va,
}

impl HardwareAccelerationDeviceType {
    /// Whether or not the device type is available on this system.
    pub fn is_available(self) -> bool {
        Self::list_available().contains(&self)
    }

    /// List available hardware acceleration device types on this system.
    ///
    /// Uses `av_hwdevice_iterate_types` internally.
    pub fn list_available() -> Vec<HardwareAccelerationDeviceType> {
        ffi_hwaccel::hwdevice_list_available_device_types()
    }
}

impl FromStr for HardwareAccelerationDeviceType {
    type Err = ();
    fn from_str(val: &str) -> Result<Self, Self::Err> {
        match val.to_lowercase().as_str() {
            "vdpau" => Ok(Self::Vdpau),
            "cuda" => Ok(Self::Cuda),
            "vaapi" => Ok(Self::VaApi),
            "dxva2" => Ok(Self::Dxva2),
            "qsv" => Ok(Self::Qsv),
            "videotoolbox" => Ok(Self::VideoToolbox),
            "d3d11va" => Ok(Self::D3D11Va),
            "drm" => Ok(Self::Drm),
            "opencl" => Ok(Self::OpenCl),
            "mediacodec" => Ok(Self::MediaCodec),
            "d3d12va" => Ok(Self::D3D12Va),
            _ => Err(()),
        }
    }
}

impl HardwareAccelerationDeviceType {
    pub fn from(value: ffmpeg::ffi::AVHWDeviceType) -> Option<HardwareAccelerationDeviceType> {
        match value {
            ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VDPAU => Some(Self::Vdpau),
            ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA => Some(Self::Cuda),
            ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI => Some(Self::VaApi),
            ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_DXVA2 => Some(Self::Dxva2),
            ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_QSV => Some(Self::Qsv),
            ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX => Some(Self::VideoToolbox),
            ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA => Some(Self::D3D11Va),
            ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_DRM => Some(Self::Drm),
            ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_OPENCL => Some(Self::OpenCl),
            ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_MEDIACODEC => Some(Self::MediaCodec),
            ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE => None,
            // FIXME: Find a way to handle the new variants in ffmpeg 7 without breaking backwards
            // compatibility...
            #[allow(unreachable_patterns)]
            _ => unimplemented!(),
        }
    }
}

impl From<HardwareAccelerationDeviceType> for ffmpeg::ffi::AVHWDeviceType {
    fn from(value: HardwareAccelerationDeviceType) -> Self {
        match value {
            HardwareAccelerationDeviceType::Vdpau => {
                ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VDPAU
            }
            HardwareAccelerationDeviceType::Cuda => {
                ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA
            }
            HardwareAccelerationDeviceType::VaApi => {
                ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI
            }
            HardwareAccelerationDeviceType::Dxva2 => {
                ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_DXVA2
            }
            HardwareAccelerationDeviceType::Qsv => {
                ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_QSV
            }
            HardwareAccelerationDeviceType::VideoToolbox => {
                ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX
            }
            HardwareAccelerationDeviceType::D3D11Va => {
                ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA
            }
            HardwareAccelerationDeviceType::Drm => {
                ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_DRM
            }
            HardwareAccelerationDeviceType::OpenCl => {
                ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_OPENCL
            }
            HardwareAccelerationDeviceType::MediaCodec => {
                ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_MEDIACODEC
            }
            HardwareAccelerationDeviceType::D3D12Va => {
                unimplemented!()
            }
        }
    }
}
