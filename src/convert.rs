use crate::utils::RawFrame;
use ffmpeg::ffi::{av_image_copy_to_buffer, AVPixelFormat};
use ffmpeg::util::frame::video::Video;
use ffmpeg_next as ffmpeg;
use tch::{Device, Kind, Tensor};
use yuv::{
    yuv420_to_rgb, yuv_nv12_to_rgb, YuvBiPlanarImage, YuvConversionMode, YuvPlanarImage, YuvRange,
    YuvStandardMatrix,
};

#[inline(always)]
pub fn frame_tensor_from_raw_vec(raw_vec: &[u8], h: i64, w: i64) -> Tensor {
    unsafe { Tensor::from_blob(raw_vec.as_ptr(), &[h, w, 3], &[], Kind::Uint8, Device::Cpu) }
}

pub fn video_tensor_from_raw_vec(raw_batch: &[RawFrame], h: i64, w: i64) -> Tensor {
    let tensor_vec: Vec<_> = raw_batch
        .iter()
        .map(|raw_vec| frame_tensor_from_raw_vec(raw_vec, h, w))
        .collect();
    Tensor::stack(&tensor_vec, 0_i64)
}

/// Converts a YUV420P video `AVFrame` produced by ffmpeg to a torch tensor.
/// * `frame` - Video frame to convert.
/// * `color_space` - Color space matrix for yuv to rgb conversion, eg BT601, BT709, etc.
/// * `color_range` - color range of the frame: Full or Limited.
/// * returns a raw rgb frame.
pub fn convert_yuv_to_rgb(
    frame: Video,
    color_space: YuvStandardMatrix,
    color_range: YuvRange,
) -> RawFrame {
    let (buf_vec, frame_width, frame_height, bytes_copied) =
        copy_image(frame, AVPixelFormat::AV_PIX_FMT_YUV420P);

    let mut rgb = vec![0_u8; (frame_width * frame_height * 3) as usize];
    if bytes_copied == buf_vec.len() as i32 {
        // let mut rgb = vec![0_u8; (frame_width * frame_height * 3) as usize];
        let cut_point1 = (frame_width * frame_height) as usize;
        let cut_point2 = cut_point1 + cut_point1 / 4;
        let yuv_planar = YuvPlanarImage {
            y_plane: &buf_vec[..cut_point1],
            y_stride: frame_width as u32,
            u_plane: &buf_vec[cut_point1..cut_point2],
            u_stride: (frame_width / 2) as u32,
            v_plane: &buf_vec[cut_point2..],
            v_stride: (frame_width / 2) as u32,
            width: frame_width as u32,
            height: frame_height as u32,
        };
        yuv420_to_rgb(
            &yuv_planar,
            &mut rgb,
            (frame_width * 3) as u32,
            color_range,
            color_space,
        )
        .unwrap();
    }
    rgb
}

/// Converts a NV12 video `AVFrame` produced by ffmpeg to an torch tensor.
/// * `frame` - Video frame to convert.
/// * `color_space` - color space of the frame, eg BT601, BT709, etc.
/// * `color_range` - color range of the frame: Full or Limited.
/// * returns a raw rgb frame.
pub fn convert_nv12_to_rgb(
    frame: Video,
    color_space: YuvStandardMatrix,
    color_range: YuvRange,
) -> RawFrame {
    let (buf_vec, frame_width, frame_height, bytes_copied) =
        copy_image(frame, AVPixelFormat::AV_PIX_FMT_NV12);

    let mut rgb = vec![0_u8; (frame_width * frame_height * 3) as usize];
    if bytes_copied == buf_vec.len() as i32 {
        // let mut rgb = vec![0_u8; (frame_width * frame_height * 3) as usize];
        let cut_point = (frame_width * frame_height) as usize;
        let yuv_planar = YuvBiPlanarImage {
            y_plane: &buf_vec[..cut_point],
            y_stride: frame_width as u32,
            uv_plane: &buf_vec[cut_point..],
            uv_stride: frame_width as u32,
            width: frame_width as u32,
            height: frame_height as u32,
        };
        yuv_nv12_to_rgb(
            &yuv_planar,
            &mut rgb,
            (frame_width * 3) as u32,
            color_range,
            color_space,
            YuvConversionMode::Balanced,
        )
        .unwrap();
    }
    rgb
}

fn copy_image(mut frame: Video, pix_fmt: AVPixelFormat) -> (RawFrame, i32, i32, i32) {
    unsafe {
        let frame_ptr = frame.as_mut_ptr();
        let frame_width: i32 = (*frame_ptr).width;
        let frame_height: i32 = (*frame_ptr).height;
        let frame_format =
            std::mem::transmute::<std::ffi::c_int, AVPixelFormat>((*frame_ptr).format);
        assert_eq!(frame_format, pix_fmt);

        let mut buf_vec = vec![0_u8; (frame_width * (frame_height + frame_height / 2)) as usize];

        let bytes_copied = av_image_copy_to_buffer(
            buf_vec.as_mut_ptr(),
            buf_vec.len() as i32,
            (*frame_ptr).data.as_ptr() as *const *const u8,
            (*frame_ptr).linesize.as_ptr(),
            frame_format,
            frame_width,
            frame_height,
            1,
        );
        (buf_vec, frame_width, frame_height, bytes_copied)
    }
}

pub fn get_colorspace(height: i32, color_space: &str) -> YuvStandardMatrix {
    match color_space {
        "BT709" => YuvStandardMatrix::Bt709,
        "BT601" => YuvStandardMatrix::Bt601,
        "BT2020" => YuvStandardMatrix::Bt2020,
        "SMPTE240" => YuvStandardMatrix::Smpte240,
        "BT470_6" => YuvStandardMatrix::Bt470_6,
        _ => {
            // By default assume HD color space
            let mut colorspace = YuvStandardMatrix::Bt709;
            if height < 720 {
                // SD color space
                colorspace = YuvStandardMatrix::Bt601;
            } else if height > 1080 {
                // UHD color space
                colorspace = YuvStandardMatrix::Bt2020;
            }
            colorspace
        }
    }
}

pub fn get_colorrange(color_range: &str) -> YuvRange {
    match color_range {
        "FULL" | "PC" | "JPEG" => YuvRange::Full,
        _ => YuvRange::Limited,
    }
}

/// Convert 4D torch tensor from RGB (N, H, W, C) to grayscale (N, H, W)
pub fn rgb2gray_tch(frames: Tensor) -> Tensor {
    // we use same coefs as torchvision, tensorflow and opencv
    let rgb_coef = Tensor::from_slice(&[0.2989, 0.5870, 0.1140]).reshape([1, 1, 1, 3]);
    let gray = frames.multiply(&rgb_coef);
    gray.sum_dim_intlist(vec![3_i64], false, Kind::Uint8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_colorspace() {
        assert_eq!(get_colorspace(480, "BT601"), YuvStandardMatrix::Bt601);
        assert_eq!(get_colorspace(480, ""), YuvStandardMatrix::Bt601);
        assert_eq!(get_colorspace(480, "BT709"), YuvStandardMatrix::Bt709);
        assert_eq!(get_colorspace(720, "BT709"), YuvStandardMatrix::Bt709);
        assert_eq!(get_colorspace(1080, "BT2020"), YuvStandardMatrix::Bt2020);
        assert_eq!(get_colorspace(1080, "blabla"), YuvStandardMatrix::Bt709);
        assert_eq!(get_colorspace(2160, "BT2020"), YuvStandardMatrix::Bt2020);
        assert_eq!(get_colorspace(2160, ""), YuvStandardMatrix::Bt2020);
        assert_eq!(get_colorspace(2160, "$éà'é"), YuvStandardMatrix::Bt2020);
        assert_eq!(get_colorspace(2160, "BT601"), YuvStandardMatrix::Bt601);
    }

    #[test]
    fn test_get_colorrange_full() {
        assert_eq!(get_colorrange("FULL"), YuvRange::Full);
        assert_eq!(get_colorrange("PC"), YuvRange::Full);
        assert_eq!(get_colorrange("JPEG"), YuvRange::Full);
    }

    #[test]
    fn test_get_colorrange_limited() {
        assert_eq!(get_colorrange(""), YuvRange::Limited);
        assert_eq!(get_colorrange("LIMITED"), YuvRange::Limited);
        assert_eq!(get_colorrange("unknown"), YuvRange::Limited);
    }
}
