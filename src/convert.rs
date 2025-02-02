use ffmpeg::ffi::{av_image_copy_to_buffer, AVPixelFormat};
use ffmpeg::util::frame::video::Video;
use ffmpeg_next as ffmpeg;
use ndarray::parallel::prelude::*;
use ndarray::{stack, Array2, Array3, Array4, ArrayView3, ArrayViewMut3, Axis};
use yuvutils_rs::{
    yuv420_to_rgb, yuv_nv12_to_rgb, YuvBiPlanarImage, YuvConversionMode, YuvPlanarImage, YuvRange,
    YuvStandardMatrix,
};

/// Converts an RGB24 video `AVFrame` produced by ffmpeg to an `ndarray`.
/// Copied from https://github.com/oddity-ai/video-rs
/// * `frame` - Video frame to convert.
/// * `frame_array` mutable reference to the ndarray where the data will be copied
pub fn convert_frame_to_ndarray_rgb24(
    frame: &mut Video,
    frame_array: &mut ArrayViewMut3<u8>,
) -> Result<(), ffmpeg::Error> {
    unsafe {
        let frame_ptr = frame.as_mut_ptr();
        let frame_width: i32 = (*frame_ptr).width;
        let frame_height: i32 = (*frame_ptr).height;
        let frame_format =
            std::mem::transmute::<std::ffi::c_int, AVPixelFormat>((*frame_ptr).format);
        assert_eq!(frame_format, AVPixelFormat::AV_PIX_FMT_RGB24);

        let bytes_copied = av_image_copy_to_buffer(
            frame_array.as_mut_ptr(),
            frame_array.len() as i32,
            (*frame_ptr).data.as_ptr() as *const *const u8,
            (*frame_ptr).linesize.as_ptr(),
            frame_format,
            frame_width,
            frame_height,
            1,
        );

        if bytes_copied == frame_array.len() as i32 {
            Ok(())
        } else {
            Err(ffmpeg::Error::InvalidData)
        }
    }
}

/// Converts a YUV420P video `AVFrame` produced by ffmpeg to an `ndarray`.
/// * `frame` - Video frame to convert.
/// * returns a three-dimensional `ndarray` with dimensions `(H, W, C)` and type byte.
pub fn convert_yuv_to_ndarray_rgb24(frame: Video) -> Array3<u8> {
    let (buf_vec, frame_width, frame_height, bytes_copied) =
        copy_image(frame, AVPixelFormat::AV_PIX_FMT_YUV420P);

    let colorspace = get_colorspace(frame_height);

    if bytes_copied == buf_vec.len() as i32 {
        let mut rgb = vec![0_u8; (frame_width * frame_height * 3) as usize];
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
            YuvRange::Full,
            colorspace,
        )
        .unwrap();
        Array3::from_shape_vec((frame_height as usize, frame_width as usize, 3_usize), rgb).unwrap()
    } else {
        Array3::zeros((frame_height as usize, frame_width as usize, 3_usize))
    }
}

/// Converts a NV12 video `AVFrame` produced by ffmpeg to an `ndarray`.
/// * `frame` - Video frame to convert.
/// * returns a three-dimensional `ndarray` with dimensions `(H, W, C)` and type byte.
pub fn convert_nv12_to_ndarray_rgb24(frame: Video) -> Array3<u8> {
    let (buf_vec, frame_width, frame_height, bytes_copied) =
        copy_image(frame, AVPixelFormat::AV_PIX_FMT_NV12);

    let colorspace = get_colorspace(frame_width);

    if bytes_copied == buf_vec.len() as i32 {
        let mut rgb = vec![0_u8; (frame_width * frame_height * 3) as usize];
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
            YuvRange::Full,
            colorspace,
            YuvConversionMode::Balanced,
        )
        .unwrap();
        Array3::from_shape_vec((frame_height as usize, frame_width as usize, 3_usize), rgb).unwrap()
    } else {
        Array3::zeros((frame_height as usize, frame_width as usize, 3_usize))
    }
}

fn copy_image(mut frame: Video, pix_fmt: AVPixelFormat) -> (Vec<u8>, i32, i32, i32) {
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

fn get_colorspace(height: i32) -> YuvStandardMatrix {
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

/// Convert RGB video (N, H, W, C) to Grayscale video (N, H, W).
/// Returns a 3D ndarray with shape (N, H, W).
pub fn rgb2gray(frames: Array4<u8>) -> Array3<u8> {
    let mut gray = Vec::new();
    frames
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(rgb2gray_2d)
        .collect_into_vec(&mut gray);
    let views: Vec<_> = gray.iter().map(|x| x.view()).collect();
    stack(Axis(0), &views[..]).unwrap()
}

/// Convert RGB Frame (H, W, C) to grayscale (H, W).
fn rgb2gray_2d(frames: ArrayView3<u8>) -> Array2<u8> {
    frames.map_axis(Axis(2), |pix| {
        (0.2125 * pix[0] as f32 + 0.7154 * pix[1] as f32 + 0.0721 * pix[2] as f32)
            .round()
            .clamp(0.0, 255.0) as u8
    })
}
