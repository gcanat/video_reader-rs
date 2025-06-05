use ffmpeg_next::format::Pixel as AvPixel;
use ndarray::{Array3, Array4, ArrayViewMut3};

pub type FrameArray = Array3<u8>;
pub type VideoArray = Array4<u8>;

/// Always use NV12 pixel format with hardware acceleration, then rescale later.
pub(crate) static HWACCEL_PIXEL_FORMAT: AvPixel = AvPixel::NV12;

pub fn insert_frame(frame_array: &mut ArrayViewMut3<u8>, frame: FrameArray) {
    frame_array.zip_mut_with(&frame, |a, b| {
        *a = *b;
    });
}
