use ffmpeg_next::format::Pixel as AvPixel;

pub type RawFrame = Vec<u8>;
pub type RawVideo = Vec<Vec<u8>>;

/// Always use NV12 pixel format with hardware acceleration, then rescale later.
pub(crate) static HWACCEL_PIXEL_FORMAT: AvPixel = AvPixel::NV12;
