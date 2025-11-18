use ffmpeg::codec::packet::side_data::Type as SideDataType;
use ffmpeg::ffi::av_display_rotation_get;

use ffmpeg::util::rational::Rational;
use ffmpeg::Stream;
use ffmpeg_next as ffmpeg;
use std::collections::{BTreeMap, HashMap};

pub struct VideoParams {
    duration: f64,
    start_time: i64,
    time_base: f64,
    time_base_rational: Rational,
    pub rotation: isize,
}

/// Timing info for key frames
#[derive(Debug)]
pub struct FrameTime {
    pts: i64,
    dur: i64,
    dts: i64,
}

impl FrameTime {
    pub fn new(pts: i64, dur: i64, dts: i64) -> Self {
        FrameTime { pts, dur, dts }
    }
    pub fn pts(&self) -> &i64 {
        &self.pts
    }
    pub fn dts(&self) -> &i64 {
        &self.dts
    }
    pub fn dur(&self) -> &i64 {
        &self.dur
    }
}

/// Info gathered from iterating over the video stream
pub struct StreamInfo {
    frame_count: usize,
    key_frames: Vec<usize>,
    frame_times: BTreeMap<usize, FrameTime>,
}

impl StreamInfo {
    pub fn new(
        frame_count: usize,
        key_frames: Vec<usize>,
        frame_times: BTreeMap<usize, FrameTime>,
    ) -> Self {
        StreamInfo {
            frame_count,
            key_frames,
            frame_times,
        }
    }
    pub fn frame_count(&self) -> &usize {
        &self.frame_count
    }
    pub fn key_frames(&self) -> &Vec<usize> {
        &self.key_frames
    }
    pub fn frame_times(&self) -> &BTreeMap<usize, FrameTime> {
        &self.frame_times
    }
    pub fn get_all_pts(&self, time_base: f64) -> Vec<f64> {
        self.frame_times
            .values()
            .map(|v| v.pts as f64 * time_base)
            .collect::<Vec<_>>()
    }
    pub fn get_pts(&self, indices: &[usize], time_base: f64) -> Vec<f64> {
        indices
            .iter()
            .map(|i| match self.frame_times.get(i) {
                None => -1.,
                Some(ft) => ft.pts as f64 * time_base,
            })
            .collect::<Vec<_>>()
    }
}

fn get_rotation_angle(input: &Stream) -> isize {
    for sd in input.side_data() {
        if sd.kind() == SideDataType::DisplayMatrix {
            let matrix = sd.data();
            let theta = unsafe { av_display_rotation_get(matrix.as_ptr() as *const _) };
            if theta.is_nan() {
                return 0_isize;
            } else {
                return theta.round() as isize;
            }
        }
    }
    0_isize
}

pub fn extract_video_params(input: &Stream) -> VideoParams {
    VideoParams {
        duration: input.duration() as f64 * f64::from(input.time_base()),
        start_time: input.start_time(),
        time_base: f64::from(input.time_base()),
        time_base_rational: input.time_base(),
        rotation: get_rotation_angle(input),
    }
}

pub fn collect_video_metadata(
    video: &ffmpeg::decoder::Video,
    params: &VideoParams,
    fps: &f64,
) -> HashMap<&'static str, String> {
    let mut info = HashMap::new();

    let fps_rational = video.frame_rate().unwrap_or(Rational(0, 1));
    info.insert("fps_rational", fps_rational.to_string());
    info.insert("fps", format!("{fps}"));
    info.insert("start_time", params.start_time.to_string());
    info.insert("time_base", params.time_base.to_string());
    info.insert("time_base_rational", params.time_base_rational.to_string());
    info.insert("duration", params.duration.to_string());
    info.insert("codec_id", format!("{:?}", video.codec().unwrap().id()));
    info.insert("height", video.height().to_string());
    info.insert("width", video.width().to_string());
    info.insert("bit_rate", video.bit_rate().to_string());
    info.insert("vid_format", format!("{:?}", video.format()));
    info.insert("aspect_ratio", format!("{:?}", video.aspect_ratio()));
    let color_space = format!("{:?}", video.color_space());
    info.insert("color_space", color_space.to_uppercase());
    let color_range = format!("{:?}", video.color_range());
    info.insert("color_range", color_range.to_uppercase());
    info.insert("color_primaries", format!("{:?}", video.color_primaries()));
    info.insert(
        "color_xfer_charac",
        format!("{:?}", video.color_transfer_characteristic()),
    );
    info.insert("chroma_location", format!("{:?}", video.chroma_location()));
    info.insert("vid_ref", video.references().to_string());
    info.insert("intra_dc_precision", video.intra_dc_precision().to_string());
    info.insert("has_b_frames", format!("{}", video.has_b_frames()));
    info.insert("rotation", format!("{}", params.rotation));
    info
}

pub fn get_frame_count(
    ictx: &mut ffmpeg::format::context::Input,
    stream_index: &usize,
) -> Result<StreamInfo, ffmpeg::Error> {
    // keep track of decoding index, ie frame count
    let mut didx = 0_usize;
    let mut key_frames = Vec::new();
    let mut frame_times = BTreeMap::new();

    for (stream, packet) in ictx.packets() {
        if &stream.index() == stream_index {
            let pts = packet.pts().unwrap_or(0);
            let dur = packet.duration();
            let dts = packet.dts().unwrap_or(0);
            frame_times.insert(didx, FrameTime::new(pts, dur, dts));
            if packet.is_key() {
                key_frames.push(didx);
            }
            didx += 1;
        }
    }

    // Seek back to the begining of the stream
    ictx.seek(0, ..10)?;
    Ok(StreamInfo::new(didx, key_frames, frame_times))
}

/// Get the resized dimension of a frame, keep the aspect ratio.
/// Resize the shorter side of the frame, and the other side accordingly
/// so that the resizing operation is minimal. Returns a resizing dimension only
/// if the shorter side of the frame is bigger than resize_shorter_side_to.
/// * height (f64): Height of the frame
/// * width (f64): Width of the frame
/// * resize_shorter_side_to (Option<f64>): Resize the shorter side of the frame to this value.
/// * resize_longer_side_to (Option<f64>): Resize the longer side of the frame to this value.
///
/// Returns: Option<(u32, u32)>: Option of the resized height and width
pub fn get_resized_dim(
    height: f64,
    width: f64,
    resize_shorter_side_to: Option<f64>,
    resize_longer_side_to: Option<f64>,
) -> (u32, u32) {
    let mut respect_aspect_ratio = true;
    // early return when nothing to do
    if resize_shorter_side_to.is_none() && resize_longer_side_to.is_none() {
        return (height as u32, width as u32);
    } else if resize_shorter_side_to.is_some() && resize_longer_side_to.is_some() {
        respect_aspect_ratio = false;
    }

    let mut shorter_is_height: bool = true;
    let new_height: f64;
    let new_width: f64;

    if width < height {
        shorter_is_height = false;
    }

    if !respect_aspect_ratio {
        // in this case both values are Some so it should be safe to unwrap
        if shorter_is_height {
            new_height = resize_shorter_side_to.unwrap();
            new_width = resize_longer_side_to.unwrap();
        } else {
            new_height = resize_longer_side_to.unwrap();
            new_width = resize_shorter_side_to.unwrap();
        }
        return (new_height as u32, new_width as u32);
    }

    // Only case remaining is one is None and the other is Some
    if let Some(res_short_side) = resize_shorter_side_to {
        if shorter_is_height {
            new_height = res_short_side;
            new_width = (width * new_height / height).round();
        } else {
            new_width = res_short_side;
            new_height = (height * new_width / width).round();
        }
    } else {
        // resize_longer_side_to can only be Some at this point
        if shorter_is_height {
            new_width = resize_longer_side_to.unwrap();
            new_height = (height * new_width / width).round();
        } else {
            new_height = resize_longer_side_to.unwrap();
            new_width = (width * new_height / height).round();
        }
    }
    (new_height as u32, new_width as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_resize_needed() {
        let height = 720.0;
        let width = 1280.0;
        let result = get_resized_dim(height, width, None, None);

        assert_eq!(result, (720, 1280));
    }

    #[test]
    fn test_resize_shorter_side_height() {
        let height = 720.0;
        let width = 1280.0;
        let resize_to = 480.0;

        let result = get_resized_dim(height, width, Some(resize_to), None);

        assert_eq!(result.0, 480);
        assert_eq!(result.1, 853);
    }

    #[test]
    fn test_resize_shorter_side_width() {
        let height = 1080.0;
        let width = 720.0;
        let resize_to = 480.0;

        let result = get_resized_dim(height, width, Some(resize_to), None);

        assert_eq!(result.1, 480);
        assert_eq!(result.0, 720);
    }

    #[test]
    fn test_resize_longer_side_height() {
        let height = 1080.0;
        let width = 720.0;
        let resize_to = 720.0;

        let result = get_resized_dim(height, width, None, Some(resize_to));

        assert_eq!(result.0, 720);
        assert_eq!(result.1, 480);
    }

    #[test]
    fn test_resize_longer_side_width() {
        let height = 720.0;
        let width = 1280.0;
        let resize_to = 854.0;

        let result = get_resized_dim(height, width, None, Some(resize_to));

        assert_eq!(result.1, 854);
        assert_eq!(result.0, 480);
    }

    #[test]
    fn test_both_dimensions_specified() {
        let height = 720.0;
        let width = 1280.0;

        let result = get_resized_dim(height, width, Some(480.0), Some(800.0));

        // Should ignore aspect ratio and use the provided dimensions
        assert_eq!(result, (480, 800));
    }

    #[test]
    fn test_both_dimensions_specified_width_smaller() {
        let height = 1080.0;
        let width = 720.0;

        let result = get_resized_dim(height, width, Some(480.0), Some(800.0));

        // Should set width to shorter_side value and height to longer_side value
        assert_eq!(result, (800, 480));
    }

    #[test]
    fn test_resize_equal_dimensions() {
        let height = 1000.0;
        let width = 1000.0;

        let result = get_resized_dim(height, width, Some(500.0), None);

        assert_eq!(result, (500, 500));
    }

    #[test]
    fn test_rounding_behavior() {
        let height = 723.0;
        let width = 1283.0;
        let resize_to = 483.0;

        let result = get_resized_dim(height, width, Some(resize_to), None);

        assert_eq!(result.0, 483);
        // 1283 * (483/723) = 857.1 which should round to 857
        assert_eq!(result.1, 857);
    }
}
