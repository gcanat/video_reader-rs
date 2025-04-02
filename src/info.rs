use ffmpeg::util::rational::Rational;
use ffmpeg_next as ffmpeg;
use std::collections::{BTreeMap, HashMap};

pub struct VideoParams {
    duration: f64,
    start_time: i64,
    time_base: f64,
    time_base_rational: Rational,
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
    decode_order: HashMap<usize, usize>,
}

impl StreamInfo {
    pub fn new(
        frame_count: usize,
        key_frames: Vec<usize>,
        frame_times: BTreeMap<usize, FrameTime>,
        decode_order: HashMap<usize, usize>,
    ) -> Self {
        StreamInfo {
            frame_count,
            key_frames,
            frame_times,
            decode_order,
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
    pub fn get_dec_idx(&self, idx: &usize) -> Option<&usize> {
        self.decode_order.get(idx)
    }
}

pub fn extract_video_params(input: &ffmpeg::Stream) -> VideoParams {
    VideoParams {
        duration: input.duration() as f64 * f64::from(input.time_base()),
        start_time: input.start_time(),
        time_base: f64::from(input.time_base()),
        time_base_rational: input.time_base(),
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
    info.insert("fps", format!("{}", fps));
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
    info.insert("color_range", format!("{:?}", video.color_range()));
    info.insert("color_primaries", format!("{:?}", video.color_primaries()));
    info.insert(
        "color_xfer_charac",
        format!("{:?}", video.color_transfer_characteristic()),
    );
    info.insert("chroma_location", format!("{:?}", video.chroma_location()));
    info.insert("vid_ref", video.references().to_string());
    info.insert("intra_dc_precision", video.intra_dc_precision().to_string());
    info.insert("has_b_frames", format!("{}", video.has_b_frames()));

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

    // mapping between decoding order and presentation order
    let mut decode_order: HashMap<usize, usize> = HashMap::new();
    for (idx, fr_info) in frame_times.iter().enumerate() {
        decode_order.insert(*fr_info.0, idx);
    }

    // Seek back to the begining of the stream
    ictx.seek(0, ..10)?;
    Ok(StreamInfo::new(didx, key_frames, frame_times, decode_order))
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
    if resize_shorter_side_to.is_some() {
        if shorter_is_height {
            new_height = resize_shorter_side_to.unwrap();
            new_width = (width * new_height / height).round();
        } else {
            new_width = resize_shorter_side_to.unwrap();
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

#[test]
fn test_get_resized_dim() {
    // No resize
    assert_eq!(get_resized_dim(720.0, 1280.0, None, None), (720, 1280));

    // Resize shorter side (height)
    assert_eq!(
        get_resized_dim(720.0, 1280.0, Some(360.0), None),
        (360, 640)
    );

    // Resize shorter side (width)
    assert_eq!(
        get_resized_dim(1280.0, 720.0, Some(360.0), None),
        (640, 360)
    );

    // Resize longer side (width)
    assert_eq!(
        get_resized_dim(720.0, 1280.0, None, Some(640.0)),
        (360, 640)
    );

    // Resize longer side (height)
    assert_eq!(
        get_resized_dim(1280.0, 720.0, None, Some(640.0)),
        (640, 360)
    );

    // Both dimensions specified (no aspect ratio)
    assert_eq!(
        get_resized_dim(720.0, 1280.0, Some(360.0), Some(640.0)),
        (360, 640)
    );
    assert_eq!(
        get_resized_dim(1280.0, 720.0, Some(360.0), Some(640.0)),
        (640, 360)
    );

    // Square image
    assert_eq!(
        get_resized_dim(1000.0, 1000.0, Some(500.0), None),
        (500, 500)
    );
    assert_eq!(
        get_resized_dim(1000.0, 1000.0, None, Some(500.0)),
        (500, 500)
    );

    // Edge cases with small numbers
    assert_eq!(get_resized_dim(10.0, 20.0, Some(5.0), None), (5, 10));
    assert_eq!(get_resized_dim(20.0, 10.0, Some(5.0), None), (10, 5));
}
