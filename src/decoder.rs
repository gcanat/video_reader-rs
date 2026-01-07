use crate::convert::{convert_nv12_to_ndarray_rgb24, convert_yuv_to_ndarray_rgb24};
use crate::convert::{get_colorrange, get_colorspace};
use crate::hwaccel::HardwareAccelerationDeviceType;
use crate::utils::{FrameArray, VideoArray};
use ffmpeg::filter;
use ffmpeg::util::frame::video::Video;
use ffmpeg_next as ffmpeg;
use ndarray::{s, Array, Array4, ArrayViewMut3};
use std::collections::HashMap;
use yuv::{YuvRange, YuvStandardMatrix};

/// How to handle out-of-bounds or failed frame fetches in get_batch
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OutOfBoundsMode {
    /// Raise an error when frame fetch fails (default, current behavior)
    #[default]
    Error,
    /// Skip failed frames - returned array may have fewer frames
    Skip,
    /// Return black (all-zero) frames for failed fetches
    Black,
}

/// Struct used when we want to decode the whole video with a compression_factor
#[derive(Clone)]
pub struct VideoReducer {
    indices: Vec<usize>,
    frame_index: usize,
    idx_counter: usize,
    full_video: VideoArray,
}

impl VideoReducer {
    pub fn new(
        indices: Vec<usize>,
        frame_index: usize,
        idx_counter: usize,
        full_video: VideoArray,
    ) -> Self {
        VideoReducer {
            indices,
            frame_index,
            idx_counter,
            full_video,
        }
    }
    pub fn get_frame_index(&self) -> usize {
        self.frame_index
    }
    pub fn incr_frame_index(&mut self, v: usize) {
        self.frame_index += v;
    }
    pub fn set_frame_index(&mut self, v: usize) {
        self.frame_index = v;
    }
    pub fn get_idx_counter(&self) -> usize {
        self.idx_counter
    }
    pub fn incr_idx_counter(&mut self, v: usize) {
        self.idx_counter += v;
    }
    pub fn get_indices(&self) -> &Vec<usize> {
        &self.indices
    }
    pub fn no_indices(&self) -> bool {
        self.indices.is_empty()
    }
    pub fn remove_idx(&mut self, idx: usize) {
        self.indices.remove(idx);
    }
    pub fn slice_mut(&mut self, idx: usize) -> ArrayViewMut3<'_, u8> {
        self.full_video.slice_mut(s![idx, .., .., ..])
    }
    pub fn get_full_video(self) -> Array4<u8> {
        self.full_video
    }
    pub fn build(
        start_frame: Option<usize>,
        end_frame: Option<usize>,
        frame_count: usize,
        compression_factor: Option<f64>,
        height: u32,
        width: u32,
    ) -> (Option<VideoReducer>, Option<usize>, Option<usize>) {
        let start = start_frame.unwrap_or(0);
        let end = end_frame.unwrap_or(frame_count).min(frame_count);

        let mut comp = compression_factor.unwrap_or(1.0);
        if !comp.is_finite() || comp <= 0.0 {
            comp = 1.0;
        }
        // Avoid creating more frames than exist and prevent huge allocations.
        comp = comp.min(1.0);

        let n_frames = ((end - start) as f64 * comp).round().max(1.0) as usize;

        let indices = Array::linspace(start as f64, end as f64 - 1., n_frames)
            .iter()
            .map(|x| x.round() as usize)
            .collect::<Vec<_>>();

        let full_video = Array::zeros((indices.len(), height as usize, width as usize, 3));

        (
            Some(VideoReducer::new(indices, 0, 0, full_video)),
            Some(start),
            Some(end),
        )
    }
}


/// Scaling algorithm for resize operations
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum ResizeAlgo {
    #[default]
    FastBilinear,
    Bilinear,
    Bicubic,
    Nearest,
    Area,
    Lanczos,
}

impl ResizeAlgo {
    /// Convert ResizeAlgo to FFmpeg scale filter flag name
    pub fn as_ffmpeg_flag(&self) -> &'static str {
        match self {
            ResizeAlgo::FastBilinear => "fast_bilinear",
            ResizeAlgo::Bilinear => "bilinear",
            ResizeAlgo::Bicubic => "bicubic",
            ResizeAlgo::Nearest => "neighbor",
            ResizeAlgo::Area => "area",
            ResizeAlgo::Lanczos => "lanczos",
        }
    }
}

/// Config to instantiate a Decoder
/// * threads: number of threads to use
/// * resize_shorter_side: resize shorter side of the video to this value
///   (preserves aspect ratio if resize_longer_side is None)
/// * resize_longer_side: resize longer side of the video to this value
///   (preserves aspect ratio if resize_shorter_side is None)
/// * target_width: optional target width
/// * target_height: optional target height
/// * resize_algo: algorithm to use when resizing
/// * hw_accel: hardware acceleration device type, eg cuda, qsv, etc
/// * ff_filter: optional custom ffmpeg filter to use, eg:
///   "format=rgb24,scale=w=256:h=256:flags=fast_bilinear"
#[derive(Default)]
pub struct DecoderConfig {
    threads: usize,
    resize_shorter_side: Option<f64>,
    resize_longer_side: Option<f64>,
    target_width: Option<u32>,
    target_height: Option<u32>,
    resize_algo: ResizeAlgo,
    hw_accel: Option<HardwareAccelerationDeviceType>,
    ff_filter: Option<String>,
}

impl DecoderConfig {
    pub fn new(
        threads: usize,
        resize_shorter_side: Option<f64>,
        resize_longer_side: Option<f64>,
        target_width: Option<u32>,
        target_height: Option<u32>,
        resize_algo: ResizeAlgo,
        hw_accel: Option<HardwareAccelerationDeviceType>,
        ff_filter: Option<String>,
    ) -> Self {
        Self {
            threads,
            resize_shorter_side,
            resize_longer_side,
            target_width,
            target_height,
            resize_algo,
            hw_accel,
            ff_filter,
        }
    }
    pub fn threads(&self) -> usize {
        self.threads
    }
    pub fn hwaccel(&self) -> Option<HardwareAccelerationDeviceType> {
        self.hw_accel
    }
    pub fn resize_shorter_side(&self) -> Option<f64> {
        self.resize_shorter_side
    }
    pub fn resize_longer_side(&self) -> Option<f64> {
        self.resize_longer_side
    }
    pub fn target_width(&self) -> Option<u32> {
        self.target_width
    }
    pub fn target_height(&self) -> Option<u32> {
        self.target_height
    }
    pub fn resize_algo(&self) -> ResizeAlgo {
        self.resize_algo
    }
    pub fn ff_filter_ref(&self) -> Option<&str> {
        self.ff_filter.as_deref()
    }
    pub fn ff_filter(self) -> Option<String> {
        self.ff_filter
    }
}

/// Struct responsible for doing the actual decoding
pub struct VideoDecoder {
    pub video: ffmpeg::decoder::Video,
    pub height: u32,
    pub width: u32,
    pub fps: f64,
    pub video_info: HashMap<&'static str, String>,
    pub is_hwaccel: bool,
    pub graph: filter::Graph,
    pub color_space: YuvStandardMatrix,
    pub color_range: YuvRange,
}

impl VideoDecoder {
    pub fn new(
        video: ffmpeg::decoder::Video,
        height: u32,
        width: u32,
        fps: f64,
        video_info: HashMap<&'static str, String>,
        is_hwaccel: bool,
        graph: filter::Graph,
    ) -> Self {
        let cspace_string = video_info
            .get("color_space")
            .map(|s| s.as_str())
            .unwrap_or("BT709");
        let crange_string = video_info
            .get("color_range")
            .map(|s| s.as_str())
            .unwrap_or("");
        let color_space = get_colorspace(height as i32, cspace_string);
        let color_range = get_colorrange(crange_string);
        VideoDecoder {
            video,
            height,
            width,
            fps,
            video_info,
            is_hwaccel,
            graph,
            color_space,
            color_range,
        }
    }
    pub fn video_info(&self) -> &HashMap<&'static str, String> {
        &self.video_info
    }
    pub fn fps(&self) -> f64 {
        self.fps
    }
    #[allow(dead_code)]
    pub fn video(&self) -> &ffmpeg::decoder::Video {
        &self.video
    }

    pub fn decode_frames(&mut self) -> Result<Option<FrameArray>, ffmpeg::Error> {
        let mut decoded = Video::empty();
        if self.video.receive_frame(&mut decoded).is_ok() {
            let rgb_frame = self.process_frame(&decoded)?;
            return Ok(rgb_frame);
        }
        Ok(None)
    }

    pub fn process_frame(&mut self, decoded: &Video) -> Result<Option<FrameArray>, ffmpeg::Error> {
        if let Some(mut in_ctx) = self.graph.get("in") {
            if in_ctx.source().add(decoded).is_err() {
                return Err(ffmpeg::Error::Bug);
            }
        } else {
            return Err(ffmpeg::Error::Bug);
        }
        let cspace = self.color_space;
        let crange = self.color_range;

        let mut yuv_frame = Video::empty();
        if let Some(mut out_ctx) = self.graph.get("out") {
            if out_ctx.sink().frame(&mut yuv_frame).is_ok() {
                let rgb_frame: FrameArray = if self.is_hwaccel {
                    convert_nv12_to_ndarray_rgb24(yuv_frame, cspace, crange)?
                } else {
                    convert_yuv_to_ndarray_rgb24(yuv_frame, cspace, crange)?
                };
                return Ok(Some(rgb_frame));
            }
        } else {
            return Err(ffmpeg::Error::Bug);
        }
        Ok(None)
    }

    /// Decode all frames that match the frame indices
    pub fn receive_and_process_decoded_frames(
        &mut self,
        reducer: &mut VideoReducer,
    ) -> Result<Option<FrameArray>, ffmpeg::Error> {
        let mut decoded = Video::empty();
        while self.video.receive_frame(&mut decoded).is_ok() {
            let match_index = reducer
                .get_indices()
                .iter()
                .position(|x| x == &reducer.get_frame_index());
            reducer.incr_frame_index(1);
            if let Some(match_idx) = match_index {
                reducer.remove_idx(match_idx);
                let rgb_frame = self.process_frame(&decoded)?;
                return Ok(rgb_frame);
            }
        }
        Ok(None)
    }
}
