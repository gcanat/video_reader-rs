use crate::convert::{convert_nv12_to_rgb, convert_yuv_to_rgb};
use crate::convert::{get_colorrange, get_colorspace};
use crate::hwaccel::HardwareAccelerationDeviceType;
use crate::utils::RawFrame;
use ffmpeg::filter;
use ffmpeg::util::frame::video::Video;
use ffmpeg_next as ffmpeg;
use std::collections::HashMap;
use tch::{Device, Kind, Tensor};
use yuv::{YuvRange, YuvStandardMatrix};

/// Struct used when we want to decode the whole video with a compression_factor
#[derive(Clone)]
pub struct VideoReducer {
    indices: Vec<usize>,
    frame_index: usize,
    idx_counter: usize,
}

impl VideoReducer {
    pub fn new(indices: Vec<usize>, frame_index: usize, idx_counter: usize) -> Self {
        VideoReducer {
            indices,
            frame_index,
            idx_counter,
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
    pub fn build(
        start_frame: Option<usize>,
        end_frame: Option<usize>,
        frame_count: usize,
        compression_factor: Option<f64>,
    ) -> (Option<VideoReducer>, Option<usize>, Option<usize>) {
        let start = start_frame.unwrap_or(0);
        let end = end_frame.unwrap_or(frame_count).min(frame_count);

        let n_frames = ((end - start) as f64 * compression_factor.unwrap_or(1.0)).round() as i64;

        // create the indices with torch linspace
        let tch_indices: Tensor = Tensor::linspace(
            start as f64,
            end as f64 - 1.,
            n_frames,
            (Kind::Float, Device::Cpu),
        )
        .round();
        // copy tensor data to a vec
        let mut indice_vec: Vec<f64> = Vec::with_capacity(n_frames as usize);
        tch_indices.copy_data(&mut indice_vec, n_frames as usize);
        // convert vec values to usize
        let indices: Vec<usize> = indice_vec.into_iter().map(|v| v as usize).collect();

        (
            Some(VideoReducer::new(indices, 0, 0)),
            Some(start),
            Some(end),
        )
    }
}

/// Config to instantiate a Decoder
/// * threads: number of threads to use
/// * resize_shorter_side: resize shorter side of the video to this value
///   (preserves aspect ratio if resize_longer_side is None)
/// * resize_longer_side: resize longer side of the video to this value
///   (preserves aspect ratio if resize_shorter_side is None)
/// * hw_accel: hardware acceleration device type, eg cuda, qsv, etc
/// * ff_filter: optional custom ffmpeg filter to use, eg:
///   "format=rgb24,scale=w=256:h=256:flags=fast_bilinear"
#[derive(Default)]
pub struct DecoderConfig {
    threads: usize,
    resize_shorter_side: Option<f64>,
    resize_longer_side: Option<f64>,
    hw_accel: Option<HardwareAccelerationDeviceType>,
    ff_filter: Option<String>,
}

impl DecoderConfig {
    pub fn new(
        threads: usize,
        resize_shorter_side: Option<f64>,
        resize_longer_side: Option<f64>,
        hw_accel: Option<HardwareAccelerationDeviceType>,
        ff_filter: Option<String>,
    ) -> Self {
        Self {
            threads,
            resize_shorter_side,
            resize_longer_side,
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
    pub fn ff_filter(self) -> Option<String> {
        self.ff_filter
    }
}

/// Struct responsible for doing the actual decoding
pub struct VideoDecoder {
    pub video: ffmpeg::decoder::Video,
    pub fps: f64,
    pub video_info: HashMap<&'static str, String>,
    pub is_hwaccel: bool,
    pub graph: filter::Graph,
    pub color_space: YuvStandardMatrix,
    pub color_range: YuvRange,
}

unsafe impl Send for VideoDecoder {}

impl VideoDecoder {
    pub fn new(
        video: ffmpeg::decoder::Video,
        height: u32,
        fps: f64,
        video_info: HashMap<&'static str, String>,
        is_hwaccel: bool,
        graph: filter::Graph,
    ) -> Self {
        let cspace_string = video_info.get("color_space").unwrap();
        let crange_string = video_info.get("color_range").unwrap();
        let color_space = get_colorspace(height as i32, cspace_string);
        let color_range = get_colorrange(crange_string);
        VideoDecoder {
            video,
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
    pub fn video(&self) -> &ffmpeg::decoder::Video {
        &self.video
    }

    pub fn decode_frames(&mut self) -> Result<Option<RawFrame>, ffmpeg::Error> {
        let mut decoded = Video::empty();
        if self.video.receive_frame(&mut decoded).is_ok() {
            let rgb_frame = self.process_frame(&decoded);
            return Ok(rgb_frame);
        }
        Ok(None)
    }

    pub fn process_frame(&mut self, decoded: &Video) -> Option<RawFrame> {
        self.graph.get("in").unwrap().source().add(decoded).unwrap();
        let cspace = self.color_space;
        let crange = self.color_range;

        let mut yuv_frame = Video::empty();
        if self
            .graph
            .get("out")
            .unwrap()
            .sink()
            .frame(&mut yuv_frame)
            .is_ok()
        {
            let raw_frame: RawFrame = if self.is_hwaccel {
                convert_nv12_to_rgb(yuv_frame, cspace, crange)
            } else {
                convert_yuv_to_rgb(yuv_frame, cspace, crange)
            };
            return Some(raw_frame);
        }
        None
    }

    /// Decode all frames that match the frame indices
    pub fn receive_and_process_decoded_frames(
        &mut self,
        reducer: &mut VideoReducer,
    ) -> Result<Option<RawFrame>, ffmpeg::Error> {
        let mut decoded = Video::empty();
        while self.video.receive_frame(&mut decoded).is_ok() {
            let match_index = reducer
                .get_indices()
                .iter()
                .position(|x| x == &reducer.get_frame_index());
            reducer.incr_frame_index(1);
            if match_index.is_some() {
                reducer.remove_idx(match_index.unwrap());
                let rgb_frame = self.process_frame(&decoded);
                return Ok(rgb_frame);
            }
        }
        Ok(None)
    }
    /// Decode frames
    pub fn skip_and_decode_frames(
        &mut self,
        reducer: &mut VideoReducer,
        indices: &[usize],
        frame_map: &mut HashMap<usize, RawFrame>,
    ) -> Result<(), ffmpeg::Error> {
        let mut decoded = Video::empty();
        while self.video.receive_frame(&mut decoded).is_ok() {
            if indices.iter().any(|x| x == &reducer.get_frame_index()) {
                if let Some(rgb_frame) = self.process_frame(&decoded) {
                    frame_map.insert(reducer.get_frame_index(), rgb_frame);
                }
            }
            reducer.incr_frame_index(1);
        }
        Ok(())
    }
}
