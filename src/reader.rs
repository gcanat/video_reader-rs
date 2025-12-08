use ffmpeg::codec::threading;
use ffmpeg::ffi::*;
use ffmpeg::format::input;
use ffmpeg::media::Type;
use ffmpeg::util::frame::video::Video;
use ffmpeg::util::rational::Rational;
use ffmpeg_next as ffmpeg;
use log::debug;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use crate::convert::{convert_nv12_to_ndarray_rgb24, convert_yuv_to_ndarray_rgb24};
use crate::decoder::{DecoderConfig, VideoDecoder, VideoReducer};
use crate::filter::{create_filter_spec, create_filters, FilterConfig};
use crate::hwaccel::{HardwareAccelerationContext, HardwareAccelerationDeviceType};
use crate::info::{
    collect_video_metadata, extract_video_params, get_frame_count, get_resized_dim, StreamInfo,
};
use crate::utils::{insert_frame, FrameArray, VideoArray, HWACCEL_PIXEL_FORMAT};
use ndarray::{s, Array, Array4};
use tokio::task;

pub fn get_init_context(
    filename: &String,
) -> Result<(ffmpeg::format::context::Input, usize), ffmpeg::Error> {
    let input_file = Path::new(filename);

    // Initialize the FFmpeg library
    ffmpeg::init()?;

    // Open the input file
    let ictx = input(&input_file)?;
    let stream_index = ictx
        .streams()
        .best(Type::Video)
        .ok_or(ffmpeg::Error::StreamNotFound)?
        .index();
    Ok((ictx, stream_index))
}

fn setup_decoder_context(
    input: &ffmpeg::Stream,
    threads: usize,
    hwaccel_device_type: Option<HardwareAccelerationDeviceType>,
) -> Result<
    (
        ffmpeg::codec::context::Context,
        Option<HardwareAccelerationContext>,
    ),
    ffmpeg::Error,
> {
    let mut context = ffmpeg::codec::context::Context::from_parameters(input.parameters())?;

    let hwaccel_context = match hwaccel_device_type {
        Some(device_type) => Some(HardwareAccelerationContext::new(&mut context, device_type)?),
        None => None,
    };
    context.set_threading(threading::Config {
        kind: threading::Type::Frame,
        count: threads,
        #[cfg(feature = "ffmpeg_5")]
        safe: true,
    });
    Ok((context, hwaccel_context))
}

/// Struct responsible for reading the stream and getting the metadata
pub struct VideoReader {
    ictx: ffmpeg::format::context::Input,
    stream_index: usize,
    stream_info: StreamInfo,
    curr_frame: usize,
    curr_dec_idx: usize,
    /// Current presentation index (display order count)
    curr_pres_idx: usize,
    n_fails: usize,
    decoder: VideoDecoder,
    draining: bool,
    /// Cached result of seek verification (None = not tested yet)
    seek_verified: Option<bool>,
    /// True if we've sent EOF and need to re-seek before processing more frames
    eof_sent: bool,
}

impl VideoReader {
    pub fn decoder(&self) -> &VideoDecoder {
        &self.decoder
    }
    pub fn stream_info(&self) -> &StreamInfo {
        &self.stream_info
    }
    /// Create a new VideoReader instance
    /// * `filename` - Path to the video file.
    /// * `decoder_config` - Config for the decoder see: [`DecoderConfig`]
    ///
    /// Returns: a VideoReader instance.
    pub fn new(
        filename: String,
        decoder_config: DecoderConfig,
    ) -> Result<VideoReader, ffmpeg::Error> {
        let (mut ictx, stream_index) = get_init_context(&filename)?;
        let stream_info = get_frame_count(&mut ictx, &stream_index)?;
        let decoder = Self::get_decoder(&ictx, decoder_config)?;
        debug!("frame_count: {}", stream_info.frame_count());
        debug!("key frames: {:?}", stream_info.key_frames());
        Ok(VideoReader {
            ictx,
            stream_index,
            stream_info,
            curr_frame: 0,
            curr_dec_idx: 0,
            curr_pres_idx: 0,
            n_fails: 0,
            decoder,
            draining: false,
            seek_verified: None, // Will be tested on first get_batch
            eof_sent: false,
        })
    }

    pub fn get_decoder(
        ictx: &ffmpeg::format::context::Input,
        config: DecoderConfig,
    ) -> Result<VideoDecoder, ffmpeg::Error> {
        let input = ictx
            .streams()
            .best(Type::Video)
            .ok_or(ffmpeg::Error::StreamNotFound)?;

        let fps = f64::from(input.avg_frame_rate());
        let video_params = extract_video_params(&input);

        let (decoder_context, hwaccel_context) =
            setup_decoder_context(&input, config.threads(), config.hwaccel())?;

        let mut video = decoder_context.decoder().video()?;
        let video_info = collect_video_metadata(&video, &video_params, &fps);

        let (orig_h, orig_w, orig_fmt) = (video.height(), video.width(), video.format());
        debug!(
            "Original stream info: width={}, height={}, format={:?}",
            orig_w, orig_h, orig_fmt
        );
        let pixel_aspect = {
            let sar = video.aspect_ratio();
            if sar.numerator() > 0 && sar.denominator() > 0 {
                sar
            } else {
                Rational(1, 1)
            }
        };

        let (mut height, mut width) = get_resized_dim(
            orig_h as f64,
            orig_w as f64,
            config.resize_shorter_side(),
            config.resize_longer_side(),
        );

        let is_hwaccel = hwaccel_context.is_some();
        let (filter_spec, hw_format, out_width, out_height, rotation_applied) = create_filter_spec(
            width,
            height,
            &mut video,
            config.ff_filter(),
            hwaccel_context,
            HWACCEL_PIXEL_FORMAT,
            video_params.rotation,
        )?;

        // Use the actual output dimensions (may differ from input if custom filter has scale)
        width = out_width;
        height = out_height;

        debug!(
            "Filter spec: {}, output size: {}x{}",
            filter_spec, width, height
        );
        let time_base_rational = video_info
            .get("time_base_rational")
            .ok_or(ffmpeg::Error::InvalidData)?;
        let filter_cfg = FilterConfig::new(
            orig_h,
            orig_w,
            orig_fmt,
            time_base_rational,
            filter_spec.as_str(),
            is_hwaccel,
            pixel_aspect,
        );

        let graph = create_filters(&mut video, hw_format, filter_cfg)?;

        if rotation_applied && video_params.rotation.abs() == 90 {
            std::mem::swap(&mut height, &mut width);
        }
        Ok(VideoDecoder::new(
            video, height, width, fps, video_info, is_hwaccel, graph,
        ))
    }

    pub fn decoder_start(
        &mut self,
        start_frame: Option<usize>,
        end_frame: Option<usize>,
        compression_factor: Option<f64>,
    ) -> Result<(VideoReducer, usize), ffmpeg::Error> {
        let (reducer, start_frame, _end_frame) = VideoReducer::build(
            start_frame,
            end_frame,
            *self.stream_info.frame_count(),
            compression_factor,
            self.decoder.height,
            self.decoder.width,
        );
        let first_index = start_frame.unwrap_or(0);

        // make sure we are at the begining of the stream
        self.seek_to_start()?;

        // check if first_index is after the first keyframe, if so we can seek
        if self
            .stream_info
            .key_frames()
            .iter()
            .any(|k| &first_index >= k)
            && (first_index > 0)
        {
            let key_pos = self.locate_keyframes(&first_index);
            self.seek_frame(&key_pos)?;
            self.curr_frame = key_pos;
        }

        let mut reducer = reducer.ok_or(ffmpeg::Error::Bug)?;
        reducer.set_frame_index(self.curr_frame);
        let max_idx = reducer.get_indices().iter().max().unwrap_or(&0).to_owned();
        Ok((reducer, max_idx))
    }

    pub fn decode_next(&mut self) -> Result<FrameArray, ffmpeg::Error> {
        // First, try to get a frame from decoder buffer (from previously sent packets)
        if let Some(rgb_frame) = self.decoder.decode_frames()? {
            return Ok(rgb_frame);
        }

        // Need more packets - read and send until we get a frame
        for (stream, packet) in self.ictx.packets() {
            if stream.index() == self.stream_index {
                self.decoder.video.send_packet(&packet)?;
                if let Some(rgb_frame) = self.decoder.decode_frames()? {
                    return Ok(rgb_frame);
                }
                // No frame yet, continue to next packet
            }
        }

        // No more packets, drain the decoder
        if !self.draining {
            self.decoder.video.send_eof()?;
            self.draining = true;
        }

        // Try to get remaining frames from decoder buffer
        match self.decoder.decode_frames()? {
            Some(rgb_frame) => Ok(rgb_frame),
            None => {
                self.draining = false;
                self.decoder.video.flush();
                self.seek_to_start()?;
                Err(ffmpeg::Error::Eof)
            }
        }
    }

    pub fn decode_video(
        &mut self,
        start_frame: Option<usize>,
        end_frame: Option<usize>,
        compression_factor: Option<f64>,
    ) -> Result<VideoArray, ffmpeg::Error> {
        let (mut reducer, max_idx) =
            self.decoder_start(start_frame, end_frame, compression_factor)?;
        for (stream, packet) in self.ictx.packets() {
            if reducer.get_frame_index() > max_idx {
                break;
            }
            if stream.index() == self.stream_index {
                self.decoder.video.send_packet(&packet)?;
                match self
                    .decoder
                    .receive_and_process_decoded_frames(&mut reducer)?
                {
                    Some(rgb_frame) => {
                        let mut slice_frame = reducer.slice_mut(reducer.get_idx_counter());
                        insert_frame(&mut slice_frame, rgb_frame);
                        reducer.incr_idx_counter(1);
                    }
                    None => debug!("No frame received!"),
                }
            } else {
                debug!("Packet for another stream");
            }
        }
        self.decoder.video.send_eof()?;
        // only process the remaining frames if we haven't reached the last frame
        while !reducer.no_indices() && (reducer.get_frame_index() <= max_idx) {
            match self
                .decoder
                .receive_and_process_decoded_frames(&mut reducer)?
            {
                Some(rgb_frame) => {
                    let mut slice_frame = reducer.slice_mut(reducer.get_idx_counter());
                    slice_frame.zip_mut_with(&rgb_frame, |a, b| {
                        *a = *b;
                    });
                    reducer.incr_idx_counter(1);
                }
                None => {
                    debug!("No frame received!");
                    break;
                }
            }
        }

        // reset decoder
        self.decoder.video.flush();
        self.seek_to_start()?;

        Ok(reducer.get_full_video())
    }

    pub async fn decode_video_fast(
        &mut self,
        start_frame: Option<usize>,
        end_frame: Option<usize>,
        compression_factor: Option<f64>,
    ) -> Result<Vec<FrameArray>, ffmpeg::Error> {
        let (reducer, start_frame, _) = VideoReducer::build(
            start_frame,
            end_frame,
            *self.stream_info.frame_count(),
            compression_factor,
            self.decoder.height,
            self.decoder.width,
        );
        let first_index = start_frame.unwrap_or(0);

        // make sure we are at the begining of the stream
        self.seek_to_start()?;

        if self
            .stream_info
            .key_frames()
            .iter()
            .any(|k| &first_index >= k)
            && (first_index > 0)
        {
            let key_pos = self.locate_keyframes(&first_index);
            // seek to closest key frame before first_index
            self.seek_frame(&key_pos)?;
        }

        let mut reducer = reducer.ok_or(ffmpeg::Error::Bug)?;
        reducer.set_frame_index(self.curr_frame);
        let mut tasks: Vec<task::JoinHandle<Result<FrameArray, ffmpeg::Error>>> = vec![];

        let mut receive_and_process_decoded_frames = |decoder: &mut ffmpeg::decoder::Video,
                                                      mut curr_frame: usize|
         -> Result<usize, ffmpeg::Error> {
            let mut decoded = Video::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                if reducer.get_indices().iter().any(|x| x == &curr_frame) {
                    self.decoder
                        .graph
                        .get("in")
                        .ok_or(ffmpeg::Error::Bug)?
                        .source()
                        .add(&decoded)?;
                    let mut rgb_frame = Video::empty();
                    if self
                        .decoder
                        .graph
                        .get("out")
                        .ok_or(ffmpeg::Error::Bug)?
                        .sink()
                        .frame(&mut rgb_frame)
                        .is_ok()
                    {
                        let cspace = self.decoder.color_space;
                        let crange = self.decoder.color_range;
                        if self.decoder.is_hwaccel {
                            tasks.push(task::spawn(async move {
                                convert_nv12_to_ndarray_rgb24(rgb_frame, cspace, crange)
                            }));
                        } else {
                            tasks.push(task::spawn(async move {
                                convert_yuv_to_ndarray_rgb24(rgb_frame, cspace, crange)
                            }));
                        }
                    }
                }
                curr_frame += 1;
            }
            Ok(curr_frame)
        };

        for (stream, packet) in self.ictx.packets() {
            if &self.curr_frame > reducer.get_indices().iter().max().unwrap_or(&0) {
                break;
            }
            if stream.index() == self.stream_index {
                self.decoder.video.send_packet(&packet)?;
                let upd_curr_frame =
                    receive_and_process_decoded_frames(&mut self.decoder.video, self.curr_frame)?;
                self.curr_frame = upd_curr_frame;
            } else {
                debug!("Packet for another stream");
            }
        }
        self.decoder.video.send_eof()?;
        // only process the remaining frames if we haven't reached the last frame
        if !reducer.no_indices()
            && (&self.curr_frame <= reducer.get_indices().iter().max().unwrap_or(&0))
        {
            let upd_curr_frame =
                receive_and_process_decoded_frames(&mut self.decoder.video, self.curr_frame)?;
            self.curr_frame = upd_curr_frame;
        }

        let mut outputs = Vec::with_capacity(tasks.len());
        for task_ in tasks {
            outputs.push(task_.await.map_err(|_| ffmpeg::Error::Bug)??);
        }

        // flush and go back to start
        self.decoder.video.flush();
        self.seek_to_start()?;

        Ok(outputs)
    }
    /// Safely get the batch of frames from the video by iterating over all frames and decoding
    /// only the ones we need. This can be more accurate when the video's metadata is not reliable,
    /// or when the video has B-frames.
    pub fn get_batch_safe(&mut self, indices: Vec<usize>) -> Result<VideoArray, ffmpeg::Error> {
        // Simple and robust implementation: iterate once from the start and grab the requested
        // frames in presentation order. This avoids any reliance on seeking or timestamp quirks
        // (B-frames, non-monotonic DTS/PTS, etc.).
        //
        // Complexity is O(total_frames) but only used when caller asks for the safe path.

        // make sure we are at the beginning of the stream
        self.seek_to_start()?;

        // For fast membership checks
        let needed: HashSet<usize> = indices.iter().cloned().collect();
        let needed_total = needed.len();
        let max_needed = indices.iter().max().copied().unwrap_or(0);
        let mut frame_map: HashMap<usize, FrameArray> = HashMap::with_capacity(needed.len());

        let mut decoded = Video::empty();
        let mut curr_idx: usize = 0;
        let mut collected: usize = 0;

        // iterate all packets
        'packets: for (stream, packet) in self.ictx.packets() {
            if stream.index() != self.stream_index {
                continue;
            }
            self.decoder.video.send_packet(&packet)?;
            while self.decoder.video.receive_frame(&mut decoded).is_ok() {
                if needed.contains(&curr_idx) {
                    // push frame through filter graph, offload color conversion to async task
                    self.decoder
                        .graph
                        .get("in")
                        .ok_or(ffmpeg::Error::Bug)?
                        .source()
                        .add(&decoded)?;
                    let mut rgb_frame = Video::empty();
                    if self
                        .decoder
                        .graph
                        .get("out")
                        .ok_or(ffmpeg::Error::Bug)?
                        .sink()
                        .frame(&mut rgb_frame)
                        .is_ok()
                    {
                        let cspace = self.decoder.color_space;
                        let crange = self.decoder.color_range;
                        let frame = if self.decoder.is_hwaccel {
                            convert_nv12_to_ndarray_rgb24(rgb_frame, cspace, crange)?
                        } else {
                            convert_yuv_to_ndarray_rgb24(rgb_frame, cspace, crange)?
                        };
                        frame_map.insert(curr_idx, frame);
                        collected += 1;
                    }
                }
                curr_idx += 1;
                if collected >= needed_total || curr_idx > max_needed {
                    break 'packets;
                }
            }
        }
        // flush remaining frames
        self.decoder.video.send_eof()?;
        while self.decoder.video.receive_frame(&mut decoded).is_ok() {
            if needed.contains(&curr_idx) {
                self.decoder
                    .graph
                    .get("in")
                    .ok_or(ffmpeg::Error::Bug)?
                    .source()
                    .add(&decoded)?;
                let mut rgb_frame = Video::empty();
                if self
                    .decoder
                    .graph
                    .get("out")
                    .ok_or(ffmpeg::Error::Bug)?
                    .sink()
                    .frame(&mut rgb_frame)
                    .is_ok()
                {
                    let cspace = self.decoder.color_space;
                    let crange = self.decoder.color_range;
                    let frame = if self.decoder.is_hwaccel {
                        convert_nv12_to_ndarray_rgb24(rgb_frame, cspace, crange)?
                    } else {
                        convert_yuv_to_ndarray_rgb24(rgb_frame, cspace, crange)?
                    };
                    frame_map.insert(curr_idx, frame);
                    collected += 1;
                }
            }
            curr_idx += 1;
            if collected >= needed_total || curr_idx > max_needed {
                break;
            }
        }

        // Build output in the same order as requested (including duplicates)
        let mut frame_batch: VideoArray = Array4::zeros((
            indices.len(),
            self.decoder.height as usize,
            self.decoder.width as usize,
            3,
        ));

        for (out_i, idx) in indices.iter().enumerate() {
            if let Some(frame) = frame_map.get(idx) {
                frame_batch.slice_mut(s![out_i, .., .., ..]).assign(frame);
            } else {
                debug!("No frame found for {}", idx);
            }
        }

        // reset decoder state for subsequent calls
        self.decoder.video.flush();
        self.seek_to_start()?;
        Ok(frame_batch)
    }

    /// Get the batch of frames from the video by seeking to the closest keyframe and skipping
    /// the frames until we reach the desired frame index. Heavily inspired by the implementation
    /// from decord library: https://github.com/dmlc/decord
    ///
    /// Sorts indices internally to minimize backward seeks.
    pub fn get_batch(&mut self, indices: Vec<usize>) -> Result<VideoArray, ffmpeg::Error> {
        // Sort indices to minimize backward seeks (keep track of original positions)
        let mut sorted_indices: Vec<(usize, usize)> = indices
            .iter()
            .enumerate()
            .map(|(orig_idx, &frame_idx)| (orig_idx, frame_idx))
            .collect();
        sorted_indices.sort_by_key(|&(_, frame_idx)| frame_idx);

        // Map frame_idx -> all output positions needing it (to preserve caller order & handle duplicates)
        let mut positions_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for (orig_idx, frame_idx) in &sorted_indices {
            positions_map.entry(*frame_idx).or_default().push(*orig_idx);
        }

        // Deduplicate frame indices (avoid decoding same frame twice)
        let mut unique_frames: Vec<usize> = positions_map.keys().copied().collect();
        unique_frames.sort_unstable();

        // make sure we are at the beginning of the stream
        self.seek_to_start()?;

        // Allocate output buffer once
        let mut video_frames: VideoArray = Array::zeros((
            indices.len(),
            self.decoder.height as usize,
            self.decoder.width as usize,
            3,
        ));

        let cspace = self.decoder.color_space;
        let crange = self.decoder.color_range;
        let is_hwaccel = self.decoder.is_hwaccel;

        // Process frames in sorted order (minimizes seeks)
        for frame_index in unique_frames {
            self.n_fails = 0;
            debug!("[NEXT INDICE] frame_index: {frame_index}");

            match self.seek_accurate_raw(frame_index) {
                Ok(Some(yuv_frame)) => {
                    let rgb_frame = if is_hwaccel {
                        convert_nv12_to_ndarray_rgb24(yuv_frame, cspace, crange)
                    } else {
                        convert_yuv_to_ndarray_rgb24(yuv_frame, cspace, crange)
                    }?;

                    if let Some(positions) = positions_map.get(&frame_index) {
                        for pos in positions {
                            video_frames
                                .slice_mut(s![*pos, .., .., ..])
                                .assign(&rgb_frame);
                        }
                    }
                }
                Ok(None) => debug!("No frame found for frame index {}", frame_index),
                Err(e) => {
                    debug!("seek_accurate_raw failed at {}: {:?}", frame_index, e);
                    return Err(e);
                }
            }
        }

        Ok(video_frames)
    }

    /// Returns the raw YUV frame (after filter graph) for a given presentation index.
    /// Uses frame counting to handle B-frame reordering correctly.
    pub fn seek_accurate_raw(
        &mut self,
        presentation_idx: usize,
    ) -> Result<Option<Video>, ffmpeg::Error> {
        // Get the decode index (packet order) for this presentation index
        // This tells us which packet produces the frame we want
        let target_decode_idx = match self
            .stream_info
            .get_decode_idx_for_presentation(presentation_idx)
        {
            Some(idx) => idx,
            None => {
                debug!(
                    "No decode index found for presentation index {}",
                    presentation_idx
                );
                return Ok(None);
            }
        };

        // Find the keyframe before this decode index
        let key_decode_idx = self.locate_keyframes(&target_decode_idx);

        // Get the presentation index of the keyframe
        // After seeking to this keyframe, the decoder will output frames starting from this presentation index
        let key_pres_idx = self
            .stream_info
            .get_presentation_idx_for_decode(key_decode_idx)
            .unwrap_or(key_decode_idx);

        // Calculate how many frames to skip after seeking
        let frames_to_skip = presentation_idx.saturating_sub(key_pres_idx);

        debug!(
            "    - [RAW] Presentation idx: {}, decode idx: {}, keyframe decode: {}, keyframe pres: {}, skip: {}",
            presentation_idx, target_decode_idx, key_decode_idx, key_pres_idx, frames_to_skip
        );

        // Check if we can skip forward without seeking
        // We track curr_pres_idx (presentation order count)
        // IMPORTANT: If we've sent EOF, we MUST re-seek because the decoder state is invalid
        let can_skip_forward = !self.eof_sent
            && presentation_idx >= self.curr_pres_idx
            && self.curr_pres_idx >= key_pres_idx;

        if can_skip_forward {
            debug!(
                "No need to seek, we can directly skip frames (curr_pres: {})",
                self.curr_pres_idx
            );
            let skip_count = presentation_idx.saturating_sub(self.curr_pres_idx);
            match self.skip_frames_raw_by_count(skip_count) {
                Ok(frame) => Ok(frame),
                Err(_) => self.get_frame_raw_after_eof_by_count(presentation_idx),
            }
        } else {
            debug!("Seeking to keyframe at decode idx: {}", key_decode_idx);

            self.seek_to_start()?;
            self.seek_frame_by_decode_idx(&key_decode_idx)?;
            self.curr_frame = key_decode_idx;
            self.curr_dec_idx = key_decode_idx;
            self.curr_pres_idx = key_pres_idx;

            match self.skip_frames_raw_by_count(frames_to_skip) {
                Ok(frame) => Ok(frame),
                Err(_) => self.get_frame_raw_after_eof_by_count(presentation_idx),
            }
        }
    }

    /// Skip `skip_count` frames and return the next raw YUV frame
    /// Used for frame counting approach (handles B-frame reordering)
    pub fn skip_frames_raw_by_count(
        &mut self,
        skip_count: usize,
    ) -> Result<Option<Video>, ffmpeg::Error> {
        debug!(
            "Skipping {} frames, starting from pres_idx={}",
            skip_count, self.curr_pres_idx
        );
        let target_pres_idx = self.curr_pres_idx + skip_count;
        let mut failsafe = (self.stream_info.frame_count() * 2) as i32;

        // First, try to get frame from decoder's existing buffer (from previous packets)
        let (yuv_frame, counter) = self.get_frame_raw_by_count(target_pres_idx);
        if yuv_frame.is_some() {
            return Ok(yuv_frame);
        }
        failsafe -= counter;

        // Need more packets
        while failsafe > -1 {
            match self.ictx.packets().next() {
                Some((stream, packet)) => {
                    if stream.index() == self.stream_index {
                        self.decoder.video.send_packet(&packet)?;
                        self.curr_dec_idx += 1;
                        let (yuv_frame, counter) = self.get_frame_raw_by_count(target_pres_idx);
                        if yuv_frame.is_some() {
                            return Ok(yuv_frame);
                        }
                        failsafe -= counter
                    }
                }
                None => {
                    debug!("No more packet!");
                    return Err(ffmpeg::Error::Eof);
                }
            }
        }
        debug!("Failed to find frame at pres_idx={}", target_pres_idx);
        Err(ffmpeg::Error::Eof)
    }

    /// Get raw YUV frame (after filter graph) at target presentation index
    /// Counts decoded frames and returns when reaching the target
    pub fn get_frame_raw_by_count(&mut self, target_pres_idx: usize) -> (Option<Video>, i32) {
        let mut decoded = Video::empty();
        let mut counter = 0;
        let expected_pts = self.stream_info.get_pts_for_presentation(target_pres_idx);
        let mut pres_idx_from_pts: Option<usize> = None;

        while self.decoder.video.receive_frame(&mut decoded).is_ok() {
            let frame_pts = decoded.pts().unwrap_or(-1);
            // debug!(
            //     "Decoded frame (raw): pres_idx={}, target={}, frame_pts={}",
            //     self.curr_pres_idx, target_pres_idx, frame_pts
            // );

            // Push through filter graph and get YUV frame
            if let Some(mut in_ctx) = self.decoder.graph.get("in") {
                if let Err(e) = in_ctx.source().add(&decoded) {
                    debug!("Failed to push frame into filter graph: {e}");
                    return (None, counter);
                }
            } else {
                debug!("Filter graph missing 'in' pad");
                return (None, counter);
            }

            let mut yuv_frame = Video::empty();
            // For simple scale filters, output is immediate (no delay)
            if let Some(mut out_ctx) = self.decoder.graph.get("out") {
                if out_ctx.sink().frame(&mut yuv_frame).is_ok() {
                    let out_pts = yuv_frame.pts();
                    // If we have PTS, try to map back to presentation index to resync after seek
                    if let Some(p) = out_pts {
                        if let Some(mapped) = self.stream_info.presentation_for_pts_norm(p) {
                            pres_idx_from_pts = Some(mapped);
                        }
                    }
                    let mut matched = self.curr_pres_idx == target_pres_idx;
                    if let (Some((exp_raw, exp_norm)), Some(p)) = (expected_pts, out_pts) {
                        if p == exp_norm || p == exp_raw {
                            matched = true;
                        }
                    }
                    if matched {
                        let out_pts_dbg = out_pts.unwrap_or(-1);
                        debug!(
                            "Found target frame at pres_idx={}, pts={}",
                            target_pres_idx, out_pts_dbg
                        );
                        self.curr_pres_idx += 1;
                        return (Some(yuv_frame), counter + 1);
                    }
                }
            } else {
                debug!("Filter graph missing 'out' pad");
                return (None, counter);
            }

            // Advance presentation index, preferring mapped value if we have it
            if let Some(mapped) = pres_idx_from_pts {
                self.curr_pres_idx = mapped + 1;
                pres_idx_from_pts = None;
            } else {
                self.curr_pres_idx += 1;
            }
            counter += 1;
        }

        (None, counter)
    }

    /// Get raw frame when there are no more packets to iterate
    /// `target_pres_idx` is the original target presentation index we're looking for
    pub fn get_frame_raw_after_eof_by_count(
        &mut self,
        target_pres_idx: usize,
    ) -> Result<Option<Video>, ffmpeg::Error> {
        debug!(
            "get_frame_raw_after_eof_by_count: target_pres_idx={}",
            target_pres_idx
        );
        self.decoder.video.send_eof()?;
        self.eof_sent = true; // Mark that we've sent EOF - need to re-seek before processing more frames
                              // Use the original target, not self.curr_pres_idx!
        let (yuv_frame, _counter) = self.get_frame_raw_by_count(target_pres_idx);
        Ok(yuv_frame)
    }

    /// Find the closest key frame before or at `pos` using binary search
    pub fn locate_keyframes(&self, pos: &usize) -> usize {
        // Borrow directly to avoid per-call cloning; borrow scope ends within this function
        let key_frames = self.stream_info.key_frames();
        if key_frames.is_empty() {
            return 0;
        }

        // Binary search for the largest keyframe <= pos
        match key_frames.binary_search(pos) {
            Ok(idx) => key_frames[idx], // Exact match
            Err(idx) => {
                if idx == 0 {
                    0 // pos is before first keyframe
                } else {
                    key_frames[idx - 1] // Keyframe just before pos
                }
            }
        }
    }
    /// Quick static checks before runtime seek verification.
    /// Returns (force_sequential, force_full_verify, summary)
    fn quick_seek_static_check(&self) -> (bool, bool, String) {
        let mut force_sequential = false;
        let mut force_full_verify = false;
        let mut reasons: Vec<String> = Vec::new();

        let key_frames = self.stream_info.key_frames();
        let frame_times = self.stream_info.frame_times();

        if self.stream_info.has_missing_pts() {
            force_sequential = true;
            reasons.push("missing pts".to_string());
        }
        if self.stream_info.has_missing_dts() {
            force_sequential = true;
            reasons.push("missing dts".to_string());
        }
        if self.stream_info.has_non_monotonic_pts() {
            force_sequential = true;
            reasons.push("non-monotonic pts".to_string());
        }
        if self.stream_info.has_non_monotonic_dts() {
            force_sequential = true;
            reasons.push("non-monotonic dts".to_string());
        }

        // Missing timing info for keyframes: cannot trust seek -> force sequential
        if key_frames.iter().any(|kf| !frame_times.contains_key(kf)) {
            force_sequential = true;
            reasons.push("missing frame_times for some keyframes".to_string());
        }

        // PTS monotonicity on keyframes (normalized) - if violated, do full verify
        let (pts_mono, pts_violation) = self.stream_info.keyframe_pts_monotonic_norm();
        if !pts_mono {
            force_full_verify = true;
            reasons.push(format!(
                "keyframe pts non-monotonic ({} windows)",
                pts_violation
            ));
        }

        // DTS monotonicity on keyframes (decode order) - if violated, do full verify
        let (dts_mono, dts_violation) = self.stream_info.keyframe_dts_monotonic();
        if !dts_mono {
            force_full_verify = true;
            reasons.push(format!(
                "keyframe dts non-monotonic ({} windows)",
                dts_violation
            ));
        }

        // // Keyframe gap outlier check: very long gap relative to median -> full verify
        // if key_frames.len() > 4 {
        //     let mut gaps: Vec<usize> = key_frames.windows(2).map(|w| w[1] - w[0]).collect();
        //     gaps.sort();
        //     let median = gaps[gaps.len() / 2].max(1);
        //     let max_gap = *gaps.last().unwrap_or(&median);
        //     if max_gap > median * 4 && max_gap > 200 {
        //         force_full_verify = true;
        //         reasons.push(format!(
        //             "keyframe gap outlier: max={} median={}",
        //             max_gap, median
        //         ));
        //     }
        // }

        let summary = if reasons.is_empty() {
            "ok".to_string()
        } else {
            reasons.join("; ")
        };
        (force_sequential, force_full_verify, summary)
    }

    /// Check if this video needs sequential mode
    /// Returns true if seek is known to be unreliable for this video
    /// Result is cached after first call to avoid repeated verification overhead
    pub fn needs_sequential_mode(&mut self) -> bool {
        let t0 = Instant::now();
        // Check cached verification result
        if let Some(seek_works) = self.seek_verified {
            if !seek_works {
                debug!(
                    "needs_sequential_mode: Cached result - seek failed (cost {:?})",
                    t0.elapsed()
                );
            }
            return !seek_works;
        }

        // For normal videos, do a runtime seek verification (only once)
        // Some videos have seek issues even with positive PTS/DTS
        let quick_start = Instant::now();
        let (force_seq, force_full_verify, quick_summary) = self.quick_seek_static_check();
        let quick_elapsed = quick_start.elapsed();
        if force_seq {
            debug!(
                "needs_sequential_mode: quick static check -> sequential (reason: {}) (quick {:?}, total {:?})",
                quick_summary,
                quick_elapsed,
                t0.elapsed()
            );
            return true;
        }

        debug!(
            "needs_sequential_mode: quick static check ok (reason: {}) force_full_verify={} (quick {:?})",
            quick_summary,
            force_full_verify,
            quick_elapsed
        );
        
        // TODO: (cy) removed this temporary fix; TO BE WATCHED! changed to the following:
        // let seek_works = self.verify_seek_works(force_full_verify);
        let seek_works = true; // Temporarily force seek path to always succeed
        self.seek_verified = Some(seek_works); // Cache the result

        if !seek_works {
            debug!(
                "needs_sequential_mode: Runtime verification failed - sequential (cost {:?})",
                t0.elapsed()
            );
            return true;
        }

        debug!(
            "needs_sequential_mode: seek verified ok (cost {:?})",
            t0.elapsed()
        );
        // Normal videos that pass verification - seek should work fine
        false
    }

    /// Verify if seek works correctly by testing seeks to several keyframes.
    /// Two-phase sampling: a small set first, then remaining points if all pass.
    /// `force_full_verify` skips the two-phase shortcut and verifies all points.
    /// Returns true only if all sampled keyframes can be reached reliably.
    fn verify_seek_works(&mut self, force_full_verify: bool) -> bool {
        let verify_start = Instant::now();
        let key_frames = self.stream_info.key_frames().clone();
        // Need at least 2 keyframes to verify; otherwise assume ok
        if key_frames.len() < 2 {
            return true;
        }

        // Build full candidate set to avoid missing locally broken GOPs
        let mut full_candidates: Vec<usize> = Vec::new();
        if key_frames.len() > 1 {
            full_candidates.push(1);
        }
        if key_frames.len() > 2 {
            full_candidates.push(2);
        }
        if key_frames.len() > 4 {
            full_candidates.push(4);
        }
        full_candidates.push(key_frames.len() / 2);
        if key_frames.len() > 2 {
            full_candidates.push(key_frames.len().saturating_sub(2));
        }
        full_candidates.sort();
        full_candidates.dedup();

        // Phase 1: small sample set
        let mut phase1: Vec<usize> = Vec::new();
        if key_frames.len() > 1 {
            phase1.push(1);
        }
        phase1.push(key_frames.len() / 2);
        if key_frames.len() > 2 {
            phase1.push(key_frames.len().saturating_sub(2));
        }
        phase1.sort();
        phase1.dedup();

        // Helper to verify a list of keyframe indices
        let mut verify_candidates = |candidates: &[usize]| -> bool {
            for &key_idx in candidates {
                let key_start = Instant::now();
                let decode_idx = key_frames[key_idx];
                let expected_pts = match self.stream_info.frame_times().get(&decode_idx) {
                    Some(ft) => *ft.pts() - self.stream_info.min_pts_offset(),
                    None => continue, // skip if missing pts
                };

                debug!(
                    "verify_seek_works: testing keyframe_idx={} decode_idx={} expected_pts={}",
                    key_idx, decode_idx, expected_pts
                );

                if self.seek_to_start().is_err() {
                    return false;
                }

                if self.avseekframe(&decode_idx, expected_pts, 1).is_err() {
                    debug!("verify_seek_works: seek failed at keyframe_idx={}", key_idx);
                    return false;
                }

                let mut decoded_pts_values: Vec<i64> = Vec::new();
                let mut packets_sent = 0;
                let max_packets = 40;
                for (stream, packet) in self.ictx.packets() {
                    if stream.index() != self.stream_index {
                        continue;
                    }
                    if self.decoder.video.send_packet(&packet).is_err() {
                        continue;
                    }
                    packets_sent += 1;

                    let mut decoded = Video::empty();
                    while self.decoder.video.receive_frame(&mut decoded).is_ok() {
                        if let Some(pts) = decoded.pts() {
                            decoded_pts_values.push(pts);
                        }
                    }
                    if packets_sent >= max_packets || decoded_pts_values.len() >= 8 {
                        break;
                    }
                }

                debug!(
                    "verify_seek_works: keyframe_idx={} packets_sent={} decoded_frames={:?}",
                    key_idx, packets_sent, decoded_pts_values
                );

                // Reset state for future operations
                let _ = self.seek_to_start();

                let found = decoded_pts_values.iter().any(|&pts| pts == expected_pts);
                if !found {
                    debug!(
                        "verify_seek_works: FAILED keyframe_idx={} expected_pts={} got {:?} (cost {:?})",
                        key_idx, expected_pts, decoded_pts_values, key_start.elapsed()
                    );
                    return false;
                } else {
                    debug!(
                        "verify_seek_works: PASSED keyframe_idx={} expected_pts={} (cost {:?})",
                        key_idx,
                        expected_pts,
                        key_start.elapsed()
                    );
                }
            }
            true
        };

        // Decide phases
        if force_full_verify {
            let ok = verify_candidates(&full_candidates);
            debug!(
                "verify_seek_works: full verification {} (cost {:?})",
                if ok { "passed" } else { "failed" },
                verify_start.elapsed()
            );
            return ok;
        }

        if !verify_candidates(&phase1) {
            return false;
        }

        // Remaining candidates after phase1
        let phase1_set: HashSet<usize> = phase1.iter().copied().collect();
        let mut remaining: Vec<usize> = full_candidates
            .into_iter()
            .filter(|idx| !phase1_set.contains(idx))
            .collect();
        remaining.sort();
        remaining.dedup();

        if remaining.is_empty() {
            debug!(
                "verify_seek_works: phase1 passed; no remaining candidates (cost {:?})",
                verify_start.elapsed()
            );
            return true;
        }

        let ok = verify_candidates(&remaining);
        debug!(
            "verify_seek_works: all sampled keyframes {} (cost {:?})",
            if ok { "passed" } else { "failed" },
            verify_start.elapsed()
        );
        ok
    }

    /// Detailed cost estimation for seek-based vs sequential methods
    /// Returns (seek_frames, seek_count, sequential_frames, unique_count, max_index)
    pub fn estimate_decode_cost_detailed(
        &self,
        indices: &[usize],
    ) -> (usize, usize, usize, usize, usize) {
        if indices.is_empty() {
            return (0, 0, 0, 0, 0);
        }

        // Deduplicate and sort for seek-based estimation
        let mut unique_sorted: Vec<usize> = indices.iter().cloned().collect();
        unique_sorted.sort();
        unique_sorted.dedup();
        let unique_count = unique_sorted.len();

        let key_frames = self.stream_info.key_frames();
        let has_keyframes = !key_frames.is_empty();

        // Sequential cost: decode from 0 to max_index
        let max_index = *unique_sorted.last().unwrap_or(&0);
        let sequential_frames = max_index + 1;

        // Seek-based cost: count frames decoded AND number of seeks (GOP transitions)
        let mut seek_frames = 0;
        let mut seek_count = 0;
        let mut last_info: Option<(usize, usize)> = None; // (last_idx, last_keyframe)

        for &idx in &unique_sorted {
            // Binary search keyframe on pre-fetched slice to avoid repeated method overhead
            let keyframe = if has_keyframes {
                match key_frames.binary_search(&idx) {
                    Ok(i) => key_frames[i],
                    Err(i) => {
                        if i == 0 {
                            0
                        } else {
                            key_frames[i - 1]
                        }
                    }
                }
            } else {
                0
            };

            match last_info {
                Some((last_idx, last_keyframe)) => {
                    if keyframe == last_keyframe && idx > last_idx {
                        // Same GOP, can skip forward: only decode (idx - last_idx) frames
                        seek_frames += idx - last_idx;
                        // No seek needed
                    } else {
                        // Different GOP: need to seek, decode from keyframe
                        seek_count += 1;
                        seek_frames += idx - keyframe + 1;
                    }
                }
                None => {
                    // First frame: seek to keyframe (counts as first seek)
                    seek_count += 1;
                    seek_frames += idx - keyframe + 1;
                }
            }
            last_info = Some((idx, keyframe));
        }

        (
            seek_frames,
            seek_count,
            sequential_frames,
            unique_count,
            max_index,
        )
    }

    /// Estimate decode cost for seek-based vs sequential methods
    /// Returns (seek_cost, sequential_cost) where each is the estimated number of frames to decode
    pub fn estimate_decode_cost(&self, indices: &[usize]) -> (usize, usize) {
        let (seek_frames, _, sequential_frames, _, _) = self.estimate_decode_cost_detailed(indices);
        (seek_frames, sequential_frames)
    }

    /// Recommend whether to use seek-based (false) or sequential (true) method
    /// Returns true if sequential is estimated to be faster
    pub fn should_use_sequential(&self, indices: &[usize]) -> bool {
        // Note: negative PTS/DTS check is now done at a higher level via needs_sequential_mode()
        // which actually verifies if seek works, rather than just checking metadata

        let (seek_frames, seek_count, sequential_frames, unique_count, _max_index) =
            self.estimate_decode_cost_detailed(indices);

        if unique_count == 0 {
            return true;
        }

        // Cost model after benchmarking:
        // - When seek_frames ≈ sequential_frames, sequential wins (simpler, cache-friendly)
        // - Seek only wins when it can skip significant portions of the video
        // - Many GOP transitions (seek_count) add overhead even with skip-forward

        debug!(
            "Cost estimation: seek_frames={}, seek_count={}, sequential={}",
            seek_frames, seek_count, sequential_frames
        );

        // Decision rules:
        // 1. If seek_frames >= seq * 0.9, use sequential (not saving enough)
        // 2. If many GOP transitions (>5) AND seek_frames >= seq * 0.7, use sequential
        //    (each GOP transition has I/O and decoder reset overhead)
        // 3. Otherwise use seek

        if seek_frames as f64 >= sequential_frames as f64 * 0.9 {
            return true; // Not saving enough, use sequential
        }

        if seek_count > 5 && seek_frames as f64 >= sequential_frames as f64 * 0.7 {
            return true; // Many seeks and not saving much, use sequential
        }

        false // Use seek - significant savings
    }

    /// Seek back to the begining of the stream
    fn seek_to_start(&mut self) -> Result<(), ffmpeg::Error> {
        self.ictx.seek(0, ..100)?;
        self.avflushbuf()?;
        self.curr_dec_idx = 0;
        self.curr_frame = 0;
        self.curr_pres_idx = 0;
        self.eof_sent = false; // Reset EOF state after seeking
        Ok(())
    }

    /// Count actual decodable frames by decoding without color conversion.
    /// This is slower than packet counting but gives accurate results for B-frame videos.
    /// Equivalent to ffprobe's `nb_read_frames` with `-count_frames` option.
    pub fn count_actual_frames(&mut self) -> usize {
        // Seek to start
        if self.seek_to_start().is_err() {
            return 0;
        }

        let mut count = 0;
        let mut decoded = Video::empty();

        // Iterate through all packets and decode (without RGB conversion)
        for (stream, packet) in self.ictx.packets() {
            if stream.index() == self.stream_index {
                // Try to send packet; if decoder queue is full (EAGAIN), drain frames then retry
                let mut sent = false;
                while !sent {
                    match self.decoder.video.send_packet(&packet) {
                        Ok(()) => {
                            sent = true;
                        }
                        Err(ffmpeg::Error::Other { errno })
                            if errno == ffmpeg::error::EAGAIN =>
                        {
                            while self.decoder.video.receive_frame(&mut decoded).is_ok() {
                                count += 1;
                            }
                            continue; // Drain then retry send_packet
                        }
                        Err(_) => break, // Ignore other errors (e.g., corrupted packet)
                    }
                }

                // Count all frames that come out of the decoder
                while self.decoder.video.receive_frame(&mut decoded).is_ok() {
                    count += 1;
                }
            }
        }

        // Drain remaining buffered frames (important for B-frame videos)
        if self.decoder.video.send_eof().is_ok() {
            while self.decoder.video.receive_frame(&mut decoded).is_ok() {
                count += 1;
            }
        }

        // Reset decoder state
        self.decoder.video.flush();
        let _ = self.seek_to_start();

        count
    }

    // AVSEEK_FLAG_BACKWARD 1 <- seek backward
    // AVSEEK_FLAG_BYTE 2 <- seeking based on position in bytes
    // AVSEEK_FLAG_ANY 4 <- seek to any frame, even non-key frames
    // AVSEEK_FLAG_FRAME 8 <- seeking based on frame number
    /// Seek to a decode order position (packet index)
    pub fn seek_frame_by_decode_idx(&mut self, decode_idx: &usize) -> Result<(), ffmpeg::Error> {
        if let Some(fr_ts) = self.stream_info.frame_times().get(decode_idx) {
            let pts = *fr_ts.pts();
            let dts = *fr_ts.dts();
            let pts_norm = pts - self.stream_info.min_pts_offset();
            let dts_norm = dts - self.stream_info.min_dts_offset();

            debug!(
                "seek_frame_by_decode_idx: decode_idx={}, pts={}, pts_norm={}, dts={}, dts_norm={}",
                decode_idx, pts, pts_norm, dts, dts_norm
            );

            // // For negative DTS videos (PTS is positive), use flag=0
            // // For normal videos, use flag=1 (AVSEEK_FLAG_BACKWARD) to seek to keyframe before timestamp
            // let seek_flag = if self.stream_info.has_negative_dts() {
            //     0
            // } else {
            //     1
            // };
            // TODO: (cy) temporaly force seek flag to 1, although might be sub-optimal
            let seek_flag = 1;

            match self.avseekframe(decode_idx, pts_norm, seek_flag) {
                Ok(()) => {
                    debug!(
                        "seek_frame_by_decode_idx: seek with pts={} (flag={}) succeeded",
                        pts_norm, seek_flag
                    );
                    Ok(())
                }
                Err(_) => {
                    // Try with DTS
                    debug!(
                        "seek_frame_by_decode_idx: trying with dts={} (flag={})",
                        dts_norm, seek_flag
                    );
                    self.avseekframe(decode_idx, dts_norm, seek_flag)
                }
            }
        } else {
            debug!(
                "seek_frame_by_decode_idx: decode_idx={}, no frame_times entry",
                decode_idx
            );
            Err(ffmpeg::Error::Bug)
        }
    }

    // Legacy seek_frame for compatibility
    pub fn seek_frame(&mut self, pos: &usize) -> Result<(), ffmpeg::Error> {
        self.seek_frame_by_decode_idx(pos)
    }
    pub fn avseekframe(
        &mut self,
        pos: &usize,
        frame_ts: i64,
        flag: i32,
    ) -> Result<(), ffmpeg::Error> {
        let res = unsafe {
            av_seek_frame(
                self.ictx.as_mut_ptr(),
                self.stream_index as i32,
                frame_ts,
                flag,
            )
        };
        self.avflushbuf()?;
        if res >= 0 {
            self.curr_dec_idx = *pos;
            self.curr_frame = *pos;
            Ok(())
        } else {
            Err(ffmpeg::Error::from(res))
        }
    }

    pub fn avflushbuf(&mut self) -> Result<(), ffmpeg::Error> {
        // Flush decoder buffers to discard any pending frames
        unsafe { avcodec_flush_buffers(self.decoder.video.as_mut_ptr()) };
        // Reset EOF state - after flushing, we can decode again
        self.eof_sent = false;
        // Also drain any remaining frames from the decoder
        let mut decoded = Video::empty();
        while self.decoder.video.receive_frame(&mut decoded).is_ok() {
            // Discard buffered frames
            debug!("Discarding buffered decoder frame after flush");
        }
        // Also flush the filter graph to discard any buffered frames there
        let mut filter_frame = Video::empty();
        while self
            .decoder
            .graph
            .get("out")
            .and_then(|mut ctx| ctx.sink().frame(&mut filter_frame).ok())
            .is_some()
        {
            debug!("Discarding buffered filter frame after flush");
        }
        Ok(())
    }
}

impl Iterator for VideoReader {
    type Item = FrameArray;
    fn next(&mut self) -> Option<Self::Item> {
        self.decode_next().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffmpeg::format::input;
    use ffmpeg::media::Type;
    use std::path::Path;

    const TEST_VIDEO: &'static str = "./assets/input.mp4";
    const TEST_AUDIO: &'static str = "./assets/audio_only.mp3";
    const NO_FILE: &'static str = "./assets/non_existent_file.mp4";

    #[test]
    fn test_get_init_context_success() {
        let filename = String::from(TEST_VIDEO);
        let result = get_init_context(&filename);
        assert!(result.is_ok());

        let (ctx, stream_index) = result.expect("context should be ok");
        assert!(stream_index < ctx.streams().count());

        let stream = ctx.streams().find(|s| s.index() == stream_index);
        assert!(stream.is_some());
        let stream = stream.expect("stream should exist");
        assert_eq!(stream.parameters().medium(), Type::Video);
    }

    #[test]
    fn test_get_init_context_file_not_found() {
        let filename = String::from(NO_FILE);
        let result = get_init_context(&filename);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_init_context_no_video_stream() {
        // This test requires a file with no video streams, like an audio-only file
        let filename = String::from(TEST_AUDIO);
        let result = get_init_context(&filename);
        assert!(result.is_err());
    }

    #[test]
    fn test_setup_decoder_context_no_hwaccel() {
        let path = Path::new(TEST_VIDEO);
        ffmpeg::init().expect("ffmpeg init failed");
        let ictx = input(&path).expect("open input failed");
        let stream = ictx
            .streams()
            .best(Type::Video)
            .expect("No video stream found");

        // test with multiple threads
        let result = setup_decoder_context(&stream, 4, None);
        assert!(result.is_ok());
        let (context, hwaccel) = result.expect("setup_decoder_context failed");
        assert!(hwaccel.is_none());
        assert!(context.decoder().video().is_ok());

        // Test with 1 thread
        let result = setup_decoder_context(&stream, 1, None);
        assert!(result.is_ok());
        let (context, hwaccel) = result.expect("setup_decoder_context failed");
        assert!(hwaccel.is_none());
        assert!(context.decoder().video().is_ok());

        // Test with threads set to 0
        let result = setup_decoder_context(&stream, 0, None);
        assert!(result.is_ok());
        let (context, hwaccel) = result.expect("setup_decoder_context failed");
        assert!(hwaccel.is_none());
        assert!(context.decoder().video().is_ok());
    }
}
