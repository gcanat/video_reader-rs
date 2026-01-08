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

use crate::convert::{
    convert_nv12_to_ndarray_rgb24, convert_yuv_to_ndarray_rgb24, get_colorrange, get_colorspace,
};
use crate::decoder::{DecoderConfig, OutOfBoundsMode, VideoDecoder, VideoReducer};
use crate::filter::{create_filter_spec, create_filters, FilterConfig};
use crate::hwaccel::{HardwareAccelerationContext, HardwareAccelerationDeviceType};
use crate::info::{
    collect_video_metadata, extract_video_params, get_frame_count, get_resized_dim, StreamInfo,
};
use crate::utils::{insert_frame, FrameArray, VideoArray, HWACCEL_PIXEL_FORMAT};
use ndarray::{s, Array, Array4};
use tokio::task;

/// Custom errno used to signal backwards jump detection (non-monotonic output).
/// This allows us to distinguish it from real InvalidData errors.
const BACKWARDS_JUMP_ERRNO: i32 = 61; // ENODATA on most systems

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
    /// Current presentation index (display order count)
    curr_pres_idx: usize,
    n_fails: usize,
    decoder: VideoDecoder,
    draining: bool,
    /// Cached result of seek verification (None = not tested yet)
    seek_verified: Option<bool>,
    /// True if we've sent EOF and need to re-seek before processing more frames
    eof_sent: bool,
    /// True if we've started sequential iteration (packets iterator has been used)
    sequential_started: bool,
    /// How to handle out-of-bounds or failed frame fetches
    oob_mode: OutOfBoundsMode,
    /// Frame indices that failed (for error reporting)
    failed_indices: Vec<usize>,
}

impl VideoReader {
    pub fn decoder(&self) -> &VideoDecoder {
        &self.decoder
    }
    pub fn stream_info(&self) -> &StreamInfo {
        &self.stream_info
    }
    /// Get the last frame index that failed (for error reporting)
    pub fn failed_indices(&self) -> &Vec<usize> {
        &self.failed_indices
    }
    /// Create a new VideoReader instance
    /// * `filename` - Path to the video file.
    /// * `decoder_config` - Config for the decoder see: [`DecoderConfig`]
    /// * `oob_mode` - How to handle out-of-bounds or failed frame fetches
    ///
    /// Returns: a VideoReader instance.
    pub fn new(
        filename: String,
        decoder_config: DecoderConfig,
        oob_mode: OutOfBoundsMode,
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
            curr_pres_idx: 0,
            n_fails: 0,
            decoder,
            draining: false,
            seek_verified: None, // Will be tested on first get_batch
            eof_sent: false,
            sequential_started: false,
            oob_mode,
            failed_indices: Vec::new(),
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

        // Determine resize method with mutual exclusion
        // Priority: target_width/height > resize_shorter_side/longer_side > filter scale
        let has_target_dims = config.target_width().is_some() && config.target_height().is_some();
        let has_resize_side =
            config.resize_shorter_side().is_some() || config.resize_longer_side().is_some();
        let filter_has_scale = config
            .ff_filter_ref()
            .map(|f| f.to_lowercase().contains("scale"))
            .unwrap_or(false);

        // Check for mutual exclusion conflicts
        let resize_methods_count = [has_target_dims, has_resize_side, filter_has_scale]
            .iter()
            .filter(|&&x| x)
            .count();

        if resize_methods_count > 1 {
            log::error!(
                "Multiple resize methods specified. Use only ONE of: \
                 target_width/target_height, resize_shorter_side/resize_longer_side, or filter with scale."
            );
            return Err(ffmpeg::Error::InvalidData);
        }

        // Calculate dimensions using unified get_resized_dim function
        // It handles both target_width/target_height and resize_shorter/longer_side
        let (mut height, mut width) = if has_target_dims || has_resize_side {
            get_resized_dim(
                orig_h as f64,
                orig_w as f64,
                config.resize_shorter_side(),
                config.resize_longer_side(),
                config.target_width(),
                config.target_height(),
            )
        } else {
            // No resize specified - use storage dimensions, but swap for 90/270 rotation
            // to get display dimensions (what user actually sees)
            let is_90_270 = video_params.rotation.abs() == 90 || video_params.rotation.abs() == 270;
            if is_90_270 {
                (orig_w, orig_h) // Swap: storage WxH -> display HxW
            } else {
                (orig_h, orig_w)
            }
        };

        let is_hwaccel = hwaccel_context.is_some();

        // Get resize algorithm and check for user filter before config is consumed by ff_filter
        let has_user_filter = config.ff_filter_ref().is_some();
        let resize_algo = config.resize_algo();

        let (filter_spec, hw_format, out_width, out_height, rotation_applied) = create_filter_spec(
            width,
            height,
            &mut video,
            config.ff_filter(),
            hwaccel_context,
            HWACCEL_PIXEL_FORMAT,
            video_params.rotation,
            resize_algo,
        )?;

        // Use filter output dimensions (may differ due to rotation)
        width = out_width;
        height = out_height;

        debug!(
            "Filter spec: {}, output size: {}x{}",
            filter_spec, width, height
        );
        let time_base_rational = video_info
            .get("time_base_rational")
            .ok_or(ffmpeg::Error::InvalidData)?;

        // Get color space and range for filter graph (same as VideoDecoder)
        let cspace_string = video_info
            .get("color_space")
            .map(|s| s.as_str())
            .unwrap_or("BT709");
        let crange_string = video_info
            .get("color_range")
            .map(|s| s.as_str())
            .unwrap_or("");
        let color_space = get_colorspace(orig_h as i32, cspace_string);
        let color_range = get_colorrange(crange_string);

        let filter_cfg = FilterConfig::new(
            orig_h,
            orig_w,
            orig_fmt,
            time_base_rational,
            filter_spec.as_str(),
            is_hwaccel,
            pixel_aspect,
            color_space,
            color_range,
        );

        let graph = create_filters(&mut video, hw_format, filter_cfg)?;

        // Swap dimensions for 90/270 rotation when user provided a filter without scale
        // (in that case, filter doesn't handle the dimension swap)
        let skip_rotation_swap = filter_has_scale || !has_user_filter;
        if !skip_rotation_swap
            && rotation_applied
            && (video_params.rotation.abs() == 90 || video_params.rotation.abs() == 270)
        {
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
    /// Sequential decoding fallback: iterate from start and collect requested frames.
    /// Used when seek-based batch fails (e.g., pathological videos) or caller needs guaranteed accuracy.
    ///
    /// Supports skip-forward optimization: if min(indices) >= curr_pres_idx and decoder
    /// is not drained, we can continue from current position instead of seeking to start.
    ///
    /// Behavior depends on `oob_mode`:
    /// - `Error`: Return error if any requested frame is not found (default)
    /// - `Skip`: Skip missing frames - returned array may have fewer frames than requested
    /// - `Black`: Return black (all-zero) frames for missing frames
    pub fn get_batch_safe(&mut self, indices: Vec<usize>) -> Result<VideoArray, ffmpeg::Error> {
        self.failed_indices.clear();

        // For fast membership checks
        let needed: HashSet<usize> = indices.iter().cloned().collect();
        let needed_total = needed.len();
        let min_needed = indices.iter().min().copied().unwrap_or(0);
        let max_needed = indices.iter().max().copied().unwrap_or(0);
        let mut frame_map: HashMap<usize, FrameArray> = HashMap::with_capacity(needed.len());

        // Skip-forward optimization: if requested frame(s) are ahead of or at current position,
        // continue from current position instead of seeking to start
        // This makes sequential access O(n) instead of O(n²)
        // Requires: sequential_started (packets iterator has been used before)
        let can_skip_forward =
            self.sequential_started && !self.eof_sent && min_needed >= self.curr_pres_idx;

        let start_idx = if can_skip_forward {
            debug!(
                "get_batch_safe: skip-forward from {} to min_needed {}",
                self.curr_pres_idx, min_needed
            );
            self.curr_pres_idx
        } else {
            debug!(
                "get_batch_safe: seek to start (started={}, eof_sent={}, min_needed={}, curr={})",
                self.sequential_started, self.eof_sent, min_needed, self.curr_pres_idx
            );
            self.seek_to_start()?;
            0
        };

        // Mark that we've started sequential iteration
        self.sequential_started = true;

        let mut decoded = Video::empty();
        let mut curr_idx: usize = start_idx;
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

        // Track whether we exhausted the packet iterator or exited early
        // If we exited early (all needed frames collected), decoder state is still valid
        // If exhausted, we need to flush remaining frames from decoder buffer
        let exhausted_packets = collected < needed_total;

        if exhausted_packets {
            // Flush remaining frames from decoder buffer
            self.decoder.video.send_eof()?;
            self.eof_sent = true; // Mark that we sent EOF, next call must seek
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
        }

        let height = self.decoder.height as usize;
        let width = self.decoder.width as usize;

        // Build output based on oob_mode
        let frame_batch = match self.oob_mode {
            OutOfBoundsMode::Skip => {
                // Collect only found frames, maintaining request order
                let mut frames: Vec<FrameArray> = Vec::new();
                for idx in &indices {
                    if let Some(frame) = frame_map.get(idx) {
                        frames.push(frame.clone());
                    } else {
                        debug!("Skipping frame {} (oob_mode=skip)", idx);
                    }
                }

                if frames.is_empty() {
                    Array4::zeros((0, height, width, 3))
                } else {
                    let mut batch = Array4::zeros((frames.len(), height, width, 3));
                    for (i, frame) in frames.iter().enumerate() {
                        batch.slice_mut(s![i, .., .., ..]).assign(frame);
                    }
                    batch
                }
            }
            OutOfBoundsMode::Black => {
                // Return zeros for missing frames
                let mut batch = Array4::zeros((indices.len(), height, width, 3));
                for (out_i, idx) in indices.iter().enumerate() {
                    if let Some(frame) = frame_map.get(idx) {
                        batch.slice_mut(s![out_i, .., .., ..]).assign(frame);
                    } else {
                        debug!("Using black frame for {} (oob_mode=black)", idx);
                    }
                }
                batch
            }
            OutOfBoundsMode::Error => {
                // Collect all missing frames, then error
                let mut batch = Array4::zeros((indices.len(), height, width, 3));
                for (out_i, idx) in indices.iter().enumerate() {
                    if let Some(frame) = frame_map.get(idx) {
                        batch.slice_mut(s![out_i, .., .., ..]).assign(frame);
                    } else {
                        debug!("No frame found for {} (oob_mode=error)", idx);
                        self.failed_indices.push(*idx);
                    }
                }
                // Check if any failures occurred
                if !self.failed_indices.is_empty() {
                    self.decoder.video.flush();
                    self.seek_to_start()?;
                    return Err(ffmpeg::Error::Bug);
                }
                batch
            }
        };

        // Update position for skip-forward on subsequent calls
        // Don't flush decoder - keep reference frames for B-frame videos
        // Don't seek to start - allow continuing from current position
        self.curr_pres_idx = max_needed + 1;
        Ok(frame_batch)
    }

    /// Get the batch of frames from the video by seeking to the closest keyframe and skipping
    /// the frames until we reach the desired frame index. Heavily inspired by the implementation
    /// from decord library: https://github.com/dmlc/decord
    ///
    /// Sorts indices internally to minimize backward seeks.
    ///
    /// Behavior depends on `oob_mode`:
    /// - `Error`: Return error on any failed frame fetch (default)
    /// - `Skip`: Skip failed frames - returned array may have fewer frames than requested
    /// - `Black`: Return black (all-zero) frames for failed fetches
    pub fn get_batch(&mut self, indices: Vec<usize>) -> Result<VideoArray, ffmpeg::Error> {
        // Clear any stale failure state from previous calls
        self.failed_indices.clear();

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

        // NOTE: We intentionally do NOT call seek_to_start() here.
        // seek_accurate_raw() will handle seeking if needed - for sequential access
        // (like [0,1,2,3...]) it will skip forward without seeking, which is much faster.

        let cspace = self.decoder.color_space;
        let crange = self.decoder.color_range;
        let is_hwaccel = self.decoder.is_hwaccel;
        let height = self.decoder.height as usize;
        let width = self.decoder.width as usize;

        // For Skip mode, we need to collect successful frames and track their positions
        let mut successful_frames: Vec<(usize, ndarray::Array3<u8>)> = Vec::new();

        // For Black/Error mode, allocate output buffer once (pre-filled with zeros)
        let mut video_frames: Option<VideoArray> = if self.oob_mode != OutOfBoundsMode::Skip {
            Some(Array::zeros((indices.len(), height, width, 3)))
        } else {
            None
        };

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
                        match self.oob_mode {
                            OutOfBoundsMode::Skip => {
                                // Store the frame for each position
                                for pos in positions {
                                    successful_frames.push((*pos, rgb_frame.clone()));
                                }
                            }
                            OutOfBoundsMode::Black | OutOfBoundsMode::Error => {
                                // Write directly to pre-allocated buffer
                                if let Some(ref mut vf) = video_frames {
                                    for pos in positions {
                                        vf.slice_mut(s![*pos, .., .., ..]).assign(&rgb_frame);
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(None) => {
                    debug!("No frame found for frame index {}", frame_index);
                    match self.oob_mode {
                        OutOfBoundsMode::Error => {
                            self.failed_indices.push(frame_index);
                        }
                        OutOfBoundsMode::Skip => {
                            // Just skip - don't add to successful_frames
                            debug!("Skipping frame {} (oob_mode=skip)", frame_index);
                        }
                        OutOfBoundsMode::Black => {
                            // Already zeros, just log
                            debug!("Using black frame for {} (oob_mode=black)", frame_index);
                        }
                    }
                }
                Err(e) => {
                    // Backwards jump detected: non-monotonic decoder output
                    // Use custom errno to distinguish from real InvalidData errors
                    if let ffmpeg::Error::Other { errno } = e {
                        if errno == BACKWARDS_JUMP_ERRNO {
                            debug!(
                                "Backwards jump detected (errno={}). Switching to sequential mode.",
                                errno
                            );
                            self.seek_verified = Some(false);
                            return self.get_batch_safe(indices);
                        }
                    }

                    debug!("seek_accurate_raw failed at {}: {:?}", frame_index, e);
                    match self.oob_mode {
                        OutOfBoundsMode::Error => {
                            self.failed_indices.push(frame_index);
                        }
                        OutOfBoundsMode::Skip => {
                            debug!(
                                "Skipping frame {} due to error (oob_mode=skip)",
                                frame_index
                            );
                        }
                        OutOfBoundsMode::Black => {
                            debug!(
                                "Using black frame for {} due to error (oob_mode=black)",
                                frame_index
                            );
                        }
                    }
                }
            }
        }

        // Check if any failures occurred in Error mode
        if self.oob_mode == OutOfBoundsMode::Error && !self.failed_indices.is_empty() {
            // Reset decoder state before returning error
            self.decoder.video.flush();
            let _ = self.seek_to_start(); // Ignore error here to ensure cleanup
            return Err(ffmpeg::Error::Bug);
        }

        // Build final output
        if self.oob_mode == OutOfBoundsMode::Skip {
            // Sort by original position to maintain requested order
            successful_frames.sort_by_key(|(pos, _)| *pos);

            let n_successful = successful_frames.len();
            if n_successful == 0 {
                // Return empty array
                return Ok(Array::zeros((0, height, width, 3)));
            }

            let mut output = Array::zeros((n_successful, height, width, 3));
            for (i, (_, frame)) in successful_frames.into_iter().enumerate() {
                output.slice_mut(s![i, .., .., ..]).assign(&frame);
            }
            Ok(output)
        } else {
            Ok(video_frames.unwrap())
        }
    }

    /// Returns the raw YUV frame (after filter graph) for a given presentation index.
    /// Uses frame counting to handle B-frame reordering correctly.
    /// For Open GOP videos, automatically seeks to an earlier keyframe to ensure
    /// all reference frames are available.
    pub fn seek_accurate_raw(
        &mut self,
        presentation_idx: usize,
    ) -> Result<Option<Video>, ffmpeg::Error> {
        // Use the safe keyframe finder which handles Open GOP correctly
        let (key_decode_idx, key_pres_idx, min_pres_in_gop) = match self
            .stream_info
            .find_safe_keyframe_for_pres_idx(presentation_idx)
        {
            Some(info) => info,
            None => {
                debug!(
                    "No safe keyframe found for presentation index {}",
                    presentation_idx
                );
                return Ok(None);
            }
        };

        debug!(
            "    - [RAW] Presentation idx: {}, keyframe decode: {}, keyframe pres: {}, min_pres_in_gop: {}",
            presentation_idx, key_decode_idx, key_pres_idx, min_pres_in_gop
        );

        // Check if we can skip forward without seeking
        // We track curr_pres_idx (presentation order count)
        // IMPORTANT: If we've sent EOF, we MUST re-seek because the decoder state is invalid
        // For Open GOP, we also need to ensure we're past the B-frames that need previous GOP refs
        let can_skip_forward = !self.eof_sent
            && presentation_idx >= self.curr_pres_idx
            && self.curr_pres_idx >= min_pres_in_gop;

        if can_skip_forward {
            debug!(
                "No need to seek, we can directly skip frames (curr_pres: {})",
                self.curr_pres_idx
            );
            // Pass target presentation index directly - uses PTS matching internally
            match self.find_frame_by_pres_idx(presentation_idx) {
                Ok(frame) => Ok(frame),
                Err(ffmpeg::Error::Eof) => {
                    // Skip-forward failed (packet iterator exhausted).
                    // This can happen when packet order != frame output order (B-frames).
                    // Force a re-seek to the keyframe and try again.
                    debug!(
                        "Skip-forward failed, re-seeking to keyframe at decode idx: {}",
                        key_decode_idx
                    );
                    self.seek_to_start()?;
                    self.seek_frame_by_decode_idx(&key_decode_idx)?;
                    self.curr_frame = key_decode_idx;
                    self.curr_pres_idx = min_pres_in_gop.min(key_pres_idx);
                    self.eof_sent = false;

                    match self.find_frame_by_pres_idx(presentation_idx) {
                        Ok(frame) => Ok(frame),
                        Err(ffmpeg::Error::Eof) => {
                            self.get_frame_raw_after_eof_by_count(presentation_idx)
                        }
                        Err(e) => Err(e),
                    }
                }
                Err(e) => Err(e),
            }
        } else {
            debug!("Seeking to keyframe at decode idx: {}", key_decode_idx);

            self.seek_to_start()?;
            self.seek_frame_by_decode_idx(&key_decode_idx)?;
            self.curr_frame = key_decode_idx;
            // For Open GOP, the first decoded output may have pres_idx > or < key_pres_idx
            // We use min_pres_in_gop as a better estimate for where we'll start decoding from
            self.curr_pres_idx = min_pres_in_gop.min(key_pres_idx);

            // Pass target presentation index directly - uses PTS matching internally
            match self.find_frame_by_pres_idx(presentation_idx) {
                Ok(frame) => Ok(frame),
                Err(ffmpeg::Error::Eof) => self.get_frame_raw_after_eof_by_count(presentation_idx),
                Err(e) => Err(e),
            }
        }
    }

    /// Find and return the raw YUV frame at the target presentation index.
    /// Uses PTS-based matching which correctly handles Open GOP structures
    /// where keyframe's presentation index may be greater than target.
    pub fn find_frame_by_pres_idx(
        &mut self,
        target_pres_idx: usize,
    ) -> Result<Option<Video>, ffmpeg::Error> {
        debug!(
            "Finding frame at pres_idx={}, curr_pres_idx={}",
            target_pres_idx, self.curr_pres_idx
        );
        let mut failsafe = (self.stream_info.frame_count() * 2) as i32;
        let mut prev_map_idx: Option<usize> = None;

        // First, try to get frame from decoder's existing buffer (from previous packets)
        let (yuv_frame, counter) =
            self.get_frame_raw_by_count(target_pres_idx, &mut prev_map_idx)?;
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
                        let (yuv_frame, counter) =
                            self.get_frame_raw_by_count(target_pres_idx, &mut prev_map_idx)?;
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

    /// Get raw YUV frame (after filter graph) at target presentation index.
    /// Counts decoded frames and returns when reaching the target.
    ///
    /// # Performance Optimization: Check-Before-Filter
    ///
    /// This function implements a "check-before-filter" strategy where we first check
    /// if the decoded frame matches the target, and ONLY THEN push it through the filter
    /// graph. This avoids wasting expensive filter operations (e.g., scale) on frames
    /// we don't need.
    ///
    /// Example: To get frame 105 when the nearest keyframe is at 100, we must decode
    /// frames 100-105. With the naive approach, we'd apply the scale filter 6 times.
    /// With this optimization, we only apply it once (to frame 105).
    ///
    /// # Filter Compatibility Note (TODO: (wizyoung) future work)
    ///
    /// This optimization is SAFE for **spatial filters** that process each frame
    /// independently (e.g., `scale`, `format`, `transpose`, `crop`).
    ///
    /// It may NOT work correctly for **temporal filters** that require consecutive
    /// frame input (e.g., `fps`, `minterpolate`, `tinterlace`, `deflicker`).
    /// If temporal filter support is needed, consider detecting such filters and
    /// falling back to the "filter-all-frames" approach.
    pub(crate) fn get_frame_raw_by_count(
        &mut self,
        target_pres_idx: usize,
        prev_map_idx: &mut Option<usize>,
    ) -> Result<(Option<Video>, i32), ffmpeg::Error> {
        let mut decoded = Video::empty();
        let mut counter = 0;
        let expected_pts = self.stream_info.get_pts_for_presentation(target_pres_idx);

        while self.decoder.video.receive_frame(&mut decoded).is_ok() {
            // Get PTS from decoded frame BEFORE filter processing.
            // This is key to the check-before-filter optimization.
            let decoded_pts = decoded.pts();

            // Check for PTS Backwards Jump (Non-Monotonic Output)
            // This detects videos where the output order is NOT reordered (Raw Decode Order),
            // which breaks our Sorted Map assumption.
            if let Some(p) = decoded_pts {
                let pts_offset = self.stream_info.min_pts_offset();
                let norm_p = p - pts_offset;
                if let Some(mapped) = self.stream_info.presentation_for_pts_norm(norm_p) {
                    // Runtime backwards jump detection: if decoder outputs frames with
                    // non-monotonic presentation indices, our PTS->pres_idx map is invalid.
                    // This catches pathological videos where packet PTS looks normal
                    // but decoder output order doesn't match expected order.
                    if let Some(prev) = *prev_map_idx {
                        if mapped < prev {
                            debug!(
                                "Non-monotonic output detected: prev_map={} curr_map={} pts={}",
                                prev, mapped, p
                            );
                            return Err(ffmpeg::Error::Other {
                                errno: BACKWARDS_JUMP_ERRNO,
                            });
                        }
                    }
                    *prev_map_idx = Some(mapped);

                    // Standard Resync logic
                    self.curr_pres_idx = mapped;
                }
            }

            // Check for match BEFORE filter processing to avoid wasted computation.
            // For spatial filters like scale/format/transpose, skipping non-target
            // frames saves significant CPU time.
            let mut matched = false;

            // PTS-based match (primary method - most reliable after seeks)
            // expected_pts returns (raw_pts, normalized_pts)
            // decoded_pts is raw from decoder, so compare with exp_raw OR normalize and compare with exp_norm
            if let (Some((exp_raw, exp_norm)), Some(p)) = (expected_pts, decoded_pts) {
                let pts_offset = self.stream_info.min_pts_offset();
                let norm_p = p - pts_offset;
                if p == exp_raw || norm_p == exp_norm {
                    matched = true;
                }
            }

            // NOTE: We intentionally do NOT use curr_pres_idx as a fallback match.
            // For some videos, packet order != frame output order, causing curr_pres_idx
            // (derived from presentation_for_pts_norm) to be out of sync with the actual
            // decoded frame. PTS-based matching is the only reliable method.

            // OPTIMIZATION: Only apply filter to the target frame.
            // This is the core of the check-before-filter optimization.
            if matched {
                // Push through filter graph and get YUV frame
                if let Some(mut in_ctx) = self.decoder.graph.get("in") {
                    if let Err(e) = in_ctx.source().add(&decoded) {
                        debug!("Failed to push frame into filter graph: {e}");
                        return Ok((None, counter));
                    }
                } else {
                    debug!("Filter graph missing 'in' pad");
                    return Ok((None, counter));
                }

                let mut yuv_frame = Video::empty();
                // For simple scale filters, output is immediate (no delay)
                if let Some(mut out_ctx) = self.decoder.graph.get("out") {
                    if out_ctx.sink().frame(&mut yuv_frame).is_ok() {
                        let out_pts_dbg = decoded_pts.unwrap_or(-1);
                        debug!(
                            "Found target frame at pres_idx={}, pts={}",
                            target_pres_idx, out_pts_dbg
                        );
                        self.curr_pres_idx += 1;
                        return Ok((Some(yuv_frame), counter + 1));
                    }
                } else {
                    debug!("Filter graph missing 'out' pad");
                    return Ok((None, counter));
                }
            }

            // Advance presentation index for next iteration
            self.curr_pres_idx += 1;
            counter += 1;
        }

        Ok((None, counter))
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
        let mut prev_map_idx: Option<usize> = None;
        let (yuv_frame, _counter) =
            self.get_frame_raw_by_count(target_pres_idx, &mut prev_map_idx)?;
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
    /// Returns (force_sequential, summary)
    fn quick_seek_static_check(&self) -> (bool, String) {
        let mut force_sequential = false;
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
        if self.stream_info.has_duplicate_pts() {
            force_sequential = true;
            reasons.push("duplicate pts".to_string());
        }
        if self.stream_info.has_duplicate_dts() {
            force_sequential = true;
            reasons.push("duplicate dts".to_string());
        }
        // Detect videos with undecodable frames at the start
        // Case 1: First keyframe's presentation index > 0 means frames before it cannot be decoded
        // Case 2: First keyframe has negative PTS typically indicates B-frames at video start
        //         that depend on future reference frames and cannot be decoded properly
        let first_kf_offset = self.stream_info.first_keyframe_pres_offset();
        let first_kf_neg_pts = self.stream_info.first_keyframe_has_negative_pts();
        if first_kf_offset > 0 || first_kf_neg_pts {
            force_sequential = true;
            reasons.push(format!(
                "undecodable frames at start (pres_offset={}, neg_pts={})",
                first_kf_offset, first_kf_neg_pts
            ));
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

        let summary = if reasons.is_empty() {
            "ok".to_string()
        } else {
            reasons.join("; ")
        };
        (force_sequential, summary)
    }

    /// Check if this video needs sequential mode, with cache design
    pub fn needs_sequential_mode(&mut self) -> bool {
        if let Some(seek_works) = self.seek_verified {
            return !seek_works;
        }

        // do a runtime seek verification (only once)
        let quick_start = Instant::now();
        let (force_seq, quick_summary) = self.quick_seek_static_check();
        let quick_elapsed = quick_start.elapsed();
        if force_seq {
            self.seek_verified = Some(false);
            debug!(
                "needs_sequential_mode: quick static check -> sequential (reason: {}) ({:?})",
                quick_summary, quick_elapsed
            );
            return true;
        }

        self.seek_verified = Some(true);
        debug!(
            "needs_sequential_mode: quick static check ok (reason: {}) ({:?})",
            quick_summary, quick_elapsed
        );
        false
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
        let mut unique_sorted: Vec<usize> = indices.to_vec();
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
    ///
    /// # Cost Model
    /// The decision is based on several factors:
    /// 1. **Codec complexity**: AV1/HEVC have higher seek overhead than H.264
    /// 2. **Resolution**: Higher resolution = higher per-frame decode cost
    /// 3. **GOP density**: Videos with many keyframes (small GOPs) have higher seek overhead
    /// 4. **Edge cases**: When costs are close, prefer sequential (more predictable)
    pub fn should_use_sequential(&self, indices: &[usize]) -> bool {
        // Note: negative PTS/DTS check is now done at a higher level via needs_sequential_mode()
        // which actually verifies if seek works, rather than just checking metadata

        let (seek_frames, seek_count, sequential_frames, unique_count, _max_index) =
            self.estimate_decode_cost_detailed(indices);

        if unique_count == 0 {
            return true;
        }

        // Get codec and resolution info for dynamic overhead calculation
        let codec_id = self
            .decoder
            .video_info
            .get("codec_id")
            .map(|s| s.to_uppercase())
            .unwrap_or_default();
        let height = self.decoder.height;

        // Dynamic seek overhead based on codec complexity
        // AV1 and HEVC have more complex reference structures and higher decode cost per frame
        // Higher resolution also increases the relative cost of decoder reset
        let base_seek_overhead: usize = if codec_id.contains("AV1") {
            25 // AV1: Most complex, high decode cost
        } else if codec_id.contains("HEVC") || codec_id.contains("H265") {
            20 // HEVC: Complex reference structures
        } else if codec_id.contains("H264") || codec_id.contains("AVC") {
            if height > 1080 {
                15 // 4K H.264: Higher per-frame cost
            } else {
                10 // Standard H.264
            }
        } else {
            10 // Default for other codecs
        };

        // GOP density factor: videos with many keyframes have higher seek overhead
        // because each seek requires decoder reset even when keyframes are close together
        let key_frames = self.stream_info.key_frames();
        let frame_count = *self.stream_info.frame_count();
        let avg_gop_size = if !key_frames.is_empty() && frame_count > 0 {
            frame_count / key_frames.len()
        } else {
            30 // Default GOP size assumption
        };

        // Apply penalty for dense I-frame videos (avg_gop < 10)
        // In such videos, seek overhead dominates because decoder resets are frequent
        let gop_penalty_factor: f64 = if avg_gop_size < 5 {
            2.0 // Very dense keyframes: double the overhead
        } else if avg_gop_size < 10 {
            1.5 // Dense keyframes: 50% more overhead
        } else {
            1.0 // Normal GOP structure
        };

        let seek_overhead = (base_seek_overhead as f64 * gop_penalty_factor).round() as usize;

        // Cost model after benchmarking:
        // - When seek_frames ≈ sequential_frames, sequential wins (simpler, cache-friendly)
        // - Seek only wins when it can skip significant portions of the video
        // - Many GOP transitions (seek_count) add overhead even with skip-forward

        debug!(
            "Cost estimation: seek_frames={}, seek_count={}, sequential={}, codec={}, avg_gop={}, overhead={}",
            seek_frames, seek_count, sequential_frames, codec_id, avg_gop_size, seek_overhead
        );

        let seek_total_cost = seek_frames + seek_count * seek_overhead;

        debug!(
            "Cost estimation: seek_frames={}, seek_count={}, seek_overhead={}, seek_total={}, sequential={}",
            seek_frames, seek_count, seek_count * seek_overhead, seek_total_cost, sequential_frames
        );

        // Rule 0: Edge case handling - when costs are close, prefer sequential
        // Sequential is more predictable and avoids worst-case seek penalties
        // This catches cases where benchmark variability could cause wrong predictions
        let cost_diff = seek_total_cost.abs_diff(sequential_frames);
        let margin_threshold = (sequential_frames as f64 * 0.15).max(5.0) as usize;
        if cost_diff < margin_threshold {
            debug!(
                "Edge case: cost_diff={} < margin={}, using sequential for stability",
                cost_diff, margin_threshold
            );
            return true; // Prefer sequential when costs are close
        }

        // Rule 1: If total seek cost (including overhead) >= sequential, use sequential
        if seek_total_cost >= sequential_frames {
            return true; // Seek not worth it with overhead
        }

        // Rule 2: If seek_frames alone >= 85% of sequential, use sequential (not saving enough)
        if seek_frames as f64 >= sequential_frames as f64 * 0.85 {
            return true; // Not saving enough, use sequential
        }

        // Rule 3: If many GOP transitions (>5), use dynamic threshold based on seek efficiency
        // avg_frames_per_seek indicates how efficient each seek is
        // Also check absolute net savings as escape hatch for long videos
        if seek_count > 5 {
            let avg_frames_per_seek = seek_frames / seek_count;

            // Dynamic threshold: efficient seeks get more lenient rules
            let rule3_threshold = if avg_frames_per_seek < 50 {
                0.80 // Very efficient seeks - each seek lands close to target
            } else if avg_frames_per_seek < 150 {
                0.70 // Moderate efficiency - standard threshold
            } else {
                0.55 // Inefficient seeks - stricter threshold
            };

            if seek_frames as f64 >= sequential_frames as f64 * rule3_threshold {
                // Check absolute net savings as escape hatch for long videos
                // Even if ratio is high, if absolute savings is significant, use SEEK
                let net_savings = sequential_frames
                    .saturating_sub(seek_frames)
                    .saturating_sub(seek_count * seek_overhead);
                let min_savings_threshold = (sequential_frames as f64 * 0.15) as usize;

                if net_savings < min_savings_threshold {
                    debug!(
                        "Rule 3: seek_count={}, avg_per_seek={}, ratio={:.2}%, net_savings={} < min={}, using SEQ",
                        seek_count, avg_frames_per_seek,
                        seek_frames as f64 / sequential_frames as f64 * 100.0,
                        net_savings, min_savings_threshold
                    );
                    return true; // Not enough savings, use sequential
                }
                // Otherwise, allow SEEK despite high seek_count (long video optimization)
                debug!(
                    "Rule 3 escaped: net_savings={} >= min={}, allowing SEEK for long video",
                    net_savings, min_savings_threshold
                );
            }
        }

        // Rule 4: For complex codecs (AV1/HEVC), require more savings to justify seek
        // These codecs have higher worst-case seek penalties
        if (codec_id.contains("AV1") || codec_id.contains("HEVC") || codec_id.contains("H265"))
            && seek_frames as f64 >= sequential_frames as f64 * 0.6
        {
            return true; // Complex codec: need more savings to justify seek
        }

        false // Use seek - significant savings
    }

    /// Seek back to the begining of the stream
    fn seek_to_start(&mut self) -> Result<(), ffmpeg::Error> {
        self.ictx.seek(0, ..100)?;
        self.avflushbuf()?;
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
                        Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::error::EAGAIN => {
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
