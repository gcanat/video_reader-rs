use ffmpeg::codec::threading;
use ffmpeg::ffi::*;
use ffmpeg::format::input;
use ffmpeg::media::Type;
use ffmpeg::util::frame::video::Video;
use ffmpeg_next as ffmpeg;
use log::debug;
use std::collections::HashMap;
use std::path::Path;

use crate::convert::{convert_nv12_to_ndarray_rgb24, convert_yuv_to_ndarray_rgb24};
use crate::decoder::{DecoderConfig, VideoDecoder, VideoReducer};
use crate::filter::{create_filter_spec, create_filters, FilterConfig};
use crate::hwaccel::{HardwareAccelerationContext, HardwareAccelerationDeviceType};
use crate::info::{
    collect_video_metadata, extract_video_params, get_frame_count, get_resized_dim, StreamInfo,
};
use crate::utils::{insert_frame, FrameArray, VideoArray, HWACCEL_PIXEL_FORMAT};
use ndarray::{s, Array, Array4, ArrayViewMut3};
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
    n_fails: usize,
    decoder: VideoDecoder,
    draining: bool,
}

unsafe impl Send for VideoReader {}

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
            n_fails: 0,
            decoder,
            draining: false,
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

        let (mut height, mut width) = get_resized_dim(
            orig_h as f64,
            orig_w as f64,
            config.resize_shorter_side(),
            config.resize_longer_side(),
        );

        let is_hwaccel = hwaccel_context.is_some();
        let (filter_spec, hw_format) = create_filter_spec(
            width,
            height,
            &mut video,
            config.ff_filter(),
            hwaccel_context,
            HWACCEL_PIXEL_FORMAT,
            video_params.rotation,
        )?;

        debug!("Filter spec: {}", filter_spec);
        let filter_cfg = FilterConfig::new(
            orig_h,
            orig_w,
            orig_fmt,
            video_info.get("time_base_rational").unwrap(),
            filter_spec.as_str(),
            is_hwaccel,
        );

        let graph = create_filters(&mut video, hw_format, filter_cfg)?;

        if video_params.rotation.abs() == 90 {
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

        let mut reducer = reducer.unwrap();
        reducer.set_frame_index(self.curr_frame);
        let max_idx = reducer.get_indices().iter().max().unwrap_or(&0).to_owned();
        Ok((reducer, max_idx))
    }

    pub fn decode_next(&mut self) -> Result<FrameArray, ffmpeg::Error> {
        loop {
            match self.ictx.packets().next() {
                Some((stream, packet)) => {
                    if stream.index() == self.stream_index {
                        self.decoder.video.send_packet(&packet)?;
                        match self.decoder.decode_frames()? {
                            Some(rgb_frame) => break Ok(rgb_frame),
                            None => continue,
                        };
                    }
                }
                None => {
                    if !self.draining {
                        self.decoder.video.send_eof()?;
                        self.draining = true;
                    }
                    match self.decoder.decode_frames()? {
                        Some(rgb_frame) => break Ok(rgb_frame),
                        None => {
                            self.draining = false;
                            self.decoder.video.flush();
                            self.seek_to_start()?;
                            break Err(ffmpeg::Error::Eof);
                        }
                    }
                }
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

        let mut reducer = reducer.unwrap();
        reducer.set_frame_index(self.curr_frame);
        let mut tasks = vec![];

        let mut receive_and_process_decoded_frames = |decoder: &mut ffmpeg::decoder::Video,
                                                      mut curr_frame: usize|
         -> Result<usize, ffmpeg::Error> {
            let mut decoded = Video::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                if reducer.get_indices().iter().any(|x| x == &curr_frame) {
                    self.decoder
                        .graph
                        .get("in")
                        .unwrap()
                        .source()
                        .add(&decoded)
                        .unwrap();
                    let mut rgb_frame = Video::empty();
                    if self
                        .decoder
                        .graph
                        .get("out")
                        .unwrap()
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
            outputs.push(task_.await.unwrap());
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
        let first_index = indices.iter().min().unwrap_or(&0);
        let max_index = self.stream_info.frame_count() - 1;
        let last_index = indices.iter().max().unwrap_or(&max_index);
        let (reducer, _, _) = VideoReducer::build(
            Some(*first_index),
            Some(*last_index),
            *self.stream_info.frame_count(),
            None,
            self.decoder.height,
            self.decoder.width,
        );

        // make sure we are at the begining of the stream
        self.seek_to_start()?;

        let mut reducer = reducer.unwrap();
        // check if closest key frames to first_index is non zero, if so we can seek
        let key_pos = self.locate_keyframes(first_index);
        if key_pos > 0 {
            self.seek_frame(&key_pos)?;
            reducer.set_frame_index(self.curr_frame);
        }
        let mut frame_map: HashMap<usize, FrameArray> = HashMap::new();

        for (stream, packet) in self.ictx.packets() {
            if stream.index() == self.stream_index {
                self.decoder.video.send_packet(&packet)?;
                self.decoder
                    .skip_and_decode_frames(&mut reducer, &indices, &mut frame_map)?;
            } else {
                debug!("Packet for another stream");
            }
            if &reducer.get_frame_index() > last_index {
                break;
            }
        }
        self.decoder.video.send_eof()?;
        if &reducer.get_frame_index() <= last_index {
            self.decoder
                .skip_and_decode_frames(&mut reducer, &indices, &mut frame_map)?;
        }

        let mut frame_batch: VideoArray = Array4::zeros((
            indices.len(),
            self.decoder.height as usize,
            self.decoder.width as usize,
            3,
        ));
        let _ = indices
            .iter()
            .enumerate()
            .map(|(i, idx)| match frame_map.get(idx) {
                Some(frame) => frame_batch.slice_mut(s![i, .., .., ..]).assign(frame),
                None => debug!("No frame found for {}", idx),
            })
            .collect::<Vec<_>>();

        Ok(frame_batch)
    }

    /// Get the batch of frames from the video by seeking to the closest keyframe and skipping
    /// the frames until we reach the desired frame index. Heavily inspired by the implementation
    /// from decord library: https://github.com/dmlc/decord
    pub fn get_batch(&mut self, indices: Vec<usize>) -> Result<VideoArray, ffmpeg::Error> {
        let mut video_frames: VideoArray = Array::zeros((
            indices.len(),
            self.decoder.height as usize,
            self.decoder.width as usize,
            3,
        ));

        // make sure we are at the begining of the stream
        self.seek_to_start()?;

        for (idx_counter, frame_index) in indices.into_iter().enumerate() {
            self.n_fails = 0;
            debug!("[NEXT INDICE] frame_index: {frame_index}");
            self.seek_accurate(
                frame_index,
                &mut video_frames.slice_mut(s![idx_counter, .., .., ..]),
            )?;
        }
        Ok(video_frames)
    }

    pub fn seek_accurate(
        &mut self,
        frame_index: usize,
        frame_array: &mut ArrayViewMut3<u8>,
    ) -> Result<(), ffmpeg::Error> {
        let key_pos = self.locate_keyframes(&frame_index);
        debug!("    - Key pos: {}", key_pos);
        let curr_key_pos = self.locate_keyframes(&self.curr_frame);
        debug!("    - Curr key pos: {}", curr_key_pos);
        if (key_pos == curr_key_pos) & (frame_index >= self.curr_frame) {
            // we can directly skip until frame_index
            debug!("No need to seek, we can directly skip frames");
            let num_skip = self.get_num_skip(&frame_index);
            match self.skip_frames(num_skip, &frame_index, frame_array) {
                Ok(()) => Ok(()),
                Err(_) => self.get_frame_after_eof(frame_array, &frame_index),
            }
        } else {
            if key_pos < curr_key_pos {
                debug!("Seeking back to start");
                self.seek_to_start()?;
            }
            debug!("Seeking to key_pos: {}", key_pos);
            self.seek_frame(&key_pos)?;
            let num_skip = self.get_num_skip(&frame_index);
            match self.skip_frames(num_skip, &frame_index, frame_array) {
                Ok(()) => Ok(()),
                Err(_) => self.get_frame_after_eof(frame_array, &frame_index),
            }
        }
    }

    /// Find the closest key frame before `pos`
    pub fn locate_keyframes(&self, pos: &usize) -> usize {
        let key_pos = self
            .stream_info
            .key_frames()
            .iter()
            .filter(|e| &pos >= e)
            .max()
            .unwrap_or(&0);
        key_pos.to_owned()
    }

    /// How many frames we need to skip to go from current decoding index `curr_dec_idx`
    /// to `target_dec_index`. The `frame_index` argument corresponds to the presentation
    /// index, while we need to know the number of frames to skip in terms of decoding index.
    pub fn get_num_skip(&self, frame_index: &usize) -> usize {
        frame_index.saturating_sub(self.curr_dec_idx)
    }

    /// Seek back to the begining of the stream
    fn seek_to_start(&mut self) -> Result<(), ffmpeg::Error> {
        self.ictx.seek(0, ..100)?;
        self.avflushbuf()?;
        self.curr_dec_idx = 0;
        self.curr_frame = 0;
        Ok(())
    }

    pub fn skip_frames(
        &mut self,
        num: usize,
        frame_index: &usize,
        frame_array: &mut ArrayViewMut3<u8>,
    ) -> Result<(), ffmpeg::Error> {
        let num_skip = num.min(self.stream_info.frame_count() - 1);
        debug!(
            "will skip {} frames, from current frame:{}",
            num_skip, self.curr_frame
        );
        // dont retry more than 2x the number of frames we are supposed to skip
        // just to make sure we get out of the loop
        let mut failsafe = (num_skip * 2) as i32;

        while failsafe > -1 {
            match self.ictx.packets().next() {
                Some((stream, packet)) => {
                    if stream.index() == self.stream_index {
                        self.decoder.video.send_packet(&packet)?;
                        let (rgb_frame, counter) = self.get_frame(frame_index);
                        if let Some(frame) = rgb_frame {
                            insert_frame(frame_array, frame);
                            return Ok(());
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
        debug!(
            "Finished skipping, current frame is now: {}",
            self.curr_frame
        );
        Err(ffmpeg::Error::Eof)
    }

    /// Get frame at `frame_index` when there is no more packets to iterate
    pub fn get_frame_after_eof(
        &mut self,
        frame_array: &mut ArrayViewMut3<u8>,
        frame_index: &usize,
    ) -> Result<(), ffmpeg::Error> {
        self.decoder.video.send_eof()?;
        let (rgb_frame, _counter) = self.get_frame(frame_index);
        if let Some(frame) = rgb_frame {
            insert_frame(frame_array, frame);
        }
        Ok(())
    }

    /// Get a single frame at `frame_index`
    pub fn get_frame(&mut self, frame_index: &usize) -> (Option<FrameArray>, i32) {
        let mut decoded = Video::empty();
        let mut counter = 0;
        let mut rgb_frame: Option<FrameArray> = None;
        while self.decoder.video.receive_frame(&mut decoded).is_ok() {
            if &self.curr_frame == frame_index {
                debug!("Decoding frame {}", frame_index);
                rgb_frame = self.decoder.process_frame(&decoded);
            }
            self.update_indices();
            counter += 1;
        }
        (rgb_frame, counter)
    }

    /// Update the current frame index and decoding index
    pub fn update_indices(&mut self) {
        self.curr_dec_idx += 1;
        self.curr_frame = self.curr_dec_idx;
        debug!(
            "dec_idx: {}, curr_frame: {}",
            self.curr_dec_idx, self.curr_frame
        );
    }

    // AVSEEK_FLAG_BACKWARD 1 <- seek backward
    // AVSEEK_FLAG_BYTE 2 <- seeking based on position in bytes
    // AVSEEK_FLAG_ANY 4 <- seek to any frame, even non-key frames
    // AVSEEK_FLAG_FRAME 8 <- seeking based on frame number
    pub fn seek_frame(&mut self, pos: &usize) -> Result<(), ffmpeg::Error> {
        // Prefer DTS when available for monotonicity; fallback to PTS.
        // If no timestamp is available for this decode index, fall back to frame-number seeking.
        if let Some(fr_ts) = self.stream_info.frame_times().get(pos) {
            let dts = *fr_ts.dts();
            let ts = if dts > 0 { dts } else { *fr_ts.pts() };
            match self.avseekframe(pos, ts, AVSEEK_FLAG_BACKWARD) {
                Ok(()) => Ok(()),
                Err(_) => self.avseekframe(pos, *pos as i64, AVSEEK_FLAG_FRAME),
            }
        } else {
            self.avseekframe(pos, *pos as i64, AVSEEK_FLAG_FRAME)
        }
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
        unsafe { avcodec_flush_buffers(self.decoder.video.as_mut_ptr()) };
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

        let (ctx, stream_index) = result.unwrap();
        assert!(stream_index < ctx.streams().count());

        let stream = ctx.streams().find(|s| s.index() == stream_index);
        assert!(stream.is_some());
        let stream = stream.unwrap();
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
        ffmpeg::init().unwrap();
        let ictx = input(&path).unwrap();
        let stream = ictx
            .streams()
            .best(Type::Video)
            .expect("No video stream found");

        // test with multiple threads
        let result = setup_decoder_context(&stream, 4, None);
        assert!(result.is_ok());
        let (context, hwaccel) = result.unwrap();
        assert!(hwaccel.is_none());
        assert!(context.decoder().video().is_ok());

        // Test with 1 thread
        let result = setup_decoder_context(&stream, 1, None);
        assert!(result.is_ok());
        let (context, hwaccel) = result.unwrap();
        assert!(hwaccel.is_none());
        assert!(context.decoder().video().is_ok());

        // Test with threads set to 0
        let result = setup_decoder_context(&stream, 0, None);
        assert!(result.is_ok());
        let (context, hwaccel) = result.unwrap();
        assert!(hwaccel.is_none());
        assert!(context.decoder().video().is_ok());
    }
}
