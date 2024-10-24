use ffmpeg::codec::threading;
use ffmpeg::ffi::*;
use ffmpeg::format::{input, Pixel as AvPixel};
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{context::Context, flag::Flags};
use ffmpeg::util::frame::video::Video;
use ffmpeg_next as ffmpeg;
use log::debug;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use video_rs::encode::{Encoder, Settings};
use video_rs::location::Location;
use video_rs::time::Time;

use ndarray::{s, Array, Array3, Array4, ArrayViewMut3, Axis};

pub type FrameArray = Array3<u8>;
pub type VideoArray = Array4<u8>;

/// Struct responsible for reading the stream and getting the metadata
pub struct VideoReader {
    pub ictx: ffmpeg::format::context::Input,
    pub stream_index: usize,
    pub stream_info: StreamInfo,
    pub curr_frame: usize,
    pub curr_dec_idx: usize,
    pub n_fails: usize,
    pub decoder: VideoDecoder,
    pub reducer: Option<VideoReducer>,
    pub first_frame: Option<usize>,
    pub last_frame: Option<usize>,
}

/// Struct responsible for doing the actual decoding
pub struct VideoDecoder {
    pub video: ffmpeg::decoder::Video,
    pub scaler: Context,
    pub height: u32,
    pub width: u32,
    pub fps: f64,
    pub video_info: HashMap<&'static str, String>,
    // pub start_time: i64,
    // pub time_base: f64,
}

/// Struct used when we want to decode the whole video with a compression_factor
pub struct VideoReducer {
    pub indices: Vec<usize>,
    pub frame_index: usize,
    pub idx_counter: usize,
    pub full_video: VideoArray,
}

/// Timing info for key frames
#[derive(Debug)]
pub struct FrameTime {
    pub pts: i64,
    pub dur: i64,
    pub dts: i64,
    pub didx: usize,
}

/// Info gathered from iterating over the video stream
pub struct StreamInfo {
    pub frame_count: usize,
    pub key_frames: Vec<usize>,
    pub frame_times: BTreeMap<i64, FrameTime>,
    pub decode_order: HashMap<usize, usize>,
}

impl VideoDecoder {
    /// Decode all frames that match the frame indices
    pub fn receive_and_process_decoded_frames(
        &mut self,
        reducer: &mut VideoReducer,
    ) -> Result<(), ffmpeg::Error> {
        let mut decoded = Video::empty();
        while self.video.receive_frame(&mut decoded).is_ok() {
            let match_index = reducer
                .indices
                .iter()
                .position(|x| x == &reducer.frame_index);
            if match_index.is_some() {
                reducer.indices.remove(match_index.unwrap());
                let mut rgb_frame = Video::empty();
                self.scaler.run(&decoded, &mut rgb_frame)?;
                let res = convert_frame_to_ndarray_rgb24(
                    &mut rgb_frame,
                    &mut reducer
                        .full_video
                        .slice_mut(s![reducer.idx_counter, .., .., ..]),
                );
                match res {
                    Ok(()) => {}
                    Err(_) => println!("Couldnt decode frame {}", reducer.frame_index),
                }
                reducer.idx_counter += 1;
            }
            reducer.frame_index += 1;
        }
        Ok(())
    }
    /// Decode frames
    pub fn skip_and_decode_frames(
        &mut self,
        reducer: &mut VideoReducer,
        indices: &[usize],
        frame_map: &mut HashMap<usize, FrameArray>,
    ) -> Result<(), ffmpeg::Error> {
        let mut decoded = Video::empty();
        while self.video.receive_frame(&mut decoded).is_ok() {
            if indices.iter().any(|x| x == &reducer.frame_index) {
                let mut rgb_frame = Video::empty();
                let mut nd_frame =
                    FrameArray::zeros((self.height as usize, self.width as usize, 3_usize));
                self.scaler.run(&decoded, &mut rgb_frame)?;
                convert_frame_to_ndarray_rgb24(&mut rgb_frame, &mut nd_frame.view_mut())?;
                frame_map.insert(reducer.frame_index, nd_frame);
            }
            reducer.frame_index += 1;
        }
        Ok(())
    }
}

impl VideoReader {
    /// Create a new VideoReader instance
    /// * `filename` - Path to the video file.
    /// * `compression_factor` - Factor to reduce the number of frames in the video.
    /// * `resize_shorter_side` - Resize the shorter side of the video to this value.
    /// * `threads` - Number of threads to use for decoding. This will be ignored when using
    ///   the `get_batch` method, as it does not work with multithreading at the moment.
    /// * `with_reducer` - Whether to use the VideoReducer to reduce the number of frames.
    ///    should be set to true to be able to use `decode_video` method. Set to false when using
    ///    the `get_batch` method.
    ///
    /// Returns: a VideoReader instance.
    pub fn new(
        filename: String,
        compression_factor: Option<f64>,
        resize_shorter_side: Option<f64>,
        threads: usize,
        with_reducer: bool,
        start_frame: Option<&usize>,
        end_frame: Option<&usize>,
    ) -> Result<VideoReader, ffmpeg::Error> {
        let (mut ictx, stream_index) = get_init_context(&filename)?;
        let stream_info = get_frame_count(&mut ictx, &stream_index)?;
        let (decoder, reducer, first_frame, last_frame) = Self::get_decoder(
            &ictx,
            threads,
            resize_shorter_side,
            compression_factor,
            stream_info.frame_count,
            with_reducer,
            start_frame,
            end_frame,
        )?;
        debug!("frame_count: {}", stream_info.frame_count);
        debug!("key frames: {:?}", stream_info.key_frames);
        Ok(VideoReader {
            ictx,
            stream_index,
            stream_info,
            curr_frame: 0,
            curr_dec_idx: 0,
            n_fails: 0,
            decoder,
            reducer,
            first_frame,
            last_frame,
        })
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn get_decoder(
        ictx: &ffmpeg::format::context::Input,
        threads: usize,
        resize_shorter_side: Option<f64>,
        compression_factor: Option<f64>,
        frame_count: usize,
        with_reducer: bool,
        start_frame: Option<&usize>,
        end_frame: Option<&usize>,
    ) -> Result<
        (
            VideoDecoder,
            Option<VideoReducer>,
            Option<usize>,
            Option<usize>,
        ),
        ffmpeg::Error,
    > {
        let input = ictx
            .streams()
            .best(Type::Video)
            .ok_or(ffmpeg::Error::StreamNotFound)?;
        let fps = f64::from(input.avg_frame_rate()).round();
        let duration = input.duration() as f64 * f64::from(input.time_base());
        let start_time = input.start_time();
        let time_base = f64::from(input.time_base());
        // setup the decoder context
        let mut context_decoder =
            ffmpeg::codec::context::Context::from_parameters(input.parameters())?;
        // setup multi-threading. `count` = 0 means let ffmpeg decide the optimal number
        // of cores to use. `kind` Frame or Slice (Frame is recommended).
        context_decoder.set_threading(threading::Config {
            kind: threading::Type::Frame,
            count: threads,
            // FIXME: beware this does not exist in ffmpeg 6.0 ?
            #[cfg(not(feature = "ffmpeg_6_0"))]
            safe: true,
        });
        let codec_id = context_decoder.id();
        let video = context_decoder.decoder().video()?;

        let orig_h = video.height();
        let orig_w = video.width();

        // Some more metadata
        let mut video_info: HashMap<&str, String> = HashMap::new();
        video_info.insert("fps", format!("{}", fps));
        video_info.insert("start_time", format!("{}", start_time));
        video_info.insert("time_base", format!("{}", time_base));
        video_info.insert("duration", format!("{}", duration));
        video_info.insert("codec_id", format!("{:?}", codec_id));
        video_info.insert("height", format!("{}", video.height()));
        video_info.insert("width", format!("{}", video.width()));
        video_info.insert("bit_rate", format!("{}", video.bit_rate()));
        video_info.insert("vid_format", format!("{:?}", video.format()));
        video_info.insert("aspect_ratio", format!("{:?}", video.aspect_ratio()));
        video_info.insert("color_space", format!("{:?}", video.color_space()));
        video_info.insert("color_range", format!("{:?}", video.color_range()));
        video_info.insert("color_primaries", format!("{:?}", video.color_primaries()));
        video_info.insert(
            "color_xfer_charac",
            format!("{:?}", video.color_transfer_characteristic()),
        );
        video_info.insert("chroma_location", format!("{:?}", video.chroma_location()));
        video_info.insert("vid_ref", format!("{}", video.references()));
        video_info.insert(
            "intra_dc_precision",
            format!("{}", video.intra_dc_precision()),
        );

        // do we need to resize the video ?
        let (h, w) = match resize_shorter_side {
            None => (orig_h, orig_w),
            Some(resize) => get_resized_dim(orig_h as f64, orig_w as f64, resize),
        };

        let scaler = Context::get(
            video.format(),
            orig_w,
            orig_h,
            AvPixel::RGB24,
            w,
            h,
            Flags::BILINEAR,
        )?;

        // setup the VideoReducer if needed
        let (reducer, first_frame, last_frame) = if with_reducer {
            // which frames we want to gather ?
            // we may want to start or end from a specific frame
            let start_frame = *start_frame.unwrap_or(&0);
            let end_frame = *end_frame.unwrap_or(&frame_count).min(&frame_count);
            let n_frames_compressed = ((end_frame - start_frame) as f64
                * compression_factor.unwrap_or(1.))
            .round() as usize;
            let indices = Array::linspace(
                start_frame as f64,
                end_frame as f64 - 1.,
                n_frames_compressed,
            )
            .iter_mut()
            .map(|x| x.round() as usize)
            .collect::<Vec<_>>();

            let frame_index = 0;
            let full_video: VideoArray = Array::zeros((indices.len(), h as usize, w as usize, 3));
            // counter to keep track of how many frames already added to the video
            (
                Some(VideoReducer {
                    indices,
                    frame_index,
                    full_video,
                    idx_counter: 0,
                }),
                Some(start_frame),
                Some(end_frame),
            )
        } else {
            (None, None, None)
        };
        Ok((
            VideoDecoder {
                video,
                scaler,
                height: h,
                width: w,
                fps,
                video_info,
                // start_time,
                // time_base,
            },
            reducer,
            first_frame,
            last_frame,
        ))
    }

    pub fn decode_video(mut self) -> Result<VideoArray, ffmpeg::Error> {
        let first_index = self.first_frame.unwrap_or(0);
        let mut seeked = false;
        let mut first_frame: FrameArray =
            Array::zeros((self.decoder.height as usize, self.decoder.width as usize, 3));
        // check if first_index is after the first keyframe, if so we can seek
        if self
            .stream_info
            .key_frames
            .iter()
            .any(|k| &first_index >= k)
            && (first_index > 0)
        {
            let frame_duration = (1. / self.decoder.fps * 1_000.0).round() as usize;
            self.seek_accurate(first_index, &frame_duration, &mut first_frame.view_mut())?;
            seeked = true;
        }
        match self.reducer {
            Some(mut reducer) => {
                reducer.frame_index = self.curr_frame;
                if seeked {
                    // first frame was seeked and decoded, so we increment the counter
                    // and assign the frame to the full video
                    reducer
                        .full_video
                        .slice_mut(s![reducer.idx_counter, .., .., ..])
                        .assign(&first_frame);
                    reducer.indices.remove(0);
                    reducer.idx_counter += 1;
                }
                for (stream, packet) in self.ictx.packets() {
                    if &reducer.frame_index > reducer.indices.iter().max().unwrap_or(&0) {
                        break;
                    }
                    if stream.index() == self.stream_index {
                        self.decoder.video.send_packet(&packet)?;
                        self.decoder
                            .receive_and_process_decoded_frames(&mut reducer)?;
                    } else {
                        debug!("Packet for another stream");
                    }
                }
                self.decoder.video.send_eof()?;
                // only process the remaining frames if we haven't reached the last frame
                if !reducer.indices.is_empty()
                    && (&reducer.frame_index <= reducer.indices.iter().max().unwrap_or(&0))
                {
                    self.decoder
                        .receive_and_process_decoded_frames(&mut reducer)?;
                }
                Ok(reducer.full_video)
            }
            None => panic!("No Reducer to get the frames"),
        }
    }
}

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
            frame_times.insert(
                pts,
                FrameTime {
                    pts,
                    dur,
                    dts,
                    didx,
                },
            );
            if packet.is_key() {
                key_frames.push(didx);
            }
            didx += 1;
        }
    }

    // mapping between decoding order and presentation order
    let mut decode_order: HashMap<usize, usize> = HashMap::new();
    for (idx, fr_info) in frame_times.iter().enumerate() {
        decode_order.insert(fr_info.1.didx, idx);
    }

    // Seek back to the begining of the stream
    ictx.seek(0, ..10)?;
    Ok(StreamInfo {
        frame_count: didx,
        key_frames,
        frame_times,
        decode_order,
    })
}

/// Get the resized dimension of a frame, keep the aspect ratio.
/// Resize the shorter side of the frame, and the other side accordingly
/// so that the resizing operation is minimal. Returns a resizing dimension only
/// if the shorter side of the frame is bigger than resize_shorter_side_to.
/// * height (f64): Height of the frame
/// * width (f64): Width of the frame
/// * resize_shorter_side_to (f64): Resize the shorter side of the frame to this value.
///
/// Returns: Option<(f64, f64)>: Option of the resized height and width
pub fn get_resized_dim(mut height: f64, mut width: f64, resize_shorter_side_to: f64) -> (u32, u32) {
    let mut short_side_res = height;
    if width < height {
        short_side_res = width;
    }
    if height == short_side_res {
        width = (width * resize_shorter_side_to / height).round();
        height = resize_shorter_side_to;
    } else {
        height = (height * resize_shorter_side_to / width).round();
        width = resize_shorter_side_to;
    }
    (height as u32, width as u32)
}

/// Encode frames to a video file with h264 codec.
/// * `frames` - Video frames to encode.
/// * `destination_path` - Path to save the video.
/// * `fps` - Frames per second of the video.
/// * `codec` - Codec to use for encoding the video, eg "h264".
pub fn save_video(
    frames: Array4<u8>,
    destination_path: &str,
    fps: usize,
    codec: &str,
) -> Result<(), ffmpeg::Error> {
    video_rs::init().unwrap();
    let shape = frames.shape();
    let n_frames = shape[0];
    let height = shape[1];
    let width = shape[2];

    let destination: Location = Location::File(PathBuf::from(destination_path));
    let mut encoder = match codec {
        "h264" => {
            let settings = Settings::preset_h264_yuv420p(width, height, false);
            Encoder::new(&destination, settings).expect("failed to create encoder")
        }
        _ => {
            // encode with mpeg4 format, ie XviD
            // FIXME: does not work, video is still encoded as h264
            let mut opts = HashMap::new();
            opts.insert("vcodec".to_string(), "libxvid".to_string());
            opts.insert("vtag".to_string(), "xvid".to_string());
            opts.insert("q:v".to_string(), "5".to_string());
            let settings =
                Settings::preset_h264_custom(width, height, AvPixel::YUV420P, opts.into());
            Encoder::new(&destination, settings).expect("failed to create encoder")
        }
    };

    let duration: Time = Time::from_nth_of_a_second(fps);
    let mut position = Time::zero();
    for i in 0..n_frames {
        encoder
            .encode(&frames.slice(s![i, .., .., ..]).to_owned(), position)
            .expect("failed to encode frame");

        // Update the current position and add the inter-frame duration to it.
        position = position.aligned_with(duration).add();
    }

    encoder.finish().expect("failed to finish encoder");
    Ok(())
}

/// Converts an RGB24 video `AVFrame` produced by ffmpeg to an `ndarray`.
/// Copied from https://github.com/oddity-ai/video-rs
/// * `frame` - Video frame to convert.
/// * returns a three-dimensional `ndarray` with dimensions `(H, W, C)` and type byte.
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

/// Convert all frames in the video from RGB to grayscale.
pub fn rgb2gray(frames: Array4<u8>) -> Array3<u8> {
    frames.map_axis(Axis(3), |pix| {
        (0.2125 * pix[0] as f32 + 0.7154 * pix[1] as f32 + 0.0721 * pix[2] as f32)
            .round()
            .clamp(0.0, 255.0) as u8
    })
}

impl VideoReader {
    /// Safely get the batch of frames from the video by iterating over all frames and decoding
    /// only the ones we need. This is of course slower than seeking to closest keyframe, but
    /// can be more accurate when the video's metadata is not reliable.
    pub fn get_batch_safe(mut self, indices: Vec<usize>) -> Result<VideoArray, ffmpeg::Error> {
        let first_index = self.first_frame.unwrap_or(0);
        let last_index = self.last_frame.unwrap_or(self.stream_info.frame_count - 1);
        // check if first_index is after the first keyframe, if so we can seek
        let key_pos = self.locate_keyframes(&first_index, &self.stream_info.key_frames);
        if key_pos > 0 {
            let frame_duration = (1. / self.decoder.fps * 1_000.0).round() as usize;
            self.seek_frame(&key_pos, &frame_duration)?;
        }
        let mut frame_map: HashMap<usize, FrameArray> = HashMap::new();
        match self.reducer {
            Some(mut reducer) => {
                if key_pos > 0 {
                    reducer.frame_index = key_pos;
                }

                for (stream, packet) in self.ictx.packets() {
                    if stream.index() == self.stream_index {
                        self.decoder.video.send_packet(&packet)?;
                        self.decoder.skip_and_decode_frames(
                            &mut reducer,
                            &indices,
                            &mut frame_map,
                        )?;
                    } else {
                        debug!("Packet for another stream");
                    }
                    if reducer.frame_index > last_index {
                        break;
                    }
                }
                self.decoder.video.send_eof()?;
                if reducer.frame_index <= last_index {
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
                    .map(|(i, idx)| {
                        frame_batch
                            .slice_mut(s![i, .., .., ..])
                            .assign(frame_map.get(idx).unwrap())
                    })
                    .collect::<Vec<_>>();

                Ok(frame_batch)
            }
            None => panic!("No Reducer to get the frames"),
        }
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
        // frame rate
        let fps = self.decoder.fps;
        // duration of a frame in micro seconds
        let frame_duration = (1. / fps * 1_000.0).round() as usize;

        for (idx_counter, frame_index) in indices.into_iter().enumerate() {
            self.n_fails = 0;
            debug!("[NEXT INDICE] frame_index: {frame_index}");
            self.seek_accurate(
                frame_index,
                &frame_duration,
                &mut video_frames.slice_mut(s![idx_counter, .., .., ..]),
            )?;
        }
        Ok(video_frames)
    }

    /// How many frames we need to skip to go from current decoding index `curr_dec_idx`
    /// to `target_dec_index`. The `frame_index` argument corresponds to the presentation
    /// index, while we need to know the number of frames to skip in terms of decoding index.
    pub fn get_num_skip(&self, frame_index: &usize) -> usize {
        let target_dec_idx = self.stream_info.decode_order.get(frame_index).unwrap();
        target_dec_idx.saturating_sub(self.curr_dec_idx)
    }

    pub fn seek_accurate(
        &mut self,
        frame_index: usize,
        frame_duration: &usize,
        frame_array: &mut ArrayViewMut3<u8>,
    ) -> Result<(), ffmpeg::Error> {
        let key_pos = self.locate_keyframes(&frame_index, &self.stream_info.key_frames);
        debug!("    - Key pos: {}", key_pos);
        let curr_key_pos = self.locate_keyframes(&self.curr_dec_idx, &self.stream_info.key_frames);
        debug!("    - Curr key pos: {}", curr_key_pos);
        if (key_pos == curr_key_pos) & (frame_index > self.curr_frame) {
            // we can directly skip until frame_index
            let num_skip = self.get_num_skip(&frame_index);
            self.skip_frames(num_skip, &frame_index, frame_array)?;
        } else {
            if key_pos < curr_key_pos {
                self.ictx.seek(0, ..100)?;
                self.avflushbuf()?;
            }
            self.seek_frame(&key_pos, frame_duration)?;
            let num_skip = self.get_num_skip(&frame_index);
            self.skip_frames(num_skip, &frame_index, frame_array)?;
        }
        Ok(())
    }

    pub fn locate_keyframes(&self, pos: &usize, key_frames: &[usize]) -> usize {
        let key_pos = key_frames.iter().filter(|e| pos >= *e).max().unwrap_or(&0);
        key_pos.to_owned()
    }

    pub fn skip_frames(
        &mut self,
        num: usize,
        frame_index: &usize,
        frame_array: &mut ArrayViewMut3<u8>,
    ) -> Result<(), ffmpeg::Error> {
        let num_skip = num.min(self.stream_info.frame_count - 1);
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
                        let mut decoded = Video::empty();
                        while self.decoder.video.receive_frame(&mut decoded).is_ok() {
                            if &self.curr_frame == frame_index {
                                let mut rgb_frame = Video::empty();
                                self.decoder.scaler.run(&decoded, &mut rgb_frame)?;
                                convert_frame_to_ndarray_rgb24(&mut rgb_frame, frame_array)?;
                                self.update_indices();
                                return Ok(());
                            }
                            self.update_indices();
                            failsafe -= 1;
                        }
                    }
                }
                None => failsafe -= 1,
            }
        }
        debug!(
            "Finished skipping, current frame is now: {}",
            self.curr_frame
        );
        Ok(())
    }

    pub fn update_indices(&mut self) {
        self.curr_dec_idx += 1;
        self.curr_frame = *self
            .stream_info
            .decode_order
            .get(&self.curr_dec_idx)
            .unwrap_or(&self.stream_info.frame_count);
    }

    // AVSEEK_FLAG_BACKWARD 1 <- seek backward
    // AVSEEK_FLAG_BYTE 2 <- seeking based on position in bytes
    // AVSEEK_FLAG_ANY 4 <- seek to any frame, even non-key frames
    // AVSEEK_FLAG_FRAME 8 <- seeking baseed on frame number
    pub fn seek_frame(&mut self, pos: &usize, frame_duration: &usize) -> Result<(), ffmpeg::Error> {
        let frame_ts = match self.stream_info.frame_times.iter().nth(*pos) {
            Some((_, fr_ts)) => fr_ts.pts,
            None => (pos * frame_duration) as i64,
        };

        let mut flag = 0;
        if pos < &self.curr_dec_idx {
            flag = AVSEEK_FLAG_BACKWARD;
        }
        match self.avseekframe(pos, frame_ts, flag) {
            Ok(()) => Ok(()),
            Err(_) => {
                if flag != 0 {
                    match self.avseekframe(pos, frame_ts, AVSEEK_FLAG_BACKWARD) {
                        Ok(()) => Ok(()),
                        Err(_) => self.avseekframe(pos, *pos as i64, AVSEEK_FLAG_FRAME),
                    }
                } else {
                    self.avseekframe(pos, *pos as i64, AVSEEK_FLAG_FRAME)
                }
            }
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
            self.curr_frame = *self
                .stream_info
                .decode_order
                .get(&self.curr_dec_idx)
                .unwrap();
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
