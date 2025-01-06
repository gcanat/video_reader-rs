use ffmpeg::codec::threading;
use ffmpeg::ffi::*;
use ffmpeg::format::{input, Pixel as AvPixel};
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{context::Context, flag::Flags};
use ffmpeg::util::frame::video::Video;
use ffmpeg::util::rational::Rational;
use ffmpeg_next as ffmpeg;
use log::debug;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use video_rs::encode::{Encoder, Settings};
use video_rs::location::Location;
use video_rs::time::Time;

use ndarray::parallel::prelude::*;
use ndarray::{s, stack, Array, Array2, Array3, Array4, ArrayView3, ArrayViewMut3, Axis};
use tokio::task;
use yuvutils_rs::{yuv420_to_rgb, YuvPlanarImage, YuvRange, YuvStandardMatrix};

pub type FrameArray = Array3<u8>;
pub type VideoArray = Array4<u8>;

struct VideoParams {
    duration: f64,
    start_time: i64,
    time_base: f64,
}

#[derive(Default)]
pub struct DecoderConfig {
    threads: usize,
    resize_shorter_side: Option<f64>,
}

impl DecoderConfig {
    pub fn new(threads: usize, resize_shorter_side: Option<f64>) -> Self {
        Self {
            threads,
            resize_shorter_side,
        }
    }
}

fn extract_video_params(input: &ffmpeg::Stream) -> VideoParams {
    VideoParams {
        duration: input.duration() as f64 * f64::from(input.time_base()),
        start_time: input.start_time(),
        time_base: f64::from(input.time_base()),
    }
}

fn setup_decoder_context(
    input: &ffmpeg::Stream,
    threads: usize,
) -> Result<ffmpeg::codec::context::Context, ffmpeg::Error> {
    let mut context = ffmpeg::codec::context::Context::from_parameters(input.parameters())?;
    context.set_threading(threading::Config {
        kind: threading::Type::Frame,
        count: threads,
        #[cfg(not(feature = "ffmpeg_6_0"))]
        safe: true,
    });
    Ok(context)
}

fn collect_video_metadata(
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
    info.insert("duration", params.duration.to_string());
    info.insert("codec_id", format!("{:?}", video.codec().unwrap().id()));
    info.insert("height", video.height().to_string());
    info.insert("width", video.width().to_string());
    info.insert("bit_rate", video.bit_rate().to_string());
    info.insert("vid_format", format!("{:?}", video.format()));
    info.insert("aspect_ratio", format!("{:?}", video.aspect_ratio()));
    info.insert("color_space", format!("{:?}", video.color_space()));
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

fn create_video_reducer(
    start_frame: Option<usize>,
    end_frame: Option<usize>,
    frame_count: usize,
    compression_factor: Option<f64>,
    height: u32,
    width: u32,
) -> (Option<VideoReducer>, Option<usize>, Option<usize>) {
    let start = start_frame.unwrap_or(0);
    let end = end_frame.unwrap_or(frame_count).min(frame_count);

    let n_frames = ((end - start) as f64 * compression_factor.unwrap_or(1.0)).round() as usize;

    let indices = Array::linspace(start as f64, end as f64 - 1., n_frames)
        .iter()
        .map(|x| x.round() as usize)
        .collect::<Vec<_>>();

    let full_video = Array::zeros((indices.len(), height as usize, width as usize, 3));

    (
        Some(VideoReducer {
            indices,
            frame_index: 0,
            full_video,
            idx_counter: 0,
        }),
        Some(start),
        Some(end),
    )
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
}

unsafe impl Send for VideoReader {}

impl VideoReader {
    pub fn decoder(&self) -> &VideoDecoder {
        &self.decoder
    }
    pub fn stream_info(&self) -> &StreamInfo {
        &self.stream_info
    }
}

/// Struct responsible for doing the actual decoding
pub struct VideoDecoder {
    video: ffmpeg::decoder::Video,
    height: u32,
    width: u32,
    fps: f64,
    video_info: HashMap<&'static str, String>,
}

unsafe impl Send for VideoDecoder {}

impl VideoDecoder {
    pub fn video_info(&self) -> &HashMap<&'static str, String> {
        &self.video_info
    }
    pub fn fps(&self) -> f64 {
        self.fps
    }
    pub fn video(&self) -> &ffmpeg::decoder::Video {
        &self.video
    }
}

/// Struct used when we want to decode the whole video with a compression_factor
#[derive(Clone)]
pub struct VideoReducer {
    indices: Vec<usize>,
    frame_index: usize,
    idx_counter: usize,
    full_video: VideoArray,
}

/// Timing info for key frames
#[derive(Debug)]
pub struct FrameTime {
    pts: i64,
    dur: i64,
    dts: i64,
}

impl FrameTime {
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
    pub fn frame_count(&self) -> &usize {
        &self.frame_count
    }
    pub fn key_frames(&self) -> &Vec<usize> {
        &self.key_frames
    }
    pub fn frame_times(&self) -> &BTreeMap<usize, FrameTime> {
        &self.frame_times
    }
}

impl VideoDecoder {
    /// Decode all frames that match the frame indices
    pub fn receive_and_process_decoded_frames(
        &mut self,
        scaler: &mut Context,
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
                scaler.run(&decoded, &mut rgb_frame)?;
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
        scaler: &mut Context,
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
                scaler.run(&decoded, &mut rgb_frame)?;
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

        let decoder_context = setup_decoder_context(&input, config.threads)?;
        let video = decoder_context.decoder().video()?;

        let video_info = collect_video_metadata(&video, &video_params, &fps);

        let (height, width) = match config.resize_shorter_side {
            Some(resize) => get_resized_dim(video.height() as f64, video.width() as f64, resize),
            None => (video.height(), video.width()),
        };

        Ok(VideoDecoder {
            video,
            height,
            width,
            fps,
            video_info,
        })
    }

    pub fn get_scaler(&self, pix_fmt: AvPixel) -> Result<Context, ffmpeg::Error> {
        let scaler = Context::get(
            self.decoder.video.format(),
            self.decoder.video.width(),
            self.decoder.video.height(),
            pix_fmt,
            self.decoder.width,
            self.decoder.height,
            Flags::BILINEAR,
        )?;
        Ok(scaler)
    }

    pub fn decode_video(
        &mut self,
        start_frame: Option<usize>,
        end_frame: Option<usize>,
        compression_factor: Option<f64>,
    ) -> Result<VideoArray, ffmpeg::Error> {
        let (reducer, start_frame, _end_frame) = create_video_reducer(
            start_frame,
            end_frame,
            self.stream_info.frame_count,
            compression_factor,
            self.decoder.height,
            self.decoder.width,
        );
        let first_index = start_frame.unwrap_or(0);

        let mut scaler = self.get_scaler(AvPixel::RGB24)?;

        // make sure we are at the begining of the stream
        self.seek_to_start()?;

        // check if first_index is after the first keyframe, if so we can seek
        if self
            .stream_info
            .key_frames
            .iter()
            .any(|k| &first_index >= k)
            && (first_index > 0)
        {
            let frame_duration = (1. / self.decoder.fps * 1_000.0).round() as usize;
            let key_pos = self.locate_keyframes(&first_index, &self.stream_info.key_frames);
            self.seek_frame(&key_pos, &frame_duration)?;
            self.curr_frame = key_pos;
        }

        let mut reducer = reducer.unwrap();
        reducer.frame_index = self.curr_frame;
        for (stream, packet) in self.ictx.packets() {
            if &reducer.frame_index > reducer.indices.iter().max().unwrap_or(&0) {
                break;
            }
            if stream.index() == self.stream_index {
                self.decoder.video.send_packet(&packet)?;
                self.decoder
                    .receive_and_process_decoded_frames(&mut scaler, &mut reducer)?;
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
                .receive_and_process_decoded_frames(&mut scaler, &mut reducer)?;
        }
        Ok(reducer.full_video)
    }
    pub async fn decode_video_fast(
        &mut self,
        start_frame: Option<usize>,
        end_frame: Option<usize>,
        compression_factor: Option<f64>,
    ) -> Result<Vec<FrameArray>, ffmpeg::Error> {
        let (reducer, start_frame, _) = create_video_reducer(
            start_frame,
            end_frame,
            self.stream_info.frame_count,
            compression_factor,
            self.decoder.height,
            self.decoder.width,
        );
        let first_index = start_frame.unwrap_or(0);

        let mut scaler = self.get_scaler(AvPixel::YUV420P)?;

        // make sure we are at the begining of the stream
        self.seek_to_start()?;

        if self
            .stream_info
            .key_frames
            .iter()
            .any(|k| &first_index >= k)
            && (first_index > 0)
        {
            let key_pos = self.locate_keyframes(&first_index, &self.stream_info.key_frames);
            let fps = self.decoder.fps;
            // duration of a frame in micro seconds
            let frame_duration = (1. / fps * 1_000.0).round() as usize;
            // seek to closest key frame before first_index
            self.seek_frame(&key_pos, &frame_duration)?;
        }

        let mut reducer = reducer.unwrap();
        reducer.frame_index = self.curr_frame;
        let mut tasks = vec![];

        let mut receive_and_process_decoded_frames = |decoder: &mut ffmpeg::decoder::Video,
                                                      mut curr_frame: usize|
         -> Result<usize, ffmpeg::Error> {
            let mut decoded = Video::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                if reducer.indices.iter().any(|x| x == &curr_frame) {
                    let mut rgb_frame = Video::empty();
                    scaler.run(&decoded, &mut rgb_frame).unwrap();
                    tasks.push(task::spawn(async move {
                        convert_yuv_to_ndarray_rgb24(rgb_frame)
                    }));
                }
                curr_frame += 1;
            }
            Ok(curr_frame)
        };

        for (stream, packet) in self.ictx.packets() {
            if &self.curr_frame > reducer.indices.iter().max().unwrap_or(&0) {
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
        if !reducer.indices.is_empty()
            && (&self.curr_frame <= reducer.indices.iter().max().unwrap_or(&0))
        {
            let upd_curr_frame =
                receive_and_process_decoded_frames(&mut self.decoder.video, self.curr_frame)?;
            self.curr_frame = upd_curr_frame;
        }

        let mut outputs = Vec::with_capacity(tasks.len());
        for task_ in tasks {
            outputs.push(task_.await.unwrap());
        }
        Ok(outputs)
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
            frame_times.insert(didx, FrameTime { pts, dur, dts });
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
/// * `frame_array` mutable reference to the ndarray where the data will be copied
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

/// Converts a YUV420P video `AVFrame` produced by ffmpeg to an `ndarray`.
/// * `frame` - Video frame to convert.
/// * returns a three-dimensional `ndarray` with dimensions `(H, W, C)` and type byte.
pub fn convert_yuv_to_ndarray_rgb24(mut frame: Video) -> Array3<u8> {
    unsafe {
        let frame_ptr = frame.as_mut_ptr();
        let frame_width: i32 = (*frame_ptr).width;
        let frame_height: i32 = (*frame_ptr).height;
        let frame_format =
            std::mem::transmute::<std::ffi::c_int, AVPixelFormat>((*frame_ptr).format);
        assert_eq!(frame_format, AVPixelFormat::AV_PIX_FMT_YUV420P);

        let mut buf_vec = vec![0_u8; (frame_width * (frame_height + frame_height / 2)) as usize];

        let bytes_copied = av_image_copy_to_buffer(
            buf_vec.as_mut_ptr(),
            buf_vec.len() as i32,
            (*frame_ptr).data.as_ptr() as *const *const u8,
            (*frame_ptr).linesize.as_ptr(),
            frame_format,
            frame_width,
            frame_height,
            1,
        );

        // By default assume HD color space
        let mut colorspace = YuvStandardMatrix::Bt709;
        if frame_height < 720 {
            // SD color space
            colorspace = YuvStandardMatrix::Bt601;
        } else if frame_height > 1080 {
            // UHD color space
            colorspace = YuvStandardMatrix::Bt2020;
        }

        if bytes_copied == buf_vec.len() as i32 {
            let mut rgb = vec![0_u8; (frame_width * frame_height * 3) as usize];
            let cut_point1 = (frame_width * frame_height) as usize;
            let cut_point2 = cut_point1 + cut_point1 / 4;
            let yuv_planar = YuvPlanarImage {
                y_plane: &buf_vec[..cut_point1],
                y_stride: frame_width as u32,
                u_plane: &buf_vec[cut_point1..cut_point2],
                u_stride: (frame_width / 2) as u32,
                v_plane: &buf_vec[cut_point2..],
                v_stride: (frame_width / 2) as u32,
                width: frame_width as u32,
                height: frame_height as u32,
            };
            yuv420_to_rgb(
                &yuv_planar,
                &mut rgb,
                (frame_width * 3) as u32,
                YuvRange::Full,
                colorspace,
            )
            .unwrap();
            Array3::from_shape_vec((frame_height as usize, frame_width as usize, 3_usize), rgb)
                .unwrap()
        } else {
            Array3::zeros((frame_height as usize, frame_width as usize, 3_usize))
        }
    }
}

/// Convert RGB video (N, H, W, C) to Grayscale video (N, H, W).
/// Returns a 3D ndarray with shape (N, H, W).
pub fn rgb2gray(frames: Array4<u8>) -> Array3<u8> {
    let mut gray = Vec::new();
    frames
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(rgb2gray_2d)
        .collect_into_vec(&mut gray);
    let views: Vec<_> = gray.iter().map(|x| x.view()).collect();
    stack(Axis(0), &views[..]).unwrap()
}

/// Convert RGB Frame (H, W, C) to grayscale (H, W).
fn rgb2gray_2d(frames: ArrayView3<u8>) -> Array2<u8> {
    frames.map_axis(Axis(2), |pix| {
        (0.2125 * pix[0] as f32 + 0.7154 * pix[1] as f32 + 0.0721 * pix[2] as f32)
            .round()
            .clamp(0.0, 255.0) as u8
    })
}

impl VideoReader {
    /// Safely get the batch of frames from the video by iterating over all frames and decoding
    /// only the ones we need. This can be more accurate when the video's metadata is not reliable,
    /// or when the video has B-frames.
    pub fn get_batch_safe(&mut self, indices: Vec<usize>) -> Result<VideoArray, ffmpeg::Error> {
        let first_index = indices.iter().min().unwrap_or(&0);
        let max_index = self.stream_info.frame_count - 1;
        let last_index = indices.iter().max().unwrap_or(&max_index);
        let (reducer, _, _) = create_video_reducer(
            Some(*first_index),
            Some(*last_index),
            self.stream_info.frame_count,
            None,
            self.decoder.height,
            self.decoder.width,
        );

        let mut scaler = self.get_scaler(AvPixel::RGB24)?;

        // make sure we are at the begining of the stream
        self.seek_to_start()?;

        // check if closest key frames to first_index is non zero, if so we can seek
        let key_pos = self.locate_keyframes(first_index, &self.stream_info.key_frames);
        if key_pos > 0 {
            let frame_duration = (1. / self.decoder.fps * 1_000.0).round() as usize;
            self.seek_frame(&key_pos, &frame_duration)?;
        }
        let mut frame_map: HashMap<usize, FrameArray> = HashMap::new();

        let mut reducer = reducer.unwrap();
        if key_pos > 0 {
            reducer.frame_index = self.curr_frame;
        }

        for (stream, packet) in self.ictx.packets() {
            if stream.index() == self.stream_index {
                self.decoder.video.send_packet(&packet)?;
                self.decoder.skip_and_decode_frames(
                    &mut scaler,
                    &mut reducer,
                    &indices,
                    &mut frame_map,
                )?;
            } else {
                debug!("Packet for another stream");
            }
            if &reducer.frame_index > last_index {
                break;
            }
        }
        self.decoder.video.send_eof()?;
        if &reducer.frame_index <= last_index {
            self.decoder.skip_and_decode_frames(
                &mut scaler,
                &mut reducer,
                &indices,
                &mut frame_map,
            )?;
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
        // frame rate
        let fps = self.decoder.fps;
        // duration of a frame in micro seconds
        let frame_duration = (1. / fps * 1_000.0).round() as usize;

        let mut scaler = self.get_scaler(AvPixel::RGB24)?;

        // make sure we are at the begining of the stream
        self.seek_to_start()?;

        for (idx_counter, frame_index) in indices.into_iter().enumerate() {
            self.n_fails = 0;
            debug!("[NEXT INDICE] frame_index: {frame_index}");
            self.seek_accurate(
                frame_index,
                &frame_duration,
                &mut video_frames.slice_mut(s![idx_counter, .., .., ..]),
                &mut scaler,
            )?;
        }
        Ok(video_frames)
    }

    pub fn seek_accurate(
        &mut self,
        frame_index: usize,
        frame_duration: &usize,
        frame_array: &mut ArrayViewMut3<u8>,
        scaler: &mut Context,
    ) -> Result<(), ffmpeg::Error> {
        let key_pos = self.locate_keyframes(&frame_index, &self.stream_info.key_frames);
        debug!("    - Key pos: {}", key_pos);
        let curr_key_pos = self.locate_keyframes(&self.curr_dec_idx, &self.stream_info.key_frames);
        debug!("    - Curr key pos: {}", curr_key_pos);
        if (key_pos == curr_key_pos) & (frame_index > self.curr_frame) {
            // we can directly skip until frame_index
            debug!("No need to seek, we can directly skip frames");
            let num_skip = self.get_num_skip(&frame_index);
            self.skip_frames(num_skip, &frame_index, frame_array, scaler)?;
        } else {
            if key_pos < curr_key_pos {
                debug!("Seeking back to start");
                self.seek_to_start()?;
            }
            debug!("Seeking to key_pos: {}", key_pos);
            self.seek_frame(&key_pos, frame_duration)?;
            let num_skip = self.get_num_skip(&frame_index);
            self.skip_frames(num_skip, &frame_index, frame_array, scaler)?;
        }
        Ok(())
    }

    /// Find the closest key frame before `pos`
    pub fn locate_keyframes(&self, pos: &usize, key_frames: &[usize]) -> usize {
        let key_pos = key_frames.iter().filter(|e| pos >= *e).max().unwrap_or(&0);
        key_pos.to_owned()
    }

    /// How many frames we need to skip to go from current decoding index `curr_dec_idx`
    /// to `target_dec_index`. The `frame_index` argument corresponds to the presentation
    /// index, while we need to know the number of frames to skip in terms of decoding index.
    pub fn get_num_skip(&self, frame_index: &usize) -> usize {
        let target_dec_idx = self.stream_info.decode_order.get(frame_index);
        match target_dec_idx {
            Some(v) => v.saturating_sub(self.curr_dec_idx),
            None => *frame_index,
        }
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
        scaler: &mut Context,
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
                                scaler.run(&decoded, &mut rgb_frame)?;
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
        debug!(
            "dec_idx: {}, curr_frame: {}",
            self.curr_dec_idx, self.curr_frame
        );
    }

    // AVSEEK_FLAG_BACKWARD 1 <- seek backward
    // AVSEEK_FLAG_BYTE 2 <- seeking based on position in bytes
    // AVSEEK_FLAG_ANY 4 <- seek to any frame, even non-key frames
    // AVSEEK_FLAG_FRAME 8 <- seeking based on frame number
    pub fn seek_frame(&mut self, pos: &usize, frame_duration: &usize) -> Result<(), ffmpeg::Error> {
        let frame_ts = match self.stream_info.frame_times.iter().nth(*pos) {
            Some((_, fr_ts)) => fr_ts.pts,
            None => (pos * frame_duration) as i64,
        };

        match self.avseekframe(pos, frame_ts, AVSEEK_FLAG_BACKWARD) {
            Ok(()) => Ok(()),
            Err(_) => self.avseekframe(pos, *pos as i64, AVSEEK_FLAG_FRAME),
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
