use crate::ffi_hwaccel::codec_context_get_hw_frames_ctx;
use crate::hwaccel::HardwareAccelerationContext;
use ffmpeg::ffi::{av_buffersrc_parameters_alloc, av_buffersrc_parameters_set, AVPixelFormat};
use ffmpeg::filter;
use ffmpeg::format::Pixel as AvPixel;
use ffmpeg::util::rational::Rational;
use ffmpeg_next as ffmpeg;
use log::debug;

pub struct FilterConfig<'a> {
    height: u32,
    width: u32,
    vid_format: ffmpeg::util::format::Pixel,
    time_base: &'a str,
    spec: &'a str,
    is_hwaccel: bool,
}
impl<'a> FilterConfig<'a> {
    pub fn new(
        height: u32,
        width: u32,
        vid_format: ffmpeg::util::format::Pixel,
        time_base: &'a str,
        spec: &'a str,
        is_hwaccel: bool,
    ) -> Self {
        FilterConfig {
            height,
            width,
            vid_format,
            time_base,
            spec,
            is_hwaccel,
        }
    }
}

pub fn create_filters(
    decoder_ctx: &mut ffmpeg::codec::Context,
    hw_fmt: Option<ffmpeg::util::format::Pixel>,
    filter_cfg: FilterConfig,
) -> Result<filter::Graph, ffmpeg::Error> {
    let mut graph = filter::Graph::new();

    let args = format!(
        "video_size={}x{}:pix_fmt={}:time_base={}:pixel_aspect=1/1",
        filter_cfg.width,
        filter_cfg.height,
        filter_cfg.vid_format.descriptor().unwrap().name(),
        filter_cfg.time_base,
    );
    debug!("Buffer args: {}", args);

    let mut buffersrc_ctx = graph.add(&filter::find("buffer").unwrap(), "in", args.as_str())?;
    if let Some(hw_pix_fmt) = hw_fmt {
        create_hwbuffer_src(
            decoder_ctx,
            &mut buffersrc_ctx,
            filter_cfg.height,
            filter_cfg.width,
            hw_pix_fmt,
            filter_cfg.time_base,
            filter_cfg.is_hwaccel,
        )?;
    }
    graph.add(&filter::find("buffersink").unwrap(), "out", "")?;
    graph
        .output("in", 0)?
        .input("out", 0)?
        .parse(filter_cfg.spec)?;
    graph.validate()?;
    Ok(graph)
}

pub fn create_hwbuffer_src(
    codec_ctx: &mut ffmpeg::codec::context::Context,
    filt_ctx: &mut filter::Context,
    height: u32,
    width: u32,
    vid_format: ffmpeg::util::format::Pixel,
    time_base: &str,
    is_hwaccel: bool,
) -> Result<filter::Context, ffmpeg::util::error::Error> {
    let time_base = time_base.split_once('/').unwrap();

    unsafe {
        let params_ptr = av_buffersrc_parameters_alloc();
        if let Some(params) = params_ptr.as_mut() {
            params.format = Into::<AVPixelFormat>::into(vid_format) as i32;
            params.width = width as i32;
            params.height = height as i32;
            params.time_base = Rational(
                time_base.0.parse::<i32>().unwrap(),
                time_base.1.parse::<i32>().unwrap(),
            )
            .into();
            if is_hwaccel {
                params.hw_frames_ctx = (*codec_ctx.as_mut_ptr()).hw_frames_ctx;
            }
        };
        match av_buffersrc_parameters_set(filt_ctx.as_mut_ptr(), params_ptr) {
            n if n >= 0 => Ok(filter::Context::wrap(filt_ctx.as_mut_ptr())),
            e => Err(ffmpeg::Error::from(e)),
        }
    }
}

fn transpose_filter(rotation: isize) -> String {
    match rotation {
        -90 => ",transpose=1".to_owned(),
        90 => ",transpose=2".to_owned(),
        180 | -180 => ",transport=1,transpose=1".to_owned(),
        _ => "".to_owned(),
    }
}

pub fn create_filter_spec(
    width: u32,
    height: u32,
    video: &mut ffmpeg::decoder::Video,
    ff_filter: Option<String>,
    hwaccel_context: Option<HardwareAccelerationContext>,
    hwaccel_fmt: AvPixel,
    rotation: isize,
) -> Result<(String, Option<AvPixel>), ffmpeg_next::Error> {
    let pix_fmt = AvPixel::YUV420P;
    let mut hw_format: Option<AvPixel> = None;

    let filter_spec = match ff_filter {
        None => {
            let mut filter_spec = format!(
                "format={},scale=w={}:h={}:flags=fast_bilinear{}",
                pix_fmt.descriptor().unwrap().name(),
                width,
                height,
                transpose_filter(rotation),
            );

            if let Some(hw_ctx) = hwaccel_context {
                hw_format = Some(hw_ctx.format());
                // Need a custom filter for hwaccel != cuda
                if hw_format != Some(ffmpeg::util::format::Pixel::CUDA) {
                    // FIXME: proper error handling
                    println!("Using hwaccel other than cuda, you should provide a custom filter");
                    return Err(ffmpeg::error::Error::DecoderNotFound);
                }
                filter_spec = format!(
                    "scale_cuda=w={}:h={}:passthrough=0,hwdownload,format={}{}",
                    width,
                    height,
                    hwaccel_fmt.descriptor().unwrap().name(),
                    transpose_filter(rotation),
                );
                codec_context_get_hw_frames_ctx(video, hw_format.unwrap(), hwaccel_fmt)?;
            }
            filter_spec
        }
        Some(spec) => {
            if let Some(hw_ctx) = hwaccel_context {
                hw_format = Some(hw_ctx.format());
                codec_context_get_hw_frames_ctx(video, hw_format.unwrap(), hwaccel_fmt)?;
            }
            spec
        }
    };
    Ok((filter_spec, hw_format))
}
