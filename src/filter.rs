use crate::ffi_hwaccel::codec_context_get_hw_frames_ctx;
use crate::hwaccel::HardwareAccelerationContext;
use ffmpeg::ffi::{
    av_buffersrc_parameters_alloc, av_buffersrc_parameters_set, av_free, AVPixelFormat,
};
use ffmpeg::filter;
use ffmpeg::format::Pixel as AvPixel;
use ffmpeg::util::rational::Rational;
use ffmpeg_next as ffmpeg;
use log::debug;
use std::ffi::c_void;

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

    // Get pixel format name, fallback to yuv420p if unknown
    let pix_fmt_name = filter_cfg
        .vid_format
        .descriptor()
        .map(|d| d.name())
        .unwrap_or("yuv420p");

    let args = format!(
        "video_size={}x{}:pix_fmt={}:time_base={}:pixel_aspect=1/1",
        filter_cfg.width, filter_cfg.height, pix_fmt_name, filter_cfg.time_base,
    );
    debug!("Buffer args: {}", args);

    let buffer = filter::find("buffer").ok_or(ffmpeg::Error::Bug)?;
    let mut buffersrc_ctx = graph.add(&buffer, "in", args.as_str())?;
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
    let buffersink = filter::find("buffersink").ok_or(ffmpeg::Error::Bug)?;
    graph.add(&buffersink, "out", "")?;
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
    let Some(time_base) = time_base.split_once('/') else {
        return Err(ffmpeg::Error::InvalidData);
    };

    unsafe {
        let params_ptr = av_buffersrc_parameters_alloc();
        if params_ptr.is_null() {
            return Err(ffmpeg::error::Error::Bug);
        }
        if let Some(params) = params_ptr.as_mut() {
            params.format = Into::<AVPixelFormat>::into(vid_format) as i32;
            params.width = width as i32;
            params.height = height as i32;
            let tb_num = time_base
                .0
                .parse::<i32>()
                .map_err(|_| ffmpeg::Error::InvalidData)?;
            let tb_den = time_base
                .1
                .parse::<i32>()
                .map_err(|_| ffmpeg::Error::InvalidData)?;
            params.time_base = Rational(tb_num, tb_den).into();
            if is_hwaccel {
                params.hw_frames_ctx = (*codec_ctx.as_mut_ptr()).hw_frames_ctx;
            }
        };
        let ret = av_buffersrc_parameters_set(filt_ctx.as_mut_ptr(), params_ptr);
        av_free(params_ptr as *mut c_void);
        match ret {
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

/// Parse scale dimensions from filter string (e.g., "scale=w=256:h=256" or "scale=256:256")
/// Returns (width, height) if found, None otherwise
fn parse_scale_from_filter(filter_spec: &str) -> Option<(u32, u32)> {
    // Try to match "scale=w=XXX:h=YYY" format
    if let Some(scale_pos) = filter_spec.find("scale=") {
        let scale_str = &filter_spec[scale_pos + 6..];

        // Try "w=XXX:h=YYY" format
        if scale_str.starts_with("w=") {
            let mut width = None;
            let mut height = None;

            for part in scale_str.split(':') {
                if part.starts_with("w=") {
                    width = part[2..].parse().ok();
                } else if part.starts_with("h=") {
                    height = part[2..].parse().ok();
                }
            }

            if let (Some(w), Some(h)) = (width, height) {
                return Some((w, h));
            }
        }

        // Try "WIDTH:HEIGHT" format (e.g., "scale=256:256")
        let parts: Vec<&str> = scale_str.split(':').collect();
        if parts.len() >= 2 {
            if let (Ok(w), Ok(h)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
                return Some((w, h));
            }
        }
    }
    None
}

pub fn create_filter_spec(
    width: u32,
    height: u32,
    video: &mut ffmpeg::decoder::Video,
    ff_filter: Option<String>,
    hwaccel_context: Option<HardwareAccelerationContext>,
    hwaccel_fmt: AvPixel,
    rotation: isize,
) -> Result<(String, Option<AvPixel>, u32, u32), ffmpeg_next::Error> {
    let pix_fmt = AvPixel::YUV420P;
    let mut hw_format: Option<AvPixel> = None;
    let mut out_width = width;
    let mut out_height = height;

    let filter_spec = match ff_filter {
        None => {
            let pix_fmt_name = pix_fmt.descriptor().map(|d| d.name()).unwrap_or("yuv420p");
            let mut filter_spec = format!(
                "format={},scale=w={}:h={}:flags=fast_bilinear{}",
                pix_fmt_name,
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
                let hwaccel_fmt_name = hwaccel_fmt.descriptor().map(|d| d.name()).unwrap_or("nv12");
                filter_spec = format!(
                    "scale_cuda=w={}:h={}:passthrough=0,hwdownload,format={}{}",
                    width,
                    height,
                    hwaccel_fmt_name,
                    transpose_filter(rotation),
                );
                if let Some(hwfmt) = hw_format {
                    codec_context_get_hw_frames_ctx(video, hwfmt, hwaccel_fmt)?;
                } else {
                    return Err(ffmpeg::error::Error::DecoderNotFound);
                }
            }
            filter_spec
        }
        Some(spec) => {
            // Parse scale dimensions from custom filter
            if let Some((w, h)) = parse_scale_from_filter(&spec) {
                out_width = w;
                out_height = h;
                debug!("Parsed scale from custom filter: {}x{}", w, h);
            }

            if let Some(hw_ctx) = hwaccel_context {
                hw_format = Some(hw_ctx.format());
                if let Some(hwfmt) = hw_format {
                    codec_context_get_hw_frames_ctx(video, hwfmt, hwaccel_fmt)?;
                } else {
                    return Err(ffmpeg::error::Error::DecoderNotFound);
                }
            }
            spec
        }
    };
    Ok((filter_spec, hw_format, out_width, out_height))
}
