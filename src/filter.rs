use crate::decoder::ResizeAlgo;
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
use yuv::{YuvRange, YuvStandardMatrix};

pub struct FilterConfig<'a> {
    height: u32,
    width: u32,
    vid_format: ffmpeg::util::format::Pixel,
    time_base: &'a str,
    spec: &'a str,
    is_hwaccel: bool,
    pixel_aspect: Rational,
    color_space: YuvStandardMatrix,
    color_range: YuvRange,
}
impl<'a> FilterConfig<'a> {
    pub fn new(
        height: u32,
        width: u32,
        vid_format: ffmpeg::util::format::Pixel,
        time_base: &'a str,
        spec: &'a str,
        is_hwaccel: bool,
        pixel_aspect: Rational,
        color_space: YuvStandardMatrix,
        color_range: YuvRange,
    ) -> Self {
        FilterConfig {
            height,
            width,
            vid_format,
            time_base,
            spec,
            is_hwaccel,
            pixel_aspect,
            color_space,
            color_range,
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

    // Map YuvStandard Matrix to FFmpeg colorspace name
    let colorspace_str = match filter_cfg.color_space {
        YuvStandardMatrix::Bt709 => "bt709",
        YuvStandardMatrix::Bt601 => "smpte170m", // BT601 = SMPTE170M
        YuvStandardMatrix::Bt2020 => "bt2020nc",
        YuvStandardMatrix::Smpte240 => "smpte240m",
        YuvStandardMatrix::Bt470_6 => "bt470bg",
        YuvStandardMatrix::Fcc => "fcc",
        YuvStandardMatrix::Custom(_, _) => "bt709", // Fallback for custom
    };

    // Map YuvRange to FFmpeg range name
    let range_str = match filter_cfg.color_range {
        YuvRange::Limited => "tv", // MPEG range
        YuvRange::Full => "pc",    // JPEG range
    };

    let args = format!(
        "video_size={}x{}:pix_fmt={}:time_base={}:pixel_aspect={}{}:colorspace={}:range={}",
        filter_cfg.width,
        filter_cfg.height,
        pix_fmt_name,
        filter_cfg.time_base,
        filter_cfg.pixel_aspect.numerator(),
        filter_cfg.pixel_aspect.denominator(),
        colorspace_str,
        range_str,
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
        // Rotate twice to achieve 180 degrees
        180 | -180 => ",transpose=1,transpose=1".to_owned(),
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

/// Swap w and h values in a filter string's scale parameter
/// e.g., "scale=w=360:h=640" -> "scale=w=640:h=360"
fn swap_scale_dimensions(filter_spec: &str) -> String {
    // Parse scale dimensions first
    if let Some((w, h)) = parse_scale_from_filter(filter_spec) {
        // Find and replace the scale parameter
        if let Some(scale_pos) = filter_spec.find("scale=") {
            let before = &filter_spec[..scale_pos];
            let after_scale = &filter_spec[scale_pos + 6..];

            // Find end of scale params (next comma or end)
            let scale_end = after_scale.find(',').unwrap_or(after_scale.len());
            let rest = &after_scale[scale_end..];

            // Check format and rebuild with swapped dimensions
            if after_scale.starts_with("w=") {
                // "scale=w=XXX:h=YYY:flags=..." format - preserve extra params like flags
                let scale_parts: Vec<&str> = after_scale[..scale_end].split(':').collect();
                let mut new_parts = Vec::new();
                for part in &scale_parts {
                    if part.starts_with("w=") {
                        new_parts.push(format!("w={}", h));
                    } else if part.starts_with("h=") {
                        new_parts.push(format!("h={}", w));
                    } else {
                        new_parts.push(part.to_string());
                    }
                }
                return format!("{}scale={}{}", before, new_parts.join(":"), rest);
            } else {
                // "scale=XXX:YYY" format
                return format!("{}scale={}:{}{}", before, h, w, rest);
            }
        }
    }
    // No scale found or parse failed, return as-is
    filter_spec.to_string()
}

pub fn create_filter_spec(
    width: u32,
    height: u32,
    video: &mut ffmpeg::decoder::Video,
    ff_filter: Option<String>,
    hwaccel_context: Option<HardwareAccelerationContext>,
    hwaccel_fmt: AvPixel,
    rotation: isize,
    resize_algo: ResizeAlgo,
) -> Result<(String, Option<AvPixel>, u32, u32, bool), ffmpeg_next::Error> {
    let pix_fmt = AvPixel::YUV420P;
    let mut hw_format: Option<AvPixel> = None;
    let mut out_width = width;
    let mut out_height = height;
    let rotation_applied: bool;

    let filter_spec = match ff_filter {
        None => {
            let pix_fmt_name = pix_fmt.descriptor().map(|d| d.name()).unwrap_or("yuv420p");
            let transpose = transpose_filter(rotation);
            rotation_applied = !transpose.is_empty();

            // For 90/270 rotation, scale happens before transpose
            // The user's width/height are already in display orientation (post-transpose)
            // So we swap dimensions for scale, but keep out_width/out_height as user specified
            let (scale_w, scale_h) =
                if rotation_applied && (rotation.abs() == 90 || rotation.abs() == 270) {
                    // Scale uses swapped dimensions because transpose will swap them back
                    // out_width/out_height stay as user specified (already in display orientation)
                    (height, width)
                } else {
                    (width, height)
                };

            // Order: format -> scale -> transpose
            let mut filter_spec = format!(
                "format={},scale=w={}:h={}:flags={}{}",
                pix_fmt_name,
                scale_w,
                scale_h,
                resize_algo.to_ffmpeg_flag(),
                transpose,
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
                    scale_w, scale_h, hwaccel_fmt_name, transpose,
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
            let has_transpose = spec.to_lowercase().contains("transpose");
            if has_transpose {
                rotation_applied = true; // Assume user handled rotation; we won't alter spec
                spec
            } else {
                let transpose = transpose_filter(rotation);
                rotation_applied = !transpose.is_empty();
                // If rotation is 90/270, user expects filter dimensions relative to rotated display
                // But scale happens before transpose, so we need to swap w/h in the filter
                let adjusted_spec =
                    if rotation_applied && (rotation.abs() == 90 || rotation.abs() == 270) {
                        swap_scale_dimensions(&spec)
                    } else {
                        spec
                    };
                // out_width/out_height remain as user specified (the final output after transpose)
                // No need to swap them since they represent what user expects
                // Append transpose at the end
                format!("{}{}", adjusted_spec, transpose)
            }
        }
    };
    Ok((
        filter_spec,
        hw_format,
        out_width,
        out_height,
        rotation_applied,
    ))
}
