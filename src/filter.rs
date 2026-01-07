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
use log::{debug, warn};
use once_cell::sync::Lazy;
use std::ffi::c_void;
use yuv::{YuvRange, YuvStandardMatrix};

/// Check if the buffer filter supports a given option (e.g., "colorspace", "range")
/// by getting the filter's AVClass and searching its options via FFmpeg's introspection API
fn buffer_filter_has_opt(opt_name: &str) -> bool {
    use ffmpeg::ffi::{av_opt_find, avfilter_get_by_name, AV_OPT_SEARCH_FAKE_OBJ};
    use std::ffi::CString;

    unsafe {
        // Get the AVFilter for "buffer" using FFmpeg's C API
        let buffer_name = match CString::new("buffer") {
            Ok(s) => s,
            Err(_) => return false,
        };
        let av_filter_ptr = avfilter_get_by_name(buffer_name.as_ptr());

        if av_filter_ptr.is_null() {
            warn!("Buffer filter not found via avfilter_get_by_name");
            return false;
        }

        // The AVFilter struct has priv_class member which is the AVClass for filter options
        let priv_class = (*av_filter_ptr).priv_class;
        if priv_class.is_null() {
            // Some filters don't have private options
            return false;
        }

        // Create a mutable pointer variable to hold the priv_class pointer
        // av_opt_find with AV_OPT_SEARCH_FAKE_OBJ expects a pointer-to-pointer (AVClass**)
        let mut priv_class_ptr = priv_class;

        // Now search for the option in the filter's AVClass
        let opt_name_c = match CString::new(opt_name) {
            Ok(s) => s,
            Err(_) => return false,
        };

        // Pass the address of priv_class_ptr (which gives us AVClass**)
        let av_option = av_opt_find(
            &mut priv_class_ptr as *mut _ as *mut c_void,
            opt_name_c.as_ptr(),
            std::ptr::null(),
            0,
            AV_OPT_SEARCH_FAKE_OBJ as i32,
        );

        !av_option.is_null()
    }
}

/// Cached result of whether buffer filter supports colorspace and range options (FFmpeg >= 7.0)
/// Tested once at first use via FFmpeg option introspection
static BUFFER_COLORSPACE_SUPPORT: Lazy<(bool, bool)> = Lazy::new(|| {
    let supports_colorspace = buffer_filter_has_opt("colorspace");
    let supports_range = buffer_filter_has_opt("range");

    if !supports_colorspace || !supports_range {
        debug!(
            "Buffer filter option support: colorspace={}, range={} (FFmpeg < 7.0 may lack these)",
            supports_colorspace, supports_range
        );
    }

    (supports_colorspace, supports_range)
});

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

    let mut args = format!(
        "video_size={}x{}:pix_fmt={}:time_base={}:pixel_aspect={}/{}",
        filter_cfg.width,
        filter_cfg.height,
        pix_fmt_name,
        filter_cfg.time_base,
        filter_cfg.pixel_aspect.numerator(),
        filter_cfg.pixel_aspect.denominator(),
    );

    // Conditionally add colorspace if supported (FFmpeg >= 7.0)
    let (supports_colorspace, supports_range) = *BUFFER_COLORSPACE_SUPPORT;
    if supports_colorspace {
        args.push_str(&format!(":colorspace={}", colorspace_str));
    }

    // Conditionally add range if supported (FFmpeg >= 7.0)
    if supports_range {
        args.push_str(&format!(":range={}", range_str));
    }

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

/// Parse scale dimensions from filter string.
/// Supports all FFmpeg scale formats:
/// - Positional: "scale=256:256", "scale=256:-1"
/// - Named w first: "scale=w=256:h=256:flags=bicubic"
/// - Named h first: "scale=h=256:w=256"
/// - Long names: "scale=width=256:height=256"
/// Returns (width, height) if both are valid integers, None otherwise.
fn parse_scale_from_filter(filter_spec: &str) -> Option<(u32, u32)> {
    let scale_pos = filter_spec.find("scale=")?;
    let scale_str = &filter_spec[scale_pos + 6..];
    let scale_end = scale_str.find(',').unwrap_or(scale_str.len());
    parse_scale_params(&scale_str[..scale_end])
}

/// Parse width and height from scale parameter string (after "scale=")
fn parse_scale_params(scale_params: &str) -> Option<(u32, u32)> {
    let mut width: Option<u32> = None;
    let mut height: Option<u32> = None;

    for (i, part) in scale_params.split(':').enumerate() {
        if let Some(val) = part
            .strip_prefix("w=")
            .or_else(|| part.strip_prefix("width="))
        {
            width = val.parse().ok();
        } else if let Some(val) = part
            .strip_prefix("h=")
            .or_else(|| part.strip_prefix("height="))
        {
            height = val.parse().ok();
        } else if !part.contains('=') {
            if i == 0 {
                width = part.parse().ok();
            } else if i == 1 {
                height = part.parse().ok();
            }
        }
    }

    match (width, height) {
        (Some(w), Some(h)) => Some((w, h)),
        _ => None,
    }
}

/// Detect if scale uses named parameters (w=, h=, width=, height=) vs positional
fn scale_uses_named_params(scale_params: &str) -> bool {
    scale_params.contains("w=")
        || scale_params.contains("h=")
        || scale_params.contains("width=")
        || scale_params.contains("height=")
}

/// Swap w and h values in a filter string's scale parameter
fn swap_scale_dimensions(filter_spec: &str) -> String {
    let Some((w, h)) = parse_scale_from_filter(filter_spec) else {
        return filter_spec.to_string();
    };
    let Some(scale_pos) = filter_spec.find("scale=") else {
        return filter_spec.to_string();
    };

    let before = &filter_spec[..scale_pos];
    let after_scale = &filter_spec[scale_pos + 6..];
    let scale_end = after_scale.find(',').unwrap_or(after_scale.len());
    let scale_params = &after_scale[..scale_end];
    let rest = &after_scale[scale_end..];

    if scale_uses_named_params(scale_params) {
        let new_parts: Vec<String> = scale_params
            .split(':')
            .map(|part| {
                if part.starts_with("w=") {
                    format!("w={}", h)
                } else if part.starts_with("h=") {
                    format!("h={}", w)
                } else if part.starts_with("width=") {
                    format!("width={}", h)
                } else if part.starts_with("height=") {
                    format!("height={}", w)
                } else {
                    part.to_string()
                }
            })
            .collect();
        format!("{}scale={}{}", before, new_parts.join(":"), rest)
    } else {
        let parts: Vec<&str> = scale_params.split(':').collect();
        if parts.len() >= 2 {
            let mut new_parts = vec![h.to_string(), w.to_string()];
            new_parts.extend(parts[2..].iter().map(|s| s.to_string()));
            format!("{}scale={}{}", before, new_parts.join(":"), rest)
        } else {
            filter_spec.to_string()
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== parse_scale_from_filter tests ====================

    #[test]
    fn test_parse_positional_format() {
        // scale=WIDTH:HEIGHT
        assert_eq!(parse_scale_from_filter("scale=256:256"), Some((256, 256)));
        assert_eq!(
            parse_scale_from_filter("scale=1920:1080"),
            Some((1920, 1080))
        );
        assert_eq!(parse_scale_from_filter("scale=640:480"), Some((640, 480)));
    }

    #[test]
    fn test_parse_named_w_first() {
        // scale=w=XXX:h=YYY
        assert_eq!(
            parse_scale_from_filter("scale=w=256:h=256"),
            Some((256, 256))
        );
        assert_eq!(
            parse_scale_from_filter("scale=w=1920:h=1080"),
            Some((1920, 1080))
        );
    }

    #[test]
    fn test_parse_named_h_first() {
        // scale=h=XXX:w=YYY (the case that was missing!)
        assert_eq!(
            parse_scale_from_filter("scale=h=256:w=256"),
            Some((256, 256))
        );
        assert_eq!(
            parse_scale_from_filter("scale=h=1080:w=1920"),
            Some((1920, 1080))
        );
    }

    #[test]
    fn test_parse_long_names() {
        // scale=width=XXX:height=YYY
        assert_eq!(
            parse_scale_from_filter("scale=width=256:height=256"),
            Some((256, 256))
        );
        assert_eq!(
            parse_scale_from_filter("scale=height=1080:width=1920"),
            Some((1920, 1080))
        );
    }

    #[test]
    fn test_parse_with_flags() {
        // scale with extra parameters
        assert_eq!(
            parse_scale_from_filter("scale=w=256:h=256:flags=bicubic"),
            Some((256, 256))
        );
        assert_eq!(
            parse_scale_from_filter("scale=256:256:flags=lanczos"),
            Some((256, 256))
        );
        assert_eq!(
            parse_scale_from_filter("scale=h=480:w=640:flags=bilinear"),
            Some((640, 480))
        );
    }

    #[test]
    fn test_parse_in_filter_chain() {
        // scale in a larger filter spec
        assert_eq!(
            parse_scale_from_filter("format=rgb24,scale=256:256"),
            Some((256, 256))
        );
        assert_eq!(
            parse_scale_from_filter("scale=w=640:h=480,transpose=1"),
            Some((640, 480))
        );
        assert_eq!(
            parse_scale_from_filter("format=yuv420p,scale=h=720:w=1280,format=rgb24"),
            Some((1280, 720))
        );
    }

    #[test]
    fn test_parse_invalid_cases() {
        // Should return None for invalid/unsupported cases
        assert_eq!(parse_scale_from_filter("scale=-1:256"), None); // negative not u32
        assert_eq!(parse_scale_from_filter("scale=iw/2:ih/2"), None); // expressions
        assert_eq!(parse_scale_from_filter("no_scale_here"), None); // no scale
        assert_eq!(parse_scale_from_filter("scale=256"), None); // only width
    }

    // ==================== swap_scale_dimensions tests ====================

    #[test]
    fn test_swap_positional() {
        assert_eq!(swap_scale_dimensions("scale=360:640"), "scale=640:360");
        assert_eq!(swap_scale_dimensions("scale=1080:1920"), "scale=1920:1080");
    }

    #[test]
    fn test_swap_named_w_first() {
        assert_eq!(
            swap_scale_dimensions("scale=w=360:h=640"),
            "scale=w=640:h=360"
        );
    }

    #[test]
    fn test_swap_named_h_first() {
        // Preserve h= first order
        assert_eq!(
            swap_scale_dimensions("scale=h=640:w=360"),
            "scale=h=360:w=640"
        );
    }

    #[test]
    fn test_swap_with_flags() {
        assert_eq!(
            swap_scale_dimensions("scale=w=360:h=640:flags=bicubic"),
            "scale=w=640:h=360:flags=bicubic"
        );
        assert_eq!(
            swap_scale_dimensions("scale=360:640:flags=lanczos"),
            "scale=640:360:flags=lanczos"
        );
    }

    #[test]
    fn test_swap_in_filter_chain() {
        assert_eq!(
            swap_scale_dimensions("format=rgb24,scale=360:640,transpose=1"),
            "format=rgb24,scale=640:360,transpose=1"
        );
        assert_eq!(
            swap_scale_dimensions("scale=h=480:w=640,format=rgb24"),
            "scale=h=640:w=480,format=rgb24"
        );
    }

    #[test]
    fn test_swap_no_scale() {
        // No change if no scale found
        assert_eq!(swap_scale_dimensions("format=rgb24"), "format=rgb24");
    }

    #[test]
    fn test_swap_long_names() {
        assert_eq!(
            swap_scale_dimensions("scale=width=360:height=640"),
            "scale=width=640:height=360"
        );
    }
}
