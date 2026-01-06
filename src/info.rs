use ffmpeg::codec::packet::side_data::Type as SideDataType;
use ffmpeg::ffi::av_display_rotation_get;

use ffmpeg::util::rational::Rational;
use ffmpeg::Stream;
use ffmpeg_next as ffmpeg;
use std::collections::{BTreeMap, HashMap};

pub struct VideoParams {
    duration: f64,
    start_time: i64,
    time_base: f64,
    time_base_rational: Rational,
    pub rotation: isize,
}

/// Timing info for key frames
#[derive(Debug)]
pub struct FrameTime {
    pts: i64,
    #[allow(dead_code)]
    dur: i64,
    dts: i64,
    has_pts: bool,
    has_dts: bool,
}

impl FrameTime {
    pub fn new(pts: i64, dur: i64, dts: i64, has_pts: bool, has_dts: bool) -> Self {
        FrameTime {
            pts,
            dur,
            dts,
            has_pts,
            has_dts,
        }
    }
    pub fn pts(&self) -> &i64 {
        &self.pts
    }
    pub fn dts(&self) -> &i64 {
        &self.dts
    }
    #[allow(dead_code)]
    pub fn dur(&self) -> &i64 {
        &self.dur
    }
    pub fn has_pts(&self) -> bool {
        self.has_pts
    }
    pub fn has_dts(&self) -> bool {
        self.has_dts
    }
}

/// Info gathered from iterating over the video stream
pub struct StreamInfo {
    frame_count: usize,
    key_frames: Vec<usize>,
    frame_times: BTreeMap<usize, FrameTime>,
    /// Mapping from presentation index to decode index (packet order)
    /// Used to find which keyframe to seek to for a given presentation index
    presentation_to_decode_idx: Vec<usize>,
    /// Mapping from decode index to presentation index
    /// Used to find which presentation frame we're at after seeking to a keyframe
    decode_to_presentation_idx: Vec<usize>,
    /// Whether the video has negative PTS values
    /// Whether video has negative PTS frames (seek may work with flag=0 depending on mapping)
    #[allow(dead_code)]
    has_negative_pts: bool,
    /// Whether video has negative DTS frames (seek may work with flag=0)
    #[allow(dead_code)]
    has_negative_dts: bool,
    /// Whether any packet is missing PTS/DTS
    has_missing_pts: bool,
    has_missing_dts: bool,
    /// Whether packet-order PTS/DTS are non-monotonic
    #[allow(dead_code)]
    has_non_monotonic_pts: bool,
    has_non_monotonic_dts: bool,
    /// Whether any packets share the same PTS/DTS values
    has_duplicate_pts: bool,
    has_duplicate_dts: bool,
    /// Minimum raw pts/dts observed (for normalization)
    min_pts: i64,
    min_dts: i64,
    /// Mapping from normalized PTS to presentation index (when PTS exists)
    pts_to_pres_idx: BTreeMap<i64, usize>,
}

impl StreamInfo {
    pub fn new(
        frame_count: usize,
        key_frames: Vec<usize>,
        frame_times: BTreeMap<usize, FrameTime>,
    ) -> Self {
        let mut min_pts = i64::MAX;
        let mut min_dts = i64::MAX;
        let mut has_missing_pts = false;
        let mut has_missing_dts = false;
        let mut has_non_monotonic_pts = false;
        let mut has_non_monotonic_dts = false;
        let mut has_duplicate_pts = false;
        let mut has_duplicate_dts = false;
        let mut pts_to_pres_idx: BTreeMap<i64, usize> = BTreeMap::new();
        let mut seen_pts: std::collections::HashSet<i64> = std::collections::HashSet::new();
        let mut seen_dts: std::collections::HashSet<i64> = std::collections::HashSet::new();

        let mut prev_pts: Option<i64> = None;
        let mut prev_dts: Option<i64> = None;

        for ft in frame_times.values() {
            if ft.has_pts() {
                let pts = *ft.pts();
                min_pts = min_pts.min(pts);
                if let Some(pp) = prev_pts {
                    if pts < pp {
                        has_non_monotonic_pts = true;
                    }
                }
                prev_pts = Some(pts);
                // Check for duplicate PTS
                if !seen_pts.insert(pts) {
                    has_duplicate_pts = true;
                }
            } else {
                has_missing_pts = true;
            }

            if ft.has_dts() {
                let dts = *ft.dts();
                min_dts = min_dts.min(dts);
                if let Some(pd) = prev_dts {
                    if dts < pd {
                        has_non_monotonic_dts = true;
                    }
                }
                prev_dts = Some(dts);
                // Check for duplicate DTS
                if !seen_dts.insert(dts) {
                    has_duplicate_dts = true;
                }
            } else {
                has_missing_dts = true;
            }
        }

        if min_pts == i64::MAX {
            min_pts = 0;
        }
        if min_dts == i64::MAX {
            min_dts = 0;
        }

        let pts_offset = if min_pts < 0 { min_pts } else { 0 };

        // Build presentation order mapping by sorting by PTS
        // For B-frame videos, decode order != presentation order
        let mut pts_sorted: Vec<(i64, usize)> = frame_times
            .iter()
            .map(|(decode_idx, ft)| {
                // Use normalized PTS if available; fallback to decode_idx to keep mapping stable
                let normalized_pts = if ft.has_pts() {
                    ft.pts - pts_offset
                } else {
                    *decode_idx as i64
                };
                (normalized_pts, *decode_idx)
            })
            .collect();
        pts_sorted.sort_by_key(|(pts, _)| *pts);

        // presentation_idx -> decode_idx
        let presentation_to_decode_idx: Vec<usize> = pts_sorted
            .iter()
            .map(|(_, decode_idx)| *decode_idx)
            .collect();

        // decode_idx -> presentation_idx (reverse mapping)
        let mut decode_to_presentation_idx = vec![0usize; frame_count];
        for (pres_idx, &decode_idx) in presentation_to_decode_idx.iter().enumerate() {
            if decode_idx < frame_count {
                decode_to_presentation_idx[decode_idx] = pres_idx;
            }
        }

        let has_negative_pts = min_pts < 0;
        let has_negative_dts = min_dts < 0;

        // Build pts -> presentation index map (only when PTS exists)
        for (pres_idx, decode_idx) in presentation_to_decode_idx.iter().enumerate() {
            if let Some(ft) = frame_times.get(decode_idx) {
                if ft.has_pts() {
                    let norm = ft.pts - pts_offset;
                    pts_to_pres_idx.insert(norm, pres_idx);
                }
            }
        }

        StreamInfo {
            frame_count,
            key_frames,
            frame_times,
            presentation_to_decode_idx,
            decode_to_presentation_idx,
            has_negative_pts,
            has_negative_dts,
            has_missing_pts,
            has_missing_dts,
            has_non_monotonic_pts,
            has_non_monotonic_dts,
            has_duplicate_pts,
            has_duplicate_dts,
            min_pts,
            min_dts,
            pts_to_pres_idx,
        }
    }
    pub fn frame_count(&self) -> &usize {
        &self.frame_count
    }
    pub fn key_frames(&self) -> &Vec<usize> {
        &self.key_frames
    }
    pub fn frame_times(&self) -> &BTreeMap<usize, FrameTime> {
        &self.frame_times
    }
    /// Get the decode index (packet order) for a presentation index
    #[allow(dead_code)]
    pub fn get_decode_idx_for_presentation(&self, presentation_idx: usize) -> Option<usize> {
        self.presentation_to_decode_idx
            .get(presentation_idx)
            .copied()
    }
    /// Get the presentation index for a decode index (packet order)
    /// Used to find which presentation frame a keyframe corresponds to
    #[allow(dead_code)]
    pub fn get_presentation_idx_for_decode(&self, decode_idx: usize) -> Option<usize> {
        self.decode_to_presentation_idx.get(decode_idx).copied()
    }
    pub fn get_pts_for_presentation(&self, pres_idx: usize) -> Option<(i64, i64)> {
        if pres_idx >= self.presentation_to_decode_idx.len() {
            return None;
        }
        let decode_idx = self.presentation_to_decode_idx[pres_idx];
        self.frame_times.get(&decode_idx).map(|ft| {
            let norm = if ft.has_pts() {
                ft.pts - self.min_pts_offset()
            } else {
                ft.pts
            };
            (ft.pts, norm)
        })
    }
    /// Check if video has negative PTS values
    #[allow(dead_code)]
    pub fn has_negative_pts(&self) -> bool {
        self.has_negative_pts
    }
    /// Check if video has negative DTS values (but positive PTS)
    #[allow(dead_code)]
    pub fn has_negative_dts(&self) -> bool {
        self.has_negative_dts
    }
    pub fn has_missing_pts(&self) -> bool {
        self.has_missing_pts
    }
    pub fn has_missing_dts(&self) -> bool {
        self.has_missing_dts
    }
    #[allow(dead_code)]
    pub fn has_non_monotonic_pts(&self) -> bool {
        self.has_non_monotonic_pts
    }
    pub fn has_non_monotonic_dts(&self) -> bool {
        self.has_non_monotonic_dts
    }
    pub fn has_duplicate_pts(&self) -> bool {
        self.has_duplicate_pts
    }
    pub fn has_duplicate_dts(&self) -> bool {
        self.has_duplicate_dts
    }
    pub fn min_pts_offset(&self) -> i64 {
        if self.min_pts < 0 {
            self.min_pts
        } else {
            0
        }
    }
    pub fn min_dts_offset(&self) -> i64 {
        if self.min_dts < 0 {
            self.min_dts
        } else {
            0
        }
    }
    /// Get the presentation index offset of the first keyframe.
    /// If first keyframe (decode_idx=0) has pres_idx > 0, it means there are
    /// frames that should be displayed before the first keyframe but cannot be decoded
    /// (typically B-frames at the start of a video with missing references).
    /// Returns 0 for normal videos, > 0 for videos with this issue.
    pub fn first_keyframe_pres_offset(&self) -> usize {
        if self.key_frames.is_empty() {
            return 0;
        }
        let first_keyframe_decode_idx = self.key_frames[0];
        // Get the presentation index for the first keyframe
        self.decode_to_presentation_idx
            .get(first_keyframe_decode_idx)
            .copied()
            .unwrap_or(0)
    }
    /// Check if first keyframe has negative PTS.
    /// When first keyframe has negative PTS, it typically indicates the video
    /// starts with B-frames that depend on reference frames decoded later.
    /// These B-frames often cannot be decoded properly, causing packet-frame
    /// count mismatch and presentation index misalignment.
    pub fn first_keyframe_has_negative_pts(&self) -> bool {
        if self.key_frames.is_empty() {
            return false;
        }
        let first_keyframe_decode_idx = self.key_frames[0];
        if let Some(ft) = self.frame_times.get(&first_keyframe_decode_idx) {
            ft.pts < 0
        } else {
            false
        }
    }
    /// Detect if video has Open GOP structure.
    /// Open GOP means some keyframes have B-frames that depend on the previous GOP.
    /// This is detected when a keyframe's presentation index is greater than
    /// some frames that come AFTER it in decode order (i.e., those frames display before the keyframe).
    /// Returns (has_open_gop, number of open GOP keyframes).
    #[allow(dead_code)]
    pub fn has_open_gop(&self) -> (bool, usize) {
        let mut open_gop_count = 0usize;

        for i in 0..self.key_frames.len() {
            let kf_decode_idx = self.key_frames[i];
            let kf_pres_idx = match self.decode_to_presentation_idx.get(kf_decode_idx) {
                Some(&idx) => idx,
                None => continue,
            };

            // Check next few frames after this keyframe (in decode order)
            // If any of them have lower presentation index, this is Open GOP
            let next_kf_decode_idx = if i + 1 < self.key_frames.len() {
                self.key_frames[i + 1]
            } else {
                self.frame_count
            };

            for decode_idx in (kf_decode_idx + 1)..next_kf_decode_idx.min(kf_decode_idx + 32) {
                if let Some(&pres_idx) = self.decode_to_presentation_idx.get(decode_idx) {
                    if pres_idx < kf_pres_idx {
                        // Found a frame that displays before the keyframe but is decoded after
                        open_gop_count += 1;
                        break; // Count each keyframe only once
                    }
                }
            }
        }

        (open_gop_count > 0, open_gop_count)
    }
    /// Check if a specific keyframe is Closed GOP (no B-frames after it that display before it)
    pub fn is_closed_gop_keyframe(&self, kf_decode_idx: usize) -> bool {
        let kf_pres_idx = match self.decode_to_presentation_idx.get(kf_decode_idx) {
            Some(&idx) => idx,
            None => return true, // Assume closed if no info
        };

        // Find next keyframe's decode index
        let kf_pos = self.key_frames.iter().position(|&kf| kf == kf_decode_idx);
        let next_kf_decode_idx = match kf_pos {
            Some(pos) if pos + 1 < self.key_frames.len() => self.key_frames[pos + 1],
            _ => self.frame_count,
        };

        // Check if any frame after this keyframe has lower pres_idx
        for decode_idx in (kf_decode_idx + 1)..next_kf_decode_idx.min(kf_decode_idx + 32) {
            if let Some(&pres_idx) = self.decode_to_presentation_idx.get(decode_idx) {
                if pres_idx < kf_pres_idx {
                    return false; // This is Open GOP
                }
            }
        }
        true
    }

    /// Find a safe keyframe for seeking to the target presentation index.
    /// For Open GOP, this may return a keyframe earlier than the one containing the target,
    /// ensuring all reference frames are available.
    /// Returns (keyframe_decode_idx, keyframe_pres_idx, min_pres_idx_in_gop)
    pub fn find_safe_keyframe_for_pres_idx(
        &self,
        target_pres_idx: usize,
    ) -> Option<(usize, usize, usize)> {
        // Get decode index for target
        let target_decode_idx = self.presentation_to_decode_idx.get(target_pres_idx)?;

        // Find keyframe index (position in key_frames array)
        // IMPORTANT: When target is itself a keyframe (Ok case), we still use the PREVIOUS keyframe.
        // This is because after seeking to a keyframe, the packet iterator may not have any packets
        // ready to send (FFmpeg seek behavior). Using previous keyframe ensures at least one packet
        // (the target keyframe itself) is available to send to the decoder.
        let mut kf_array_idx = match self.key_frames.binary_search(target_decode_idx) {
            Ok(0) => 0,            // Target is first keyframe: must use it
            Ok(idx) => idx - 1,    // Target is keyframe: use previous keyframe
            Err(0) => return None, // Before first keyframe
            Err(idx) => idx - 1,   // Keyframe before target
        };

        // Walk backwards until we find a Closed GOP keyframe
        // or until the keyframe's pres_idx <= target_pres_idx
        loop {
            let kf_decode_idx = self.key_frames[kf_array_idx];
            let kf_pres_idx = self
                .decode_to_presentation_idx
                .get(kf_decode_idx)
                .copied()
                .unwrap_or(kf_decode_idx);

            // Find minimum pres_idx in this GOP (for B-frames before keyframe)
            let next_kf_decode = if kf_array_idx + 1 < self.key_frames.len() {
                self.key_frames[kf_array_idx + 1]
            } else {
                self.frame_count
            };

            let mut min_pres_in_gop = kf_pres_idx;
            for decode_idx in (kf_decode_idx + 1)..next_kf_decode.min(kf_decode_idx + 32) {
                if let Some(&pres_idx) = self.decode_to_presentation_idx.get(decode_idx) {
                    min_pres_in_gop = min_pres_in_gop.min(pres_idx);
                }
            }

            // If this keyframe can reach our target (its GOP contains the target),
            // and either it's Closed GOP or we can decode from here with all refs
            if min_pres_in_gop <= target_pres_idx && self.is_closed_gop_keyframe(kf_decode_idx) {
                return Some((kf_decode_idx, kf_pres_idx, min_pres_in_gop));
            }

            // If target is >= keyframe's pres_idx, this keyframe is safe
            // (we don't need B-frames from before the keyframe)
            if target_pres_idx >= kf_pres_idx {
                return Some((kf_decode_idx, kf_pres_idx, min_pres_in_gop));
            }

            // Need to go to previous keyframe for proper reference frames
            if kf_array_idx == 0 {
                // Already at first keyframe, use it
                return Some((kf_decode_idx, kf_pres_idx, min_pres_in_gop));
            }
            kf_array_idx -= 1;
        }
    }
    #[allow(dead_code)]
    pub fn keyframe_pts_monotonic_norm(&self) -> (bool, usize) {
        let offset = self.min_pts_offset();
        let key_frames = self.key_frames();
        let frame_times = self.frame_times();
        let mut violation = 0usize;
        for pair in key_frames.windows(2) {
            if let (Some(a), Some(b)) = (frame_times.get(&pair[0]), frame_times.get(&pair[1])) {
                if !(a.has_pts() && b.has_pts()) {
                    continue;
                }
                let pa = a.pts - offset;
                let pb = b.pts - offset;
                if pb < pa {
                    violation += 1;
                }
            }
        }
        (violation == 0, violation)
    }
    #[allow(dead_code)]
    pub fn keyframe_dts_monotonic(&self) -> (bool, usize) {
        let key_frames = self.key_frames();
        let frame_times = self.frame_times();
        let mut violation = 0usize;
        for pair in key_frames.windows(2) {
            if let (Some(a), Some(b)) = (frame_times.get(&pair[0]), frame_times.get(&pair[1])) {
                if !(a.has_dts() && b.has_dts()) {
                    continue;
                }
                if b.dts < a.dts {
                    violation += 1;
                }
            }
        }
        (violation == 0, violation)
    }
    pub fn presentation_for_pts_norm(&self, norm_pts: i64) -> Option<usize> {
        self.pts_to_pres_idx.get(&norm_pts).copied()
    }
    pub fn get_all_pts(&self, time_base: f64) -> Vec<f64> {
        self.frame_times
            .values()
            .map(|v| v.pts as f64 * time_base)
            .collect::<Vec<_>>()
    }
    pub fn get_pts(&self, indices: &[usize], time_base: f64) -> Vec<f64> {
        indices
            .iter()
            .map(|i| match self.frame_times.get(i) {
                None => -1.,
                Some(ft) => ft.pts as f64 * time_base,
            })
            .collect::<Vec<_>>()
    }
}

fn get_rotation_angle(input: &Stream) -> isize {
    for sd in input.side_data() {
        if sd.kind() == SideDataType::DisplayMatrix {
            let matrix = sd.data();
            let theta = unsafe { av_display_rotation_get(matrix.as_ptr() as *const _) };
            if theta.is_nan() {
                return 0_isize;
            } else {
                return theta.round() as isize;
            }
        }
    }
    0_isize
}

pub fn extract_video_params(input: &Stream) -> VideoParams {
    VideoParams {
        duration: input.duration() as f64 * f64::from(input.time_base()),
        start_time: input.start_time(),
        time_base: f64::from(input.time_base()),
        time_base_rational: input.time_base(),
        rotation: get_rotation_angle(input),
    }
}

pub fn collect_video_metadata(
    video: &ffmpeg::decoder::Video,
    params: &VideoParams,
    fps: &f64,
) -> HashMap<&'static str, String> {
    let mut info = HashMap::new();

    let fps_rational = video.frame_rate().unwrap_or(Rational(0, 1));
    info.insert("fps_rational", fps_rational.to_string());
    info.insert("fps", format!("{fps}"));
    info.insert("start_time", params.start_time.to_string());
    info.insert("time_base", params.time_base.to_string());
    info.insert("time_base_rational", params.time_base_rational.to_string());
    info.insert("duration", params.duration.to_string());
    let codec_id = video
        .codec()
        .map(|c| format!("{:?}", c.id()))
        .unwrap_or_else(|| "UNKNOWN".to_string());
    info.insert("codec_id", codec_id);
    info.insert("height", video.height().to_string());
    info.insert("width", video.width().to_string());
    info.insert("bit_rate", video.bit_rate().to_string());
    info.insert("vid_format", format!("{:?}", video.format()));
    info.insert("aspect_ratio", format!("{:?}", video.aspect_ratio()));
    let color_space = format!("{:?}", video.color_space());
    info.insert("color_space", color_space.to_uppercase());
    let color_range = format!("{:?}", video.color_range());
    info.insert("color_range", color_range.to_uppercase());
    info.insert("color_primaries", format!("{:?}", video.color_primaries()));
    info.insert(
        "color_xfer_charac",
        format!("{:?}", video.color_transfer_characteristic()),
    );
    info.insert("chroma_location", format!("{:?}", video.chroma_location()));
    info.insert("vid_ref", video.references().to_string());
    info.insert("intra_dc_precision", video.intra_dc_precision().to_string());
    info.insert("has_b_frames", format!("{}", video.has_b_frames()));
    info.insert("rotation", format!("{}", params.rotation));
    info
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
            let pts_opt = packet.pts();
            let dts_opt = packet.dts();
            let pts = pts_opt.unwrap_or(0);
            let dts = dts_opt.unwrap_or(0);
            let dur = packet.duration();
            frame_times.insert(
                didx,
                FrameTime::new(pts, dur, dts, pts_opt.is_some(), dts_opt.is_some()),
            );
            if packet.is_key() {
                key_frames.push(didx);
            }
            didx += 1;
        }
    }

    // Seek back to the begining of the stream
    ictx.seek(0, ..10)?;
    Ok(StreamInfo::new(didx, key_frames, frame_times))
}

/// Get the resized dimension of a frame, keep the aspect ratio.
/// Resize the shorter side of the frame, and the other side accordingly
/// so that the resizing operation is minimal. Returns a resizing dimension only
/// if the shorter side of the frame is bigger than resize_shorter_side_to.
/// * height (f64): Height of the frame
/// * width (f64): Width of the frame
/// * resize_shorter_side_to (Option<f64>): Resize the shorter side of the frame to this value.
/// * resize_longer_side_to (Option<f64>): Resize the longer side of the frame to this value.
/// * target_width (Option<u32>): Target width (takes priority over resize_shorter/longer_side).
/// * target_height (Option<u32>): Target height (takes priority over resize_shorter/longer_side).
///
/// Returns: (u32, u32): The resized height and width
pub fn get_resized_dim(
    height: f64,
    width: f64,
    resize_shorter_side_to: Option<f64>,
    resize_longer_side_to: Option<f64>,
    target_width: Option<u32>,
    target_height: Option<u32>,
) -> (u32, u32) {
    // Priority 1: If both target_width and target_height are specified, use them directly
    if let (Some(tw), Some(th)) = (target_width, target_height) {
        return (th, tw);
    }

    let mut respect_aspect_ratio = true;
    // early return when nothing to do
    if resize_shorter_side_to.is_none() && resize_longer_side_to.is_none() {
        return (height as u32, width as u32);
    } else if resize_shorter_side_to.is_some() && resize_longer_side_to.is_some() {
        respect_aspect_ratio = false;
    }

    let mut shorter_is_height: bool = true;
    let new_height: f64;
    let new_width: f64;

    if width < height {
        shorter_is_height = false;
    }

    if !respect_aspect_ratio {
        // in this case both values are Some so it should be safe to unwrap
        if let (Some(s), Some(l)) = (resize_shorter_side_to, resize_longer_side_to) {
            if shorter_is_height {
                new_height = s;
                new_width = l;
            } else {
                new_height = l;
                new_width = s;
            }
        } else {
            return (height as u32, width as u32);
        }
        return (new_height as u32, new_width as u32);
    }

    // Only case remaining is one is None and the other is Some
    if let Some(res_short_side) = resize_shorter_side_to {
        if shorter_is_height {
            new_height = res_short_side;
            new_width = (width * new_height / height).round();
        } else {
            new_width = res_short_side;
            new_height = (height * new_width / width).round();
        }
    } else {
        // resize_longer_side_to can only be Some at this point
        if let Some(res_long_side) = resize_longer_side_to {
            if shorter_is_height {
                new_width = res_long_side;
                new_height = (height * new_width / width).round();
            } else {
                new_height = res_long_side;
                new_width = (width * new_height / height).round();
            }
        } else {
            return (height as u32, width as u32);
        }
    }
    (new_height as u32, new_width as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_resize_needed() {
        let height = 720.0;
        let width = 1280.0;
        let result = get_resized_dim(height, width, None, None, None, None);

        assert_eq!(result, (720, 1280));
    }

    #[test]
    fn test_resize_shorter_side_height() {
        let height = 720.0;
        let width = 1280.0;
        let resize_to = 480.0;

        let result = get_resized_dim(height, width, Some(resize_to), None, None, None);

        assert_eq!(result.0, 480);
        assert_eq!(result.1, 853);
    }

    #[test]
    fn test_resize_shorter_side_width() {
        let height = 1080.0;
        let width = 720.0;
        let resize_to = 480.0;

        let result = get_resized_dim(height, width, Some(resize_to), None, None, None);

        assert_eq!(result.1, 480);
        assert_eq!(result.0, 720);
    }

    #[test]
    fn test_resize_longer_side_height() {
        let height = 1080.0;
        let width = 720.0;
        let resize_to = 720.0;

        let result = get_resized_dim(height, width, None, Some(resize_to), None, None);

        assert_eq!(result.0, 720);
        assert_eq!(result.1, 480);
    }

    #[test]
    fn test_resize_longer_side_width() {
        let height = 720.0;
        let width = 1280.0;
        let resize_to = 854.0;

        let result = get_resized_dim(height, width, None, Some(resize_to), None, None);

        assert_eq!(result.1, 854);
        assert_eq!(result.0, 480);
    }

    #[test]
    fn test_both_dimensions_specified() {
        let height = 720.0;
        let width = 1280.0;

        let result = get_resized_dim(height, width, Some(480.0), Some(800.0), None, None);

        // Should ignore aspect ratio and use the provided dimensions
        assert_eq!(result, (480, 800));
    }

    #[test]
    fn test_both_dimensions_specified_width_smaller() {
        let height = 1080.0;
        let width = 720.0;

        let result = get_resized_dim(height, width, Some(480.0), Some(800.0), None, None);

        // Should set width to shorter_side value and height to longer_side value
        assert_eq!(result, (800, 480));
    }

    #[test]
    fn test_resize_equal_dimensions() {
        let height = 1000.0;
        let width = 1000.0;

        let result = get_resized_dim(height, width, Some(500.0), None, None, None);

        assert_eq!(result, (500, 500));
    }

    #[test]
    fn test_rounding_behavior() {
        let height = 723.0;
        let width = 1283.0;
        let resize_to = 483.0;

        let result = get_resized_dim(height, width, Some(resize_to), None, None, None);

        assert_eq!(result.0, 483);
        // 1283 * (483/723) = 857.1 which should round to 857
        assert_eq!(result.1, 857);
    }
}
