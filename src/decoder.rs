use crate::convert::{
    convert_frame_to_ndarray_rgb24, convert_nv12_to_ndarray_rgb24, convert_yuv_to_ndarray_rgb24,
};
use crate::ffi_hwaccel::download_frame;
use crate::video_io::{FrameArray, VideoReducer};
use ffmpeg::filter;
use ffmpeg::software::scaling::context::Context;
use ffmpeg::util::frame::video::Video;
use ffmpeg_next as ffmpeg;
use ndarray::{s, Array3};
use std::collections::HashMap;

/// Struct responsible for doing the actual decoding
pub struct VideoDecoder {
    pub video: ffmpeg::decoder::Video,
    pub height: u32,
    pub width: u32,
    pub fps: f64,
    pub video_info: HashMap<&'static str, String>,
    pub is_hwaccel: bool,
    pub graph: filter::Graph,
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
                self.graph
                    .get("in")
                    .unwrap()
                    .source()
                    .add(&decoded)
                    .unwrap();

                let mut yuv_frame = Video::empty();
                if self
                    .graph
                    .get("out")
                    .unwrap()
                    .sink()
                    .frame(&mut yuv_frame)
                    .is_ok()
                {
                    let mut slice_frame =
                        reducer
                            .full_video
                            .slice_mut(s![reducer.idx_counter, .., .., ..]);

                    let rgb_frame: Array3<u8> = if self.is_hwaccel {
                        convert_nv12_to_ndarray_rgb24(yuv_frame)
                    } else {
                        convert_yuv_to_ndarray_rgb24(yuv_frame)
                    };

                    slice_frame.zip_mut_with(&rgb_frame, |a, b| {
                        *a = *b;
                    });
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
                if self.is_hwaccel {
                    decoded = download_frame(&decoded)?;
                }
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
