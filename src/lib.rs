use numpy::ndarray::{Array3, Array4, Dim};
use numpy::{IntoPyArray, PyArray, PyReadonlyArray4};
mod video_io;
use ffmpeg_next as ffmpeg;
use log::debug;
use pyo3::{exceptions::PyRuntimeError, pymodule, types::PyModule, Bound, PyResult, Python};
use video_io::{rgb2gray, save_video, VideoReader};

#[pymodule]
fn video_reader<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    env_logger::init();
    /// Get shape of the video: number of frames, height and width
    fn get_shape(filename: &String) -> Result<(usize, usize, usize), ffmpeg::Error> {
        let vr = VideoReader::new(filename.to_owned(), None, None, 0, false, None, None)?;
        let width = vr.decoder.decoder.width() as usize;
        let height = vr.decoder.decoder.height() as usize;
        let num_frames = vr.stream_info.frame_count;
        Ok((num_frames, height, width))
    }

    /// Decode video and return ndarray representing RGB frames
    fn decode_video(
        filename: &String,
        resize_shorter_side: Option<f64>,
        compression_factor: Option<f64>,
        threads: usize,
        start_frame: Option<usize>,
        end_frame: Option<usize>,
    ) -> Result<Array4<u8>, ffmpeg::Error> {
        let vr = VideoReader::new(
            filename.to_owned(),
            compression_factor,
            resize_shorter_side,
            threads,
            true,
            start_frame,
            end_frame,
        )?;
        vr.decode_video()
    }

    /// Decode video and return ndarray representing grayscale frames
    fn decode_video_gray(
        filename: &String,
        resize_shorter_side: Option<f64>,
        compression_factor: Option<f64>,
        threads: usize,
        start_frame: Option<usize>,
        end_frame: Option<usize>,
    ) -> Result<Array3<u8>, ffmpeg::Error> {
        let vr = VideoReader::new(
            filename.to_owned(),
            compression_factor,
            resize_shorter_side,
            threads,
            true,
            start_frame,
            end_frame,
        )?;
        let vid = vr.decode_video()?;
        let gray_vid = rgb2gray(vid);
        Ok(gray_vid)
    }

    fn get_batch(
        filename: &String,
        indices: Vec<usize>,
        resize_shorter_side: Option<f64>,
        threads: Option<usize>,
        with_fallback: bool,
    ) -> Result<Array4<u8>, ffmpeg::Error> {
        let video: Array4<u8>;
        // video reader that will use Seeking, only works in single thread mode for now
        let mut vr = VideoReader::new(
            filename.to_owned(),
            None,
            resize_shorter_side,
            1,
            false,
            None,
            None,
        )?;
        let num_zero_pts = vr
            .stream_info
            .frame_times
            .iter()
            .filter(|(_, v)| v.pts <= 0)
            .collect::<Vec<_>>()
            .len();
        let first_key = vr.stream_info.frame_times.get(&0).unwrap();
        // if fallback is enabled, try to detect weird cases and if so switch to decoding without seeking
        if with_fallback && ((num_zero_pts > 1) || (first_key.dur <= 0) || (first_key.dts < 0)) {
            debug!("Warning: video has weird timestamps, decoding without seeking");
            // switch to video reader that will iterate on all frames, we can use threading here
            vr = VideoReader::new(
                filename.to_owned(),
                Some(1.0),
                resize_shorter_side,
                threads.unwrap_or(0),
                true,
                None,
                None,
            )?;
            video = vr.get_batch_safe(indices)?;
        } else {
            video = vr.get_batch(indices)?;
        }
        Ok(video)
    }

    // wrapper of `decode_video`
    /// Decode video and return a 4D ndarray representing RGB frames with the shape (N, H, W, C)
    /// * `filename` - Path to the video file
    /// * `resize_shorter_side` - Resize the shorter side of the video to this value.
    /// * `compression_factor` - Factor for temporal compression. If None, then no compression.
    /// * `threads` - Number of threads to use for decoding. If None, let ffmpeg decide the optimal
    /// * `start_frame` - Start decoding from this frame index
    /// * `end_frame` - Stop decoding at this frame index
    /// number.
    /// * Returns a 4D ndarray with shape (N, H, W, C)
    #[pyfn(m)]
    #[pyo3(name = "decode")]
    fn decode_py<'py>(
        py: Python<'py>,
        filename: &str,
        resize_shorter_side: Option<f64>,
        compression_factor: Option<f64>,
        threads: Option<usize>,
        start_frame: Option<usize>,
        end_frame: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray<u8, Dim<[usize; 4]>>>> {
        let threads = threads.unwrap_or(0);
        let res_decode = decode_video(
            &filename.to_string(),
            resize_shorter_side,
            compression_factor,
            threads,
            start_frame,
            end_frame,
        );
        match res_decode {
            Ok(vid) => Ok(vid.into_pyarray_bound(py)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Error: {}", e))),
        }
    }

    // wrapper of `decode_video_gray`
    /// Decode video and return a 3D ndarray representing gray frames with the shape (N, H, W)
    /// * `filename` - Path to the video file
    /// * `resize_shorter_side` - Resize the shorter side of the video to this value.
    /// * `compression_factor` - Factor for temporal compression. If None, then no compression.
    /// * `threads` - Number of threads to use for decoding. If None, let ffmpeg decide the optimal
    /// * `start_frame` - Start decoding from this frame index
    /// * `end_frame` - Stop decoding at this frame index
    /// number.
    /// * Returns a 3D ndarray with shape (N, H, W)
    #[pyfn(m)]
    #[pyo3(name = "decode_gray")]
    fn decode_gray_py<'py>(
        py: Python<'py>,
        filename: &str,
        resize_shorter_side: Option<f64>,
        compression_factor: Option<f64>,
        threads: Option<usize>,
        start_frame: Option<usize>,
        end_frame: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray<u8, Dim<[usize; 3]>>>> {
        let threads = threads.unwrap_or(0);
        let res_decode = decode_video_gray(
            &filename.to_string(),
            resize_shorter_side,
            compression_factor,
            threads,
            start_frame,
            end_frame,
        );
        match res_decode {
            Ok(vid) => Ok(vid.into_pyarray_bound(py)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Error: {}", e))),
        }
    }

    /// Get a batch of frames from a video file, trying to use Seek when possible, and falling back
    /// to iterating over all frames when bad metadata is detected (eg. several frames with pts=0,
    /// frame with dts < 0, etc).
    /// * `filename` - Path to the video file
    /// * `indices` - Indices of frames to retrieve
    /// * `threads` - Number of threads to use (only usefull when falling back to safe frame
    /// iterating).
    /// * `resize_shorter_side` - Resize the shorter side of the video to this value.
    /// * `threads` - Number of threads to use for decoding. If None, let ffmpeg decide the optimal
    /// * `with_fallback` - If True, will try to detect wrong metadata in video and if so fallback
    /// to decoding without seeking. If False (or None), will always use seeking. Default is None.
    /// number.
    /// * Returns a 4D ndarray with shape (N, H, W, C)
    #[pyfn(m)]
    #[pyo3(name = "get_batch")]
    fn get_batch_py<'py>(
        py: Python<'py>,
        filename: &str,
        indices: Vec<usize>,
        threads: Option<usize>,
        resize_shorter_side: Option<f64>,
        with_fallback: Option<bool>,
    ) -> PyResult<Bound<'py, PyArray<u8, Dim<[usize; 4]>>>> {
        let vid = get_batch(
            &filename.to_string(),
            indices,
            resize_shorter_side,
            threads,
            with_fallback.unwrap_or(false),
        );
        match vid {
            Ok(vid) => Ok(vid.into_pyarray_bound(py)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Error: {}", e))),
        }
    }

    // wrapper of `get_shape`
    /// Get shape of the video: number of frames, height and width
    /// * `filename` - Path to the video file
    #[pyfn(m)]
    #[pyo3(name = "get_shape")]
    fn get_shape_py<'py>(
        py: Python<'py>,
        filename: &str,
    ) -> PyResult<Bound<'py, PyArray<usize, Dim<[usize; 1]>>>> {
        let res = get_shape(&filename.to_string());
        match res {
            Ok((num_frames, height, width)) => {
                let shape = vec![num_frames, height, width];
                Ok(shape.into_pyarray_bound(py))
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Error: {}", e))),
        }
    }

    // wrapper of `save_video`
    /// Save 4D np.ndarray of frames to video file
    /// * `ndarray` - np.ndarray of shape (N, H, W, C)
    /// * `output_filename` - Path to the output video file
    /// * `fps` - Frames per second of the output video
    /// * `codec` - Codec to use for the output video, eg "h264"
    /// * Returns None
    #[pyfn(m)]
    #[pyo3(name = "save_video")]
    fn save_video_py(
        ndarray: PyReadonlyArray4<u8>,
        output_filename: &str,
        fps: usize,
        codec: &str,
    ) -> PyResult<()> {
        let ndarray = ndarray.as_array().to_owned();
        let res = save_video(ndarray, output_filename, fps, codec);
        match res {
            Ok(_) => Ok(()),
            Err(e) => Err(PyRuntimeError::new_err(format!("Error: {}", e))),
        }
    }
    Ok(())
}
