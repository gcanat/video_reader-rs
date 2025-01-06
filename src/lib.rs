use numpy::ndarray::Dim;
use numpy::{IntoPyArray, PyArray, PyReadonlyArray4};
mod video_io;
use log::debug;
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods, pymodule,
    types::{IntoPyDict, PyDict, PyFloat, PyList, PyModule, PyModuleMethods},
    Bound, PyResult, Python,
};
use std::sync::Mutex;
use video_io::{rgb2gray, save_video, DecoderConfig, VideoReader};

use once_cell::sync::Lazy;
use tokio::runtime::{self, Runtime};

static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap()
});

type Frame = PyArray<u8, Dim<[usize; 3]>>;

#[pyclass]
struct PyVideoReader {
    inner: Mutex<VideoReader>,
}

#[pymethods]
impl PyVideoReader {
    #[new]
    #[pyo3(signature = (filename, threads=None, resize_shorter_side=None))]
    /// create an instance of VideoReader
    /// * `filename` - path to the video file
    /// * `threads` - number of threads to use. If None, let ffmpeg choose the optimal number.
    /// * `resize_shorter_side - Optional, resize shorted side of the video to this value while
    /// preserving the aspect ratio.
    /// * returns a PyVideoReader instance.
    fn new(
        filename: &str,
        threads: Option<usize>,
        resize_shorter_side: Option<f64>,
    ) -> PyResult<Self> {
        let decoder_config = DecoderConfig::new(threads.unwrap_or(0), resize_shorter_side);
        match VideoReader::new(filename.to_string(), decoder_config) {
            Ok(reader) => Ok(PyVideoReader {
                inner: Mutex::new(reader),
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!("Error: {}", e))),
        }
    }

    /// Returns the dict with metadata information of the video. All values in the dict
    /// are strings.
    fn get_info<'a>(&'a self, py: Python<'a>) -> PyResult<Bound<'a, PyDict>> {
        match self.inner.lock() {
            Ok(vr) => {
                let mut info_dict = vr.decoder().video_info().clone();
                info_dict.insert("frame_count", vr.stream_info().frame_count().to_string());
                Ok(info_dict.into_py_dict_bound(py))
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {}", e))),
        }
    }

    /// Returns the average fps of the video as float.
    fn get_fps<'a>(&'a self, py: Python<'a>) -> PyResult<Bound<'a, PyFloat>> {
        match self.inner.lock() {
            Ok(vr) => {
                let fps = vr.decoder().fps();
                Ok(PyFloat::new_bound(py, fps))
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {}", e))),
        }
    }

    /// Get shape of the video: [number of frames, height and width]
    fn get_shape<'a>(&'a self, py: Python<'a>) -> PyResult<Bound<'a, PyList>> {
        match self.inner.lock() {
            Ok(vr) => {
                let width = vr.decoder().video().width() as usize;
                let height = vr.decoder().video().height() as usize;
                let num_frames = vr.stream_info().frame_count();
                let list = PyList::new_bound(py, [*num_frames, height, width]);
                Ok(list)
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {}", e))),
        }
    }

    #[pyo3(signature = (start_frame=None, end_frame=None, compression_factor=None))]
    /// Decode the video.
    /// * `start_frame` - optional starting index (will start decoding from this frame)
    /// * `end_frame` - optional last frame index (will stop decoding after this frame)
    /// * `compression_factor` - optional temporal compression, eg if set to 0.25, will
    /// decode 1 frame out of 4. If None, will default to 1.0, ie decoding all frames.
    /// * returns a numpy array of shape (N, H, W, C), where N is the number of frames
    fn decode<'a>(
        &'a self,
        py: Python<'a>,
        start_frame: Option<usize>,
        end_frame: Option<usize>,
        compression_factor: Option<f64>,
    ) -> PyResult<Bound<'a, PyArray<u8, Dim<[usize; 4]>>>> {
        match self.inner.lock() {
            Ok(mut reader) => match reader.decode_video(start_frame, end_frame, compression_factor)
            {
                Ok(video) => Ok(video.into_pyarray_bound(py)),
                Err(e) => Err(PyRuntimeError::new_err(format!("Error: {}", e))),
            },
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {}", e))),
        }
    }

    #[pyo3(signature = (start_frame=None, end_frame=None, compression_factor=None))]
    /// Decode the video using YUV420P format in the ffmpeg scaler followed by asynchronous
    /// YUV to RGB conversion. Can be must faster than `decode()` for High Res videos.
    /// * `start_frame` - optional starting index (will start decoding from this frame)
    /// * `end_frame` - optional last frame index (will stop decoding after this frame)
    /// * `compression_factor` - optional temporal compression, eg if set to 0.25, will
    /// decode 1 frame out of 4. If None, will default to 1.0, ie decoding all frames.
    /// * returns a list of numpy array, each ndarray being a frame.
    fn decode_fast<'a>(
        &'a self,
        py: Python<'a>,
        start_frame: Option<usize>,
        end_frame: Option<usize>,
        compression_factor: Option<f64>,
    ) -> PyResult<Vec<Bound<'a, Frame>>> {
        match self.inner.lock() {
            Ok(mut reader) => {
                let res_decode = RUNTIME.block_on(async {
                    reader
                        .decode_video_fast(start_frame, end_frame, compression_factor)
                        .await
                        .unwrap()
                });
                Ok(res_decode
                    .into_iter()
                    .map(|x| x.into_pyarray_bound(py))
                    .collect::<Vec<_>>())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {}", e))),
        }
    }

    #[pyo3(signature = (start_frame=None, end_frame=None, compression_factor=None))]
    /// Decode the video, returning grayscale frames.
    /// * `start_frame` - optional starting index (will start decoding from this frame)
    /// * `end_frame` - optional last frame index (will stop decoding after this frame)
    /// * `compression_factor` - optional temporal compression, eg if set to 0.25, will
    /// decode 1 frame out of 4. If None, will default to 1.0, ie decoding all frames.
    /// * returns a numpy array of shape (N, H, W), where N is the number of frames.
    fn decode_gray<'a>(
        &'a self,
        py: Python<'a>,
        start_frame: Option<usize>,
        end_frame: Option<usize>,
        compression_factor: Option<f64>,
    ) -> PyResult<Bound<'a, PyArray<u8, Dim<[usize; 3]>>>> {
        match self.inner.lock() {
            Ok(mut reader) => match reader.decode_video(start_frame, end_frame, compression_factor)
            {
                Ok(video) => {
                    let gray_video = rgb2gray(video);
                    Ok(gray_video.into_pyarray_bound(py))
                }
                Err(e) => Err(PyRuntimeError::new_err(format!("Error: {}", e))),
            },
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {}", e))),
        }
    }

    #[pyo3(signature = (indices, with_fallback=false))]
    /// Decodes the frames in the video corresponding to the indices in `indices`.
    /// * `indices` - list of frame index to decode.
    /// * `with_fallback` - whether to fallback to safe decoding when video has weird
    /// timestamps or B-frames.
    fn get_batch<'a>(
        &'a self,
        py: Python<'a>,
        indices: Vec<usize>,
        with_fallback: bool,
    ) -> PyResult<Bound<'a, PyArray<u8, Dim<[usize; 4]>>>> {
        match self.inner.lock() {
            Ok(mut vr) => {
                // let video: Array4<u8>;
                let start_time: i32 = vr
                    .decoder()
                    .video_info()
                    .get("start_time")
                    .unwrap()
                    .as_str()
                    .parse::<i32>()
                    .unwrap_or(0);
                let num_zero_pts = vr
                    .stream_info()
                    .frame_times()
                    .iter()
                    .filter(|(_, v)| v.pts() <= &0)
                    .collect::<Vec<_>>()
                    .len();
                let first_key_idx = vr.stream_info().key_frames()[0];
                let (_, first_key) = vr
                    .stream_info()
                    .frame_times()
                    .iter()
                    .nth(first_key_idx)
                    .unwrap();
                // Try to detect weird cases and if so switch to decoding without seeking
                // NOTE: start_time > 0 means we have B-frames. Currently `get_batch` does not guarantee
                // that we get the exact frame we want in this case, so by setting with_fallback to True
                // we can enable a more accurate method, namely `get_batch_safe`.
                if with_fallback
                    && ((num_zero_pts > 1)
                        || (first_key.dur() <= &0)
                        || (first_key.dts() < &0)
                        || start_time > 0)
                {
                    debug!("Switching to get_batch_safe!");
                    match vr.get_batch_safe(indices) {
                        Ok(batch) => Ok(batch.into_pyarray_bound(py)),
                        Err(e) => Err(PyRuntimeError::new_err(format!("Error: {}", e))),
                    }
                } else {
                    match vr.get_batch(indices) {
                        Ok(batch) => Ok(batch.into_pyarray_bound(py)),
                        Err(e) => Err(PyRuntimeError::new_err(format!("Error: {}", e))),
                    }
                }
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {}", e))),
        }
    }
}

#[pymodule]
fn video_reader<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    env_logger::init();

    // Add the VideoReader class to the module
    m.add_class::<PyVideoReader>()?;

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
