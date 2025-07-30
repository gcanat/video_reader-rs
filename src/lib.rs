use numpy::ndarray::{Dim, IxDyn};
mod convert;
mod ffi_hwaccel;
use std::str::FromStr;
mod filter;
mod hwaccel;
mod info;
use hwaccel::HardwareAccelerationDeviceType;
use numpy::{IntoPyArray, PyArray};
mod decoder;
mod reader;
mod utils;
use convert::rgb2gray;
use decoder::DecoderConfig;
use log::debug;
use ndarray::Array;
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods, pymodule,
    types::{
        IntoPyDict, PyAnyMethods, PyDict, PyFloat, PyList, PyModule, PyModuleMethods, PySlice,
    },
    Bound, FromPyObject, PyRef, PyRefMut, PyResult, Python,
};
use reader::VideoReader;
use std::sync::Mutex;

use once_cell::sync::Lazy;
use tokio::runtime::{self, Runtime};

static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap()
});

type Frame = PyArray<u8, Dim<[usize; 3]>>;
type FrameOrVid = PyArray<u8, IxDyn>;

#[derive(FromPyObject)]
enum IntOrSlice<'py> {
    Int(i32),
    Slice(Bound<'py, PySlice>),
    IntList(Vec<i32>),
}

impl<'py> IntOrSlice<'py> {
    /// Helper function to handle indices and slices
    fn to_indices(&self, frame_count: usize) -> PyResult<Vec<usize>> {
        match self {
            IntOrSlice::Int(index) => {
                let pos_index = if *index < 0 {
                    (frame_count as i32 + index) as usize
                } else {
                    *index as usize
                };
                Ok(vec![pos_index])
            }
            IntOrSlice::Slice(slice) => {
                let start: i32 = slice.getattr("start")?.extract().unwrap_or(0_i32);
                let stop: i32 = slice
                    .getattr("stop")?
                    .extract()
                    .unwrap_or(frame_count as i32);
                let step: i32 = slice.getattr("step")?.extract().unwrap_or(1_i32);
                if ((step < 0) && (stop - start > 0)) || ((step > 0) && (stop - start < 0)) {
                    return Err(PyRuntimeError::new_err(
                        "Incompatible values in slice. step and (stop - start) must have the same sign.",
                    ));
                }
                let indices = Array::range(start as f32, stop as f32, step as f32);
                let indices = indices.mapv(|x| x as usize);
                Ok(indices.to_vec())
            }
            IntOrSlice::IntList(indices) => Ok(indices
                .iter()
                .map(|x| {
                    if x < &0 {
                        (frame_count as i32 + x) as usize
                    } else {
                        *x as usize
                    }
                })
                .collect::<Vec<_>>()),
        }
    }
}

#[pyclass]
struct PyVideoReader {
    inner: Mutex<VideoReader>,
}

#[pymethods]
impl PyVideoReader {
    #[new]
    #[pyo3(signature = (filename, threads=None, resize_shorter_side=None, resize_longer_side=None, device=None, filter=None))]
    /// create an instance of VideoReader
    /// * `filename` - path to the video file
    /// * `threads` - number of threads to use. If None, let ffmpeg choose the optimal number.
    /// * `resize_shorter_side - Optional, resize shorted side of the video to this value. If
    /// resize_longer_side is set to None, will try to preserve original aspect ratio.
    /// * `resize_longer_side - Optional, resize longer side of the video to this value. If
    /// resize_shorter_side is set to None, will try to preserve aspect ratio.
    /// * `device` - type of hardware acceleration, eg: 'cuda', 'vdpau', 'drm', etc.
    /// * `filter` - custome ffmpeg filter to use, eg "format=rgb24,scale=w=256:h=256:flags=fast_bilinear"
    /// If set to None (default) or 'cpu' then cpu is used.
    /// * returns a PyVideoReader instance.
    fn new(
        filename: &str,
        threads: Option<usize>,
        resize_shorter_side: Option<f64>,
        resize_longer_side: Option<f64>,
        device: Option<&str>,
        filter: Option<String>,
    ) -> PyResult<Self> {
        let hwaccel = match device {
            Some("cpu") | None => None,
            Some(other) => Some(HardwareAccelerationDeviceType::from_str(other).unwrap()),
        };
        let decoder_config = DecoderConfig::new(
            threads.unwrap_or(0),
            resize_shorter_side,
            resize_longer_side,
            hwaccel,
            filter,
        );
        match VideoReader::new(filename.to_string(), decoder_config) {
            Ok(reader) => Ok(PyVideoReader {
                inner: Mutex::new(reader),
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!("Error: {e}"))),
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__<'a>(
        slf: PyRefMut<'_, Self>,
        py: Python<'a>,
    ) -> Option<Bound<'a, PyArray<u8, Dim<[usize; 3]>>>> {
        slf.inner
            .lock()
            .unwrap()
            .next()
            .map(|rgb_frame| rgb_frame.into_pyarray(py))
    }

    fn __getitem__<'a>(&self, py: Python<'a>, key: IntOrSlice) -> PyResult<Bound<'a, FrameOrVid>> {
        match self.inner.lock() {
            Ok(mut vr) => {
                let frame_count = *vr.stream_info().frame_count();
                let index = key.to_indices(frame_count)?;
                let res_array = vr.get_batch(index).unwrap().into_dyn();
                // remove first dim if key was a single int
                if matches!(key, IntOrSlice::Int { .. }) {
                    Ok(res_array.squeeze().into_pyarray(py))
                } else {
                    Ok(res_array.into_pyarray(py))
                }
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {e}"))),
        }
    }

    /// Returns the number of frames in the video
    fn __len__(&self) -> PyResult<usize> {
        match self.inner.lock() {
            Ok(vr) => {
                let num_frames = vr.stream_info().frame_count().to_owned();
                Ok(num_frames)
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {e}"))),
        }
    }

    #[pyo3(signature = (index=None))]
    /// Get the PTS for a given index or an index slice. If None, will return all pts.
    /// If some index values are out of bounds, pts will be set to -1.
    fn get_pts(&self, index: Option<IntOrSlice>) -> PyResult<Vec<f64>> {
        match self.inner.lock() {
            Ok(vr) => {
                let time_base = vr.decoder().video_info().get("time_base").unwrap();
                let time_base = f64::from_str(time_base).unwrap();
                match index {
                    None => Ok(vr.stream_info().get_all_pts(time_base)),
                    Some(int_or_slice) => {
                        let frame_count = vr.stream_info().frame_count();
                        let index = int_or_slice.to_indices(*frame_count)?;
                        Ok(vr.stream_info().get_pts(&index, time_base))
                    }
                }
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {e}"))),
        }
    }

    /// Returns the dict with metadata information of the video. All values in the dict
    /// are strings.
    fn get_info<'a>(&'a self, py: Python<'a>) -> PyResult<Bound<'a, PyDict>> {
        match self.inner.lock() {
            Ok(vr) => {
                let mut info_dict = vr.decoder().video_info().clone();
                info_dict.insert("frame_count", vr.stream_info().frame_count().to_string());
                Ok(info_dict.into_py_dict(py)?)
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {e}"))),
        }
    }

    /// Returns the average fps of the video as float.
    fn get_fps<'a>(&'a self, py: Python<'a>) -> PyResult<Bound<'a, PyFloat>> {
        match self.inner.lock() {
            Ok(vr) => {
                let fps = vr.decoder().fps();
                Ok(PyFloat::new(py, fps))
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {e}"))),
        }
    }

    /// Get shape of the video: [number of frames, height and width]
    fn get_shape<'a>(&'a self, py: Python<'a>) -> PyResult<Bound<'a, PyList>> {
        match self.inner.lock() {
            Ok(vr) => {
                let width = vr.decoder().video().width() as usize;
                let height = vr.decoder().video().height() as usize;
                let num_frames = vr.stream_info().frame_count();
                let list = PyList::new(py, [*num_frames, height, width]);
                Ok(list?)
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {e}"))),
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
                Ok(video) => Ok(video.into_pyarray(py)),
                Err(e) => Err(PyRuntimeError::new_err(format!("Error: {e}"))),
            },
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {e}"))),
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
                    .map(|x| x.into_pyarray(py))
                    .collect::<Vec<_>>())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {e}"))),
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
                    Ok(gray_video.into_pyarray(py))
                }
                Err(e) => Err(PyRuntimeError::new_err(format!("Error: {e}"))),
            },
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {e}"))),
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
                        Ok(batch) => Ok(batch.into_pyarray(py)),
                        Err(e) => Err(PyRuntimeError::new_err(format!("Error: {e}"))),
                    }
                } else {
                    match vr.get_batch(indices) {
                        Ok(batch) => Ok(batch.into_pyarray(py)),
                        Err(e) => Err(PyRuntimeError::new_err(format!("Error: {e}"))),
                    }
                }
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {e}"))),
        }
    }
}

#[pymodule]
fn video_reader<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    env_logger::init();
    // Add the VideoReader class to the module
    m.add_class::<PyVideoReader>()?;
    Ok(())
}
