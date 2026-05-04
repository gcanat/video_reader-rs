mod convert;
mod decoder;
mod ffi_hwaccel;
mod filter;
mod hwaccel;
mod info;
mod reader;
mod utils;
use convert::rgb2gray;
use decoder::{DecoderConfig, OutOfBoundsMode, ResizeAlgo};
use dlpark::prelude::*;
use ffmpeg::log as ffmpeg_log;
use ffmpeg_next as ffmpeg;
use hwaccel::HardwareAccelerationDeviceType;
use log::debug;
use ndarray::Array;
use once_cell::sync::Lazy;
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods, pymodule,
    types::{
        IntoPyDict, PyAnyMethods, PyDict, PyFloat, PyList, PyModule, PyModuleMethods, PySlice,
    },
    Bound, FromPyObject, PyRef, PyRefMut, PyResult, Python,
};
use reader::VideoReader;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use tokio::runtime::{self, Runtime};
use utils::{DlPackTensor, VideoArray};

static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap_or_else(|e| panic!("Failed to build tokio runtime: {e}"))
});

/// Python-facing DLPack tensor implementing the Array API protocol.
///
/// Exposes `__dlpack__()` and `__dlpack_device__()` so frameworks can consume it via
/// `np.from_dlpack(t)`, `torch.from_dlpack(t)`, etc.
///
/// The raw pointer is stored as `usize` so the struct is `Send` (required by `#[pyclass]`).
/// Ownership is transferred on the first `__dlpack__()` call; subsequent calls raise an error.
#[pyclass(name = "DlPackTensor")]
struct PyDlPackTensor {
    ptr: AtomicUsize,
}

impl Drop for PyDlPackTensor {
    fn drop(&mut self) {
        let raw = self.ptr.swap(0, Ordering::SeqCst);
        if raw != 0 {
            unsafe { drop(SafeManagedTensorVersioned::from_raw(raw as *mut _)) };
        }
    }
}

#[pymethods]
impl PyDlPackTensor {
    #[pyo3(signature = (stream=None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<pyo3::Py<pyo3::types::PyAny>>,
    ) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
        let _ = stream;
        let raw = self.ptr.swap(0, Ordering::SeqCst);
        if raw == 0 {
            return Err(PyRuntimeError::new_err("DLPack tensor already consumed"));
        }
        let tensor = unsafe { SafeManagedTensorVersioned::from_raw(raw as *mut _) };
        to_dlpack_capsule(py, tensor)
    }

    fn __dlpack_device__(&self) -> (u32, u32) {
        (1, 0) // kDLCPU = 1, device_id = 0
    }
}

/// Build a PyCapsule named "dltensor_versioned" from an owned `SafeManagedTensorVersioned`.
/// Used internally by `PyDlPackTensor.__dlpack__`.
fn to_dlpack_capsule<'py>(
    py: Python<'py>,
    tensor: SafeManagedTensorVersioned,
) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
    unsafe extern "C" fn capsule_deleter(capsule: *mut pyo3::ffi::PyObject) {
        unsafe {
            if pyo3::ffi::PyCapsule_IsValid(capsule, c"used_dltensor_versioned".as_ptr()) == 1 {
                return;
            }
            let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, c"dltensor_versioned".as_ptr());
            if ptr.is_null() {
                pyo3::ffi::PyErr_WriteUnraisable(capsule);
                return;
            }
            drop(SafeManagedTensorVersioned::from_raw(ptr as *mut _));
        }
    }
    unsafe {
        let raw = tensor.into_raw() as *mut std::ffi::c_void;
        let capsule =
            pyo3::ffi::PyCapsule_New(raw, c"dltensor_versioned".as_ptr(), Some(capsule_deleter));
        Bound::from_owned_ptr_or_err(py, capsule)
    }
}

/// Wrap a `SafeManagedTensorVersioned` into a `PyDlPackTensor` Python object.
fn into_py_tensor<'py>(
    py: Python<'py>,
    tensor: SafeManagedTensorVersioned,
) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
    let raw = unsafe { tensor.into_raw() } as usize;
    Ok(pyo3::Py::new(
        py,
        PyDlPackTensor {
            ptr: AtomicUsize::new(raw),
        },
    )?
    .into_bound(py)
    .into_any())
}

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
    #[pyo3(signature = (filename, threads=None, resize_shorter_side=None, resize_longer_side=None, target_width=None, target_height=None, resize_algo=None, device=None, filter=None, log_level=None, oob_mode=None))]
    /// create an instance of VideoReader
    /// * `filename` - path to the video file
    /// * `threads` - number of threads to use. If None, let ffmpeg choose the optimal number.
    /// * `resize_shorter_side - Optional, resize shorted side of the video to this value. If
    /// resize_longer_side is set to None, will try to preserve original aspect ratio.
    /// * `resize_longer_side - Optional, resize longer side of the video to this value. If
    /// resize_shorter_side is set to None, will try to preserve aspect ratio.
    /// * `target_width` - Optional, resize to exact width. Must be used with target_height.
    /// * `target_height` - Optional, resize to exact height. Must be used with target_width.
    /// * `device` - type of hardware acceleration, eg: 'cuda', 'vdpau', 'drm', etc.
    /// * `filter` - custome ffmpeg filter to use, eg "format=rgb24,scale=w=256:h=256:flags=fast_bilinear"
    /// If set to None (default) or 'cpu' then cpu is used.
    /// * `oob_mode` - how to handle out-of-bounds or failed frame fetches:
    ///   - None or "error": raise an error (default, current behavior)
    ///   - "skip": skip failed frames - returned array may have fewer frames
    ///   - "black": return black (all-zero) frames for failed fetches
    /// * returns a PyVideoReader instance.
    #[allow(clippy::too_many_arguments)]
    fn new(
        filename: &str,
        threads: Option<usize>,
        resize_shorter_side: Option<f64>,
        resize_longer_side: Option<f64>,
        target_width: Option<u32>,
        target_height: Option<u32>,
        resize_algo: Option<&str>,
        device: Option<&str>,
        filter: Option<String>,
        log_level: Option<&str>,
        oob_mode: Option<&str>,
    ) -> PyResult<Self> {
        // Configure ffmpeg log level (global). Default to Error to suppress noisy warnings.
        let ffmpeg_level = match log_level {
            None => ffmpeg_log::Level::Error,
            Some(lv) => match lv.to_lowercase().as_str() {
                "quiet" => ffmpeg_log::Level::Quiet,
                "panic" => ffmpeg_log::Level::Panic,
                "fatal" => ffmpeg_log::Level::Fatal,
                "error" => ffmpeg_log::Level::Error,
                "warning" | "warn" => ffmpeg_log::Level::Warning,
                "info" => ffmpeg_log::Level::Info,
                "verbose" => ffmpeg_log::Level::Verbose,
                "debug" => ffmpeg_log::Level::Debug,
                "trace" => ffmpeg_log::Level::Trace,
                other => {
                    return Err(PyRuntimeError::new_err(format!(
                        "Invalid log_level: {other}. Use one of: quiet, panic, fatal, error, warning, info, verbose, debug, trace"
                    )))
                }
            },
        };
        ffmpeg_log::set_level(ffmpeg_level);

        // Parse oob_mode
        let out_of_bounds_mode = match oob_mode {
            None | Some("error") => OutOfBoundsMode::Error,
            Some("skip") => OutOfBoundsMode::Skip,
            Some("black") => OutOfBoundsMode::Black,
            Some(other) => {
                return Err(PyRuntimeError::new_err(format!(
                    "Invalid oob_mode: {other}. Use one of: error, skip, black"
                )))
            }
        };

        let hwaccel = match device {
            Some("cpu") | None => None,
            Some(other) => Some(
                HardwareAccelerationDeviceType::from_str(other)
                    .map_err(|_| PyRuntimeError::new_err(format!("Invalid device: {other}")))?,
            ),
        };

        // Parse resize algorithm
        let resize_algorithm = match resize_algo {
            None | Some("fast_bilinear") => ResizeAlgo::FastBilinear,
            Some("bilinear") => ResizeAlgo::Bilinear,
            Some("bicubic") => ResizeAlgo::Bicubic,
            Some("nearest") => ResizeAlgo::Nearest,
            Some("area") => ResizeAlgo::Area,
            Some("lanczos") => ResizeAlgo::Lanczos,
            Some(other) => {
                return Err(PyRuntimeError::new_err(format!(
                    "Invalid resize_algo: {other}. Use one of: fast_bilinear, bilinear, bicubic, nearest, area, lanczos"
                )))
            }
        };

        let decoder_config = DecoderConfig::new(
            threads.unwrap_or(0),
            resize_shorter_side,
            resize_longer_side,
            target_width,
            target_height,
            resize_algorithm,
            hwaccel,
            filter,
        );
        match VideoReader::new(filename.to_string(), decoder_config, out_of_bounds_mode) {
            Ok(reader) => Ok(PyVideoReader {
                inner: Mutex::new(reader),
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!("Error: {e}"))),
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__<'py>(
        slf: PyRefMut<'_, Self>,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, pyo3::types::PyAny>>> {
        match slf.inner.lock() {
            Ok(mut vr) => match vr.next() {
                Some(frame) => {
                    let t = SafeManagedTensorVersioned::new(frame).unwrap();
                    Ok(Some(into_py_tensor(py, t)?))
                }
                None => Ok(None),
            },
            Err(e) => {
                debug!("Lock error in __next__: {e}");
                Ok(None)
            }
        }
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        key: IntOrSlice,
    ) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
        let frame_count = match self.inner.lock() {
            Ok(vr) => Ok(*vr.stream_info().frame_count()),
            Err(e) => Err(e),
        };
        if let Ok(frame_cnt) = frame_count {
            let index = key.to_indices(frame_cnt)?;
            let index_clone = index.clone();
            // For single frame access (reader[i]), use seek-based method to enable
            // skip-forward optimisation for sequential access like: for i in range(n): reader[i]
            let is_single_frame = matches!(key, IntOrSlice::Int { .. });

            // Decode without GIL — returns VideoArray (Send)
            let video_array: PyResult<VideoArray> = py.detach(|| {
                match self.inner.lock() {
                    Ok(mut vr) => {
                        let force_sequential = vr.needs_sequential_mode();
                        let use_sequential = if is_single_frame {
                            force_sequential
                        } else if force_sequential {
                            true
                        } else {
                            vr.should_use_sequential(&index)
                        };

                        let res_array = if use_sequential {
                            vr.get_batch_safe(index.clone())
                        } else {
                            match vr.get_batch(index.clone()) {
                                Ok(arr) => Ok(arr),
                                Err(_) => {
                                    debug!("__getitem__: get_batch failed, falling back to get_batch_safe");
                                    vr.get_batch_safe(index.clone())
                                }
                            }
                        }
                        .map_err(|e| {
                            let failed = vr.failed_indices();
                            let msg = match e {
                                ffmpeg::Error::Bug => format!(
                                    "Failed to decode frame(s) at index {:?} (requested {:?}, frame_count={})",
                                    failed, index_clone, frame_cnt
                                ),
                                _ => format!("{e}"),
                            };
                            PyRuntimeError::new_err(format!("Error: {msg}"))
                        })?;

                        Ok(res_array)
                    }
                    Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {e}"))),
                }
            });

            // Create DLPack tensor with GIL held (SafeManagedTensorVersioned is !Send)
            let arr = video_array?;
            let tensor = if is_single_frame {
                SafeManagedTensorVersioned::new(arr.first_frame()).unwrap()
            } else {
                SafeManagedTensorVersioned::new(arr).unwrap()
            };
            into_py_tensor(py, tensor)
        } else {
            Err(PyRuntimeError::new_err(
                "Could not find frame count".to_string(),
            ))
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
                let time_base = vr
                    .decoder()
                    .video_info()
                    .get("time_base")
                    .ok_or_else(|| PyRuntimeError::new_err("time_base missing"))?;
                let time_base = f64::from_str(time_base)
                    .map_err(|e| PyRuntimeError::new_err(format!("Invalid time_base: {e}")))?;
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
                info_dict.insert("width", vr.decoder().width.to_string());
                info_dict.insert("height", vr.decoder().height.to_string());

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
                // Use decoded/output dimensions (after rotation/filters)
                let width = vr.decoder().width as usize;
                let height = vr.decoder().height as usize;
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
    /// * returns a DLPack tensor of shape (N, H, W, C)
    fn decode<'py>(
        &self,
        py: Python<'py>,
        start_frame: Option<usize>,
        end_frame: Option<usize>,
        compression_factor: Option<f64>,
    ) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
        let video: PyResult<VideoArray> = py.detach(|| match self.inner.lock() {
            Ok(mut reader) => reader
                .decode_video(start_frame, end_frame, compression_factor)
                .map_err(|e| PyRuntimeError::new_err(format!("Error: {e}"))),
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {e}"))),
        });
        let t = SafeManagedTensorVersioned::new(video?).unwrap();
        into_py_tensor(py, t)
    }

    #[pyo3(signature = (start_frame=None, end_frame=None, compression_factor=None))]
    /// Decode the video using async YUV-to-RGB conversion (faster for high-res videos).
    /// * `start_frame` - optional starting index (will start decoding from this frame)
    /// * `end_frame` - optional last frame index (will stop decoding after this frame)
    /// * `compression_factor` - optional temporal compression, eg if set to 0.25, will
    /// decode 1 frame out of 4. If None, will default to 1.0, ie decoding all frames.
    /// * returns a DLPack tensor of shape (N, H, W, C)
    fn decode_fast<'py>(
        &self,
        py: Python<'py>,
        start_frame: Option<usize>,
        end_frame: Option<usize>,
        compression_factor: Option<f64>,
    ) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
        let frames: PyResult<VideoArray> = py.detach(|| match self.inner.lock() {
            Ok(mut reader) => {
                let raw = RUNTIME
                    .block_on(async {
                        reader
                            .decode_video_fast(start_frame, end_frame, compression_factor)
                            .await
                    })
                    .map_err(|e| PyRuntimeError::new_err(format!("Error: {e}")))?;
                Ok(VideoArray::from_frames(raw))
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {e}"))),
        });
        let t = SafeManagedTensorVersioned::new(frames?).unwrap();
        into_py_tensor(py, t)
    }

    #[pyo3(signature = (start_frame=None, end_frame=None, compression_factor=None))]
    /// Decode the video, returning grayscale frames.
    /// * `start_frame` - optional starting index (will start decoding from this frame)
    /// * `end_frame` - optional last frame index (will stop decoding after this frame)
    /// * `compression_factor` - optional temporal compression, eg if set to 0.25, will
    /// decode 1 frame out of 4. If None, will default to 1.0, ie decoding all frames.
    /// * returns a DLPack tensor of shape (N, H, W)
    fn decode_gray<'py>(
        &self,
        py: Python<'py>,
        start_frame: Option<usize>,
        end_frame: Option<usize>,
        compression_factor: Option<f64>,
    ) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
        let gray: PyResult<DlPackTensor> = py.detach(|| match self.inner.lock() {
            Ok(mut reader) => {
                let video = reader
                    .decode_video(start_frame, end_frame, compression_factor)
                    .map_err(|e| PyRuntimeError::new_err(format!("Error: {e}")))?;
                let gray = rgb2gray(video.into_ndarray4())
                    .map_err(|e| PyRuntimeError::new_err(format!("Error: {e}")))?;
                let shape = gray.shape().iter().map(|&s| s as i64).collect();
                let data: Vec<u8> = gray.into_iter().collect();
                Ok(DlPackTensor::new(data, shape))
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {e}"))),
        });
        let t = SafeManagedTensorVersioned::new(gray?).unwrap();
        into_py_tensor(py, t)
    }

    #[pyo3(signature = (indices, with_fallback=None))]
    /// Decodes the frames in the video corresponding to the indices in `indices`.
    /// * `indices` - list of frame index to decode.
    /// * `with_fallback` - None (auto), True (sequential), or False (seek-based).
    ///   - None: automatically choose the faster method based on cost estimation
    ///   - True: use sequential decoding (iterate through all frames)
    ///   - False: use seek-based decoding (seek to keyframes)
    /// * returns a DLPack tensor of shape (N, H, W, C)
    fn get_batch<'py>(
        &self,
        py: Python<'py>,
        indices: Vec<usize>,
        with_fallback: Option<bool>,
    ) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
        let batch: PyResult<VideoArray> = py.detach(|| match self.inner.lock() {
            Ok(mut vr) => {
                let force_sequential = vr.needs_sequential_mode();
                let use_sequential = match with_fallback {
                    Some(true) => true,
                    Some(false) => force_sequential,
                    None => force_sequential || vr.should_use_sequential(&indices),
                };
                let res = if use_sequential {
                    vr.get_batch_safe(indices.clone())
                } else {
                    vr.get_batch(indices.clone())
                };
                res.map_err(|e| {
                    let failed = vr.failed_indices();
                    let msg = match e {
                        ffmpeg::Error::Bug => format!(
                            "Out of bounds: frame indices {:?} exceed video length or could not be decoded",
                            failed
                        ),
                        _ => format!("{e}"),
                    };
                    PyRuntimeError::new_err(format!("Error: {msg}"))
                })
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {e}"))),
        });
        let t = SafeManagedTensorVersioned::new(batch?).unwrap();
        into_py_tensor(py, t)
    }

    /// Estimate decode cost for both methods.
    /// Returns (seek_cost, sequential_cost) - the estimated number of frames to decode.
    fn estimate_decode_cost(&self, indices: Vec<usize>) -> PyResult<(usize, usize)> {
        match self.inner.lock() {
            Ok(vr) => Ok(vr.estimate_decode_cost(&indices)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {e}"))),
        }
    }

    /// Detailed decode cost estimation.
    /// Returns dict with: seek_frames, seek_count, sequential_frames, unique_count, max_index, recommendation
    fn estimate_decode_cost_detailed(
        &self,
        indices: Vec<usize>,
    ) -> PyResult<std::collections::HashMap<String, usize>> {
        match self.inner.lock() {
            Ok(vr) => {
                let (seek_frames, seek_count, sequential_frames, unique_count, max_index) =
                    vr.estimate_decode_cost_detailed(&indices);
                let use_sequential = vr.should_use_sequential(&indices);

                let mut result = std::collections::HashMap::new();
                result.insert("seek_frames".to_string(), seek_frames);
                result.insert("seek_count".to_string(), seek_count);
                result.insert("sequential_frames".to_string(), sequential_frames);
                result.insert("unique_count".to_string(), unique_count);
                result.insert("max_index".to_string(), max_index);
                result.insert(
                    "recommendation".to_string(),
                    if use_sequential { 1 } else { 0 },
                ); // 1=True, 0=False

                // Calculate total cost with overhead
                const SEEK_OVERHEAD_FRAMES: usize = 5;
                result.insert(
                    "seek_total_cost".to_string(),
                    seek_frames + seek_count * SEEK_OVERHEAD_FRAMES,
                );

                Ok(result)
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {e}"))),
        }
    }

    /// Count actual decodable frames by decoding without color conversion.
    /// This is slower than reading metadata but gives accurate results for B-frame videos.
    /// Equivalent to ffprobe's `nb_read_frames` with `-count_frames` option.
    fn count_actual_frames(&self) -> PyResult<usize> {
        match self.inner.lock() {
            Ok(mut vr) => Ok(vr.count_actual_frames()),
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {e}"))),
        }
    }
}

#[pymodule]
fn video_reader<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    env_logger::init();
    m.add_class::<PyVideoReader>()?;
    m.add_class::<PyDlPackTensor>()?;
    Ok(())
}
