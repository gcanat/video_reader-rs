use ffmpeg::log as ffmpeg_log;
use ffmpeg_next as ffmpeg;
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
use decoder::{DecoderConfig, OutOfBoundsMode, ResizeAlgo};
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
        .unwrap_or_else(|e| panic!("Failed to build tokio runtime: {e}"))
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
    fn __next__<'a>(
        slf: PyRefMut<'_, Self>,
        py: Python<'a>,
    ) -> Option<Bound<'a, PyArray<u8, Dim<[usize; 3]>>>> {
        match slf.inner.lock() {
            Ok(mut vr) => vr.next().map(|rgb_frame| rgb_frame.into_pyarray(py)),
            Err(e) => {
                debug!("Lock error in __next__: {e}");
                None
            }
        }
    }

    fn __getitem__<'a>(&self, py: Python<'a>, key: IntOrSlice) -> PyResult<Bound<'a, FrameOrVid>> {
        match self.inner.lock() {
            Ok(mut vr) => {
                let frame_count = *vr.stream_info().frame_count();
                let index = key.to_indices(frame_count)?;
                let index_clone = index.clone();

                // For single frame access (reader[i]), always use seek-based method
                // This enables skip-forward optimization for sequential access patterns
                // like: for i in range(n): reader[i]
                let is_single_frame = matches!(key, IntOrSlice::Int { .. });

                // For slices/lists, use the cost estimation logic
                let force_sequential = vr.needs_sequential_mode();
                let use_sequential = if is_single_frame {
                    // Single frame: use seek-based unless seek is completely broken
                    force_sequential
                } else if force_sequential {
                    true
                } else {
                    vr.should_use_sequential(&index)
                };

                // Try the selected method, with automatic fallback for seek-based -> sequential
                let res_array = if use_sequential {
                    vr.get_batch_safe(index.clone())
                } else {
                    // Try seek-based first
                    match vr.get_batch(index.clone()) {
                        Ok(arr) => Ok(arr),
                        Err(_) => {
                            // Fallback to sequential mode if seek-based fails
                            debug!("__getitem__: get_batch failed, falling back to get_batch_safe");
                            vr.get_batch_safe(index.clone())
                        }
                    }
                }.map_err(|e| {
                    // Convert Bug error to a more meaningful message
                    let failed = vr.failed_indices();
                    let msg = match e {
                        ffmpeg::Error::Bug => {
                            if !failed.is_empty() {
                                format!(
                                    "Failed to decode frame(s) at index {:?} (requested {:?}, frame_count={})",
                                    failed, index_clone, frame_count
                                )
                            } else {
                                format!(
                                    "Failed to decode frame(s) at index {:?} (frame_count={})",
                                    index_clone, frame_count
                                )
                            }
                        }
                        _ => format!("{e}"),
                    };
                    PyRuntimeError::new_err(format!("Error: {msg}"))
                })?;

                // remove first dim if key was a single int
                if matches!(key, IntOrSlice::Int { .. }) {
                    // Extract the first frame and convert to owned array
                    use ndarray::Axis;
                    let single_frame = res_array.index_axis(Axis(0), 0).to_owned();
                    Ok(single_frame.into_dyn().into_pyarray(py))
                } else {
                    Ok(res_array.into_dyn().into_pyarray(py))
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
                let res_decode = RUNTIME
                    .block_on(async {
                        reader
                            .decode_video_fast(start_frame, end_frame, compression_factor)
                            .await
                    })
                    .map_err(|e| PyRuntimeError::new_err(format!("Error: {e}")))?;
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
                    let gray_video = rgb2gray(video)
                        .map_err(|e| PyRuntimeError::new_err(format!("Error: {e}")))?;
                    Ok(gray_video.into_pyarray(py))
                }
                Err(e) => Err(PyRuntimeError::new_err(format!("Error: {e}"))),
            },
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {e}"))),
        }
    }

    #[pyo3(signature = (indices, with_fallback=None))]
    /// Decodes the frames in the video corresponding to the indices in `indices`.
    /// * `indices` - list of frame index to decode.
    /// * `with_fallback` - None (auto), True (sequential), or False (seek-based).
    ///   - None: automatically choose the faster method based on cost estimation
    ///   - True: use sequential decoding (iterate through all frames)
    ///   - False: use seek-based decoding (seek to keyframes)
    fn get_batch<'a>(
        &'a self,
        py: Python<'a>,
        indices: Vec<usize>,
        with_fallback: Option<bool>,
    ) -> PyResult<Bound<'a, PyArray<u8, Dim<[usize; 4]>>>> {
        match self.inner.lock() {
            Ok(mut vr) => {
                // For videos with negative PTS/DTS, verify if seek actually works
                // Some negative DTS videos work fine (e.g., time_base 1/15360)
                // while others fail (e.g., time_base 1/1000000 or negative PTS)
                let force_sequential = vr.needs_sequential_mode();

                // Determine which method to use
                let use_sequential = match with_fallback {
                    Some(true) => true, // Explicitly use sequential
                    Some(false) => {
                        // User requested seek-based, but force sequential if seek is broken
                        if force_sequential {
                            debug!("Seek verification failed - forcing sequential mode");
                            true
                        } else {
                            false
                        }
                    }
                    None => {
                        // Auto mode: if seek is broken, use sequential
                        if force_sequential {
                            debug!("Seek verification failed - using sequential mode");
                            true
                        } else {
                            // estimate which is faster
                            vr.should_use_sequential(&indices)
                        }
                    }
                };

                if use_sequential {
                    debug!("Using sequential method (get_batch_safe)");
                    match vr.get_batch_safe(indices.clone()) {
                        Ok(batch) => Ok(batch.into_pyarray(py)),
                        Err(e) => {
                            // Convert Bug error to a more meaningful message
                            let failed = vr.failed_indices();
                            let msg = match e {
                                ffmpeg::Error::Bug => {
                                    if !failed.is_empty() {
                                        format!("Out of bounds: frame indices {:?} exceed video length or could not be decoded", failed)
                                    } else {
                                        "Out of bounds: frame index exceeds video length or could not be decoded".to_string()
                                    }
                                }
                                _ => format!("{e}"),
                            };
                            Err(PyRuntimeError::new_err(format!("Error: {msg}")))
                        }
                    }
                } else {
                    debug!("Using seek-based method (get_batch)");
                    match vr.get_batch(indices.clone()) {
                        Ok(batch) => Ok(batch.into_pyarray(py)),
                        Err(e) => {
                            // Convert Bug error to a more meaningful message
                            let failed = vr.failed_indices();
                            let msg = match e {
                                ffmpeg::Error::Bug => {
                                    if !failed.is_empty() {
                                        format!("Out of bounds: frame indices {:?} exceed video length or could not be decoded", failed)
                                    } else {
                                        "Out of bounds: frame index exceeds video length or could not be decoded".to_string()
                                    }
                                }
                                _ => format!("{e}"),
                            };
                            Err(PyRuntimeError::new_err(format!("Error: {msg}")))
                        }
                    }
                }
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Lock error: {e}"))),
        }
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
    // Add the VideoReader class to the module
    m.add_class::<PyVideoReader>()?;
    Ok(())
}
