use ffmpeg_next::format::Pixel as AvPixel;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;

/// A single RGB frame with flat row-major layout, shape (H, W, 3).
#[derive(Clone)]
pub struct FrameTensor {
    pub data: Vec<u8>,
    pub height: usize,
    pub width: usize,
}

impl FrameTensor {
    pub fn new(data: Vec<u8>, height: usize, width: usize) -> Self {
        Self {
            data,
            height,
            width,
        }
    }
}

/// A batch of RGB frames with flat row-major layout, shape (N, H, W, 3).
#[derive(Clone)]
pub struct VideoArray {
    pub data: Vec<u8>,
    pub n: usize,
    pub height: usize,
    pub width: usize,
}

impl VideoArray {
    pub fn zeros(n: usize, height: usize, width: usize) -> Self {
        Self {
            data: vec![0u8; n * height * width * 3],
            n,
            height,
            width,
        }
    }

    /// Assemble a VideoArray from a sequence of individual frames.
    pub fn from_frames(frames: Vec<FrameTensor>) -> Self {
        if frames.is_empty() {
            return Self::zeros(0, 0, 0);
        }
        let n = frames.len();
        let height = frames[0].height;
        let width = frames[0].width;
        let mut data = vec![0u8; n * height * width * 3];
        data.par_chunks_mut(height * width * 3)
            .zip(frames)
            .for_each(|(slice, frame)| slice.copy_from_slice(&frame.data));
        Self {
            data,
            n,
            height,
            width,
        }
    }

    pub fn frame_size(&self) -> usize {
        self.height * self.width * 3
    }

    pub fn get_frame_mut(&mut self, idx: usize) -> &mut [u8] {
        let fs = self.frame_size();
        &mut self.data[idx * fs..(idx + 1) * fs]
    }

    /// Consume self and return the first frame as a FrameTensor (shape [H, W, 3]).
    pub fn first_frame(mut self) -> FrameTensor {
        let h = self.height;
        let w = self.width;
        self.data.truncate(h * w * 3);
        FrameTensor {
            data: self.data,
            height: h,
            width: w,
        }
    }

    /// Convert to ndarray Array4 for internal processing (e.g. grayscale conversion).
    pub fn into_ndarray4(self) -> ndarray::Array4<u8> {
        ndarray::Array4::from_shape_vec((self.n, self.height, self.width, 3), self.data)
            .expect("VideoArray shape mismatch")
    }
}

/// Generic DLPack tensor with arbitrary shape (e.g. grayscale [N, H, W]).
pub struct DlPackTensor {
    pub data: Vec<u8>,
    pub shape: Vec<i64>,
}

impl DlPackTensor {
    pub fn new(data: Vec<u8>, shape: Vec<i64>) -> Self {
        Self { data, shape }
    }
}

/// Types that can be turned into a flat u8 buffer plus its row-major shape,
/// ready to be exported as a DLPack tensor.
pub trait IntoTensorParts {
    fn into_parts(self) -> (Vec<u8>, Vec<i64>);
}

impl IntoTensorParts for FrameTensor {
    fn into_parts(self) -> (Vec<u8>, Vec<i64>) {
        (self.data, vec![self.height as i64, self.width as i64, 3])
    }
}

impl IntoTensorParts for VideoArray {
    fn into_parts(self) -> (Vec<u8>, Vec<i64>) {
        (
            self.data,
            vec![self.n as i64, self.height as i64, self.width as i64, 3],
        )
    }
}

impl IntoTensorParts for DlPackTensor {
    fn into_parts(self) -> (Vec<u8>, Vec<i64>) {
        (self.data, self.shape)
    }
}

pub type FrameArray = FrameTensor;

/// Always use NV12 pixel format with hardware acceleration, then rescale later.
pub(crate) static HWACCEL_PIXEL_FORMAT: AvPixel = AvPixel::NV12;
