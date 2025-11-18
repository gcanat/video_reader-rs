<h1 align="center">
  <code>video_reader-rs</code>
</h1>
<p align="center">A python module to decode videos based on rust ffmpeg-next, with a focus on ML use cases.</p>

## 💡 Why yet another library based on ffmpeg ?

When training ML models on videos, it is usefull to load small sub-clips of videos. So decoding the
entire video is not necessary.

The great [decord](https://github.com/dmlc/decord) library seems to be unmaintained, while having
a few issues. The main one (for us) is bad memory management, which makes it crash on large videos.
Indeed it allocates memory for the whole video when instantiating a VideoReader object. While in fact
you might want to only get a few frames from this video.

So we took great inspiration from this library to rewrite the `get_batch` function using ffmpeg-next
rust bindings. We also added the `decode` function which is usefull for decoding the entire video or
for temporally reducing it using a `compression_factor`. Option to resize the video while decoding is also
added.

NOTE: other functionalities of `decord` are not implemented (yet?).

Benchmark indicates that `video_reader-rs` is performing equally or better than `decord`, while using less memory.
At least on the intended ML uses cases where video resolution remains reasonable, eg not 4K videos.

## 🛠️ Installation
### Install via pip
```bash
pip install video-reader-rs
```
Should work with python >= 3.8 on recent linux x86_64 and macos.

### Manual installation
You need to have ffmpeg installed on your system.
Install maturin:
```bash
pip install maturin
```

Activate a virtual-env where you want to use the video_reader library and build the library as follows:
```bash
maturin develop --release
```
`maturin develop` builds the crate and installs it as a python module directly in the current virtualenv.
the `--release` flag ensures the Rust part of the code is compiled in release mode, which enables compiler optimizations.

⚠️ If you are using a version of **ffmpeg <= 5** you need to enable the `ffmpeg_5` feature:
```bash
maturin develop --release --features ffmpeg_5
```

## 💻 Usage
Decoding a video is as simple as:
```python
from video_reader import PyVideoReader

vr = PyVideoReader(filename)
# or if you want to resize and use a specific number of threads
vr = PyVideoReader(filename, threads=8, resize_shorter_side=480)
# similar but by resizing longer side
vr = PyVideoReader(filename, threads=8, resize_longer_side=640)
# or to use GPU decoding:
vr = PyVideoReader(filename, device='cuda')

# decode all frames from the video
frames = vr.decode()
# or decode a subset of frames
frames = vr.decode(start_frame=100, end_frame=300, compression_factor=0.5)
# alternatively one can iterate over frames
for frame in vr:
    # do something with a single frame
    print("top left red pixel value:", frame[0, 0, 0])
```
* **filename**: path to the video file to decode
* **resize**: optional resizing for the video.
* **compression_factor**: temporal sampling, eg if 0.25, take 25% of the frames, evenly spaced.
* **threads**: number of CPU cores to use for ffmpeg decoding, 0 means auto (let ffmpeg pick the optimal number).
* **start_frame** - Start decoding from this frame index
* **end_frame** - Stop decoding at this frame index

Returns a numpy array of shape (N, H, W, C).

We can do the same thing if we want grayscale frames, and it will retun an array of shape (N, H, W).
```python
# this method has the same arguments as decode()
frames = vr.decode_gray()
```

If we only need a sub-clip of the video we can use the `get_batch` function:
```python
frames = vr.get_batch(indices)
```
* **indices**: list of indices of the frames to get
* **with_fallback**: False by default, if True will fallback to iterating over all packets of the video and only decoding the frames that match in `indices`. It is safer to use when the video contains B-frames and you really need to get the frames exactly corresponding to the given indices. It can also be faster in some use cases if you have many cpu cores available.

It is also possible to directly use slicing or indexing:
```python
last_frame = vr[-1]
odd_frames = vr[1::2]
sub_clip = vr[128:337]
```

We can also get the shape of the raw video
```python
# (number of frames, height, width)
(n, h, w) = vr.get_shape()
# if we only want the number of frames
n = len(vr)
```

Or get a dict with information about the video, returned as Dict[str, str]
```python
info_dict = vr.get_info()
print(info_dict)
# example output:
# {'color_space': 'BT709', 'aspect_ratio': 'Rational(1/1)', 'color_xfer_charac': 'BT709', 'codec_id': 'H264', 'fps_rational': '0/1', 'width': '1280', 'vid_ref': '1', 'duration': '148.28736979166666', 'height': '720', 'has_b_frames': 'true', 'color_primaries': 'BT709', 'chroma_location': 'Left', 'time_base': '0.00006510416666666667', 'vid_format': 'YUV420P', 'bit_rate': '900436', 'fps': '33.57669643068823', 'start_time': '0', 'color_range': 'MPEG', 'intra_dc_precision': '0', 'frame_count': '4979'}
```


### ⚠️  Dealing with High Res videos
 If you are dealing with High Resolution videos such as HD, UHD etc. We recommend using `vr.decode_fast()` which has the same arguments as `vr.decode()` but will return a list of frames. It uses async conversion from yuv420p to RGB to speed things up.

If you have some memory limitations that wont let you decode the entire video at once, you can decode by chunk like so:
```python
from video_reader import PyVideoReader

videoname = "/path/to/your/video.mp4"
vr = PyVideoReader(videoname)

chunk_size = 800 # adjust to fit within your memory limit
video_length = vr.get_shape()[0]

for i in range(0, video_length, chunk_size):
    end = min(i + chunk_size, video_length)
    frames = vr.decode_fast(
        start_frame=i,
        end_frame=end,
    )
    # do something with this chunk of 800 `frames`
```

## 🧪 Experimental support for Hardware Acceleration
You need to install `video-reader-rs` from source by cloning this repo and running `maturin develop -r`. Your ffmpeg installation should have support for cuda. Check with `ffmpeg -version | grep cuda` for example.

```python
from video_reader import PyVideoReader

videoname = "/path/to/your/video.mp4"
vr = PyVideoReader(videoname, device='cuda')
```

You can also pass your own ffmpeg [filter](https://ffmpeg.org/ffmpeg-filters.html#Video-Filters) if you feel adventurous enough. For example, this would be the default filter used when specifying `devide='cuda'` and `resize_shorter_side=512`.
```python
vr = PyVideoReader(videoname, device='cuda', filter='scale_cuda:h=512:w=-1:passthrough=0,hwdownload,format=nv12', resize_shorter_side=512)
```
In theory any hwaccel should work if you provide the correct filters, ie qsv, vaapi, vdpau, etc. It has not been tested though. Feel free to report.

Another example with VAAPI hardware acceleration:
```python
vr = PyVideoReader(videoname, device='vaapi', filter='hwmap,format=nv12')
```

## 🚀 Performance comparison
Decoding a video with shape (2004, 1472, 1472, 3). Tested on a laptop (12 cores Intel i7-9750H CPU @ 2.60GHz), 15Gb of RAM with Ubuntu 22.04.

Options: 
- f: compression factor
- r: resize shorter side
- g: grayscale

| Options | OpenCV | decord* | vr.decode | vr.decode_fast |
|:---:|:---:|:---:|:---:|:---:|
| f 1.0 | 65s | 18s | 9.3s | **6.2s** | 
| f 0.5 | 33.96s | 14.6s | 5.5s | **4.2s** | 
|f 0.25 | 7.16s | 14.03s | 4.2s | **3.8s** |
|f 0.25, r 512| 6.5s | 13.3s | 3.92s | **3.5s** |
| f 0.25, g | 20.2s | 25.7s | **6.6s** | N/A |

\* decord was tested on a machine with more RAM and CPU cores because it was crashing on the laptop with only 15Gb. See below.

## 💥 Crash test
Tested on a laptop with 15Gb of RAM, with ubuntu 22.04 and python 3.10.
Run this script:
```python
from video_reader import PyVideoReader
from time import time

def bench_video_decode(filename, compress_factor, resize):
    start =  time()
    vr = PyVideoReader(filename, resize_shorter_side=resize, threads=0)
    vid = vr.decode(compression_factor=compress_factor)
    duration = time() - start
    print(f"Duration {duration:.2f}sec")
    return vid

vid = bench_video_decode("sample.mp4", 0.25)
print("video shape:", vid.shape)

# Terminal output:
# Duration 4.81sec
# video shape: (501, 1472, 1472, 3)
```

And then run this script:
```python
from decord import VideoReader

vr = VideoReader("sample.mp4")

# Terminal output:
# terminate called after throwing an instance of 'std::bad_alloc'
#  what():  std::bad_alloc
# [1]    9636 IOT instruction (core dumped)
```

## 🌠 Credits
- [decord](https://github.com/dmlc/decord) for showing how to `get_batch` efficiently.
- [ffmpeg-next](https://github.com/zmwangx/rust-ffmpeg) for the Rust bindings to ffmpeg.
- [video-rs](https://github.com/oddity-ai/video-rs) for the nice high level api which makes it easy to encode videos and for the code snippet to convert ffmpeg frames to ndarray ;-)
