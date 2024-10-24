<h1 align="center">
  <code>video_reader-rs</code>
</h1>
<p align="center">A python module to decode videos based on rust ffmpeg-next, with a focus on ML use cases.</p>

## :bulb: Why yet another library based on ffmpeg ?

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

## :hammer_and_wrench: Installation
### Install via pip
```bash
pip install video-reader-rs
```
Should work with python >= 3.8 on recent linux x86_64, macos and windows.

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

:warning: If you are using a version of **ffmpeg >= 6.0** you need to enable the `ffmpeg_6_0` feature:
```bash
maturin develop --release --features ffmpeg_6_0
```

## :computer: Usage
Decoding a video is as simple as:
```python
import video_reader as vr
frames = vr.decode(filename, resize, compression_factor, threads, start_frame, end_frame)
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
frames = vr.decode_gray(filename, resize, compression_factor, threads, start_frame, end_frame)
```

If we only need a sub-clip of the video we can use the `get_batch` function:
```python
frames = vr.get_batch(filename, indices, threads=0, resize_shorter_side=None, with_fallback=False)
```
* **filename**: path to the video file to decode
* **indices**: list of indices of the frames to get
* **threads**: number of CPU cores to use for ffmpeg decoding, currently has no effect as `get_batch` does not support multithreading. (NOTE: it is still as fast as decord from our benchmarking)
* **resize_shorter_side**: optional resizing for the video.
* **with_fallback**: False by default, if True will fallback to iterating over all packets of the video and only decoding the frames that match in `indices`. It is safer to use when the video contains B-frames and you really need to get the frames exactly corresponding to the given indices. It can also be faster in some use cases if you have many cpu cores available.

We can also get the shape of the raw video
```python
(n, h, w) = vr.get_shape(filename)
```

Or get a dict with information about the video, returned as Dict[str, str]
```python
info_dict = vr.get_info(filename)
print(info_dict["fps"])
```

We can encode the video with h264 codec
```python
vr.save_video(frames, "video.mp4", fps=15, codec="h264")
```
NOTE: currently only work if the frames shape is a multiple of 32.

## :rocket: Performance comparison
Decoding a video with shape (2004, 1472, 1472, 3). Tested on a laptop (12 cores Intel i7-9750H CPU @ 2.60GHz), 15Gb of RAM with Ubuntu 22.04.

Options: 
- f: compression factor
- r: resize shorter side
- g: grayscale

| Options | OpenCV | decord* | video_reader |
|:---:|:---:|:---:|:---:|
| f 0.5 | 33.96s | **14.6s** | 26.76s | 
|f 0.25 | 7.16s | 14.03s | **6.73s** |
|f 0.25, r 512| 6.49s | 13.33s | **3.92s** |
| f 0.25, g | 20.24s | 25.67s | **14.11s** |

\* decord was tested on a machine with more RAM and CPU cores because it was crashing on the laptop with only 15Gb. See below.

## :boom: Crash test
Tested on a laptop with 15Gb of RAM, with ubuntu 22.04 and python 3.10.
Run this script:
```python
import video_reader as vr
from time import time

def bench_video_decode(filename, compress_factor, resize):
    start =  time()
    vid = vr.decode(filename, resize_shorter_side=resize, compression_factor=compress_factor, threads=0)
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

## :stars: Credits
- [decord](https://github.com/dmlc/decord) for showing how to `get_batch` efficiently.
- [ffmpeg-next](https://github.com/zmwangx/rust-ffmpeg) for the Rust bindings to ffmpeg.
- [video-rs](https://github.com/oddity-ai/video-rs) for the nice high level api which makes it easy to encode videos and for the code snippet to convert ffmpeg frames to ndarray ;-)
