"""Regression checks for packet recovery and Python error/thread behaviour."""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from video_reader import PyVideoReader

ASSET = Path(__file__).resolve().parents[1] / "assets" / "input.mp4"
pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires the FFmpeg executable")


def ffmpeg(*args):
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *map(str, args)],
        check=True, capture_output=True, timeout=30,
    )


@pytest.mark.parametrize("in_memory", [False, True], ids=["path", "bytes"])
def test_frame_dropping_filters_drain_decoder_output(tmp_path, in_memory):
    if b"libvpx-vp9" not in ffmpeg("-encoders").stdout:
        pytest.skip("requires the libvpx-vp9 encoder")
    path = tmp_path / "vp9.webm"
    ffmpeg("-i", ASSET, "-c:v", "libvpx-vp9", "-b:v", "200k", "-deadline", "realtime", "-cpu-used", "8", path)
    frames = np.asarray(list(PyVideoReader(str(path))))
    source = path.read_bytes() if in_memory else str(path)
    selected = list(PyVideoReader(source, filter="select='not(mod(n\\,3))'"))
    np.testing.assert_array_equal(selected, frames[::3], strict=True)
    assert sum(1 for _ in PyVideoReader(source, filter="fps=1")) == 10


@pytest.mark.parametrize("in_memory", [False, True], ids=["path", "bytes"])
def test_truncated_mp4_iteration_keeps_decodable_frames(tmp_path, in_memory):
    path = tmp_path / "truncated.mp4"
    ffmpeg("-stream_loop", "9", "-i", ASSET, "-c", "copy", "-movflags", "+faststart", path)
    data = path.read_bytes()
    path.write_bytes(data[:len(data) * 85 // 100])
    reference = ffmpeg("-i", path, "-f", "null", "-", "-progress", "pipe:1", "-nostats")
    expected = int([line.split(b"=")[1] for line in reference.stdout.splitlines() if line.startswith(b"frame=")][-1])
    assert expected > 700
    source = path.read_bytes() if in_memory else str(path)
    assert sum(1 for _ in PyVideoReader(source, threads=1)) == expected


@pytest.mark.parametrize("in_memory", [False, True], ids=["path", "bytes"])
def test_truncated_apng_recovers_after_unknown_chunk(tmp_path, in_memory):
    path = tmp_path / "animation.apng"
    ffmpeg("-i", ASSET, "-vf", "scale=64:48", "-plays", "1", path)
    frames = np.asarray(list(PyVideoReader(str(path), threads=1)))
    data = path.read_bytes()
    assert data[-8:-4] == b"IEND"
    path.write_bytes(data[:-12])
    source = path.read_bytes() if in_memory else str(path)
    reader = PyVideoReader(source, threads=1)
    assert len(reader) == len(frames)
    np.testing.assert_array_equal(list(reader), frames, strict=True)


@pytest.mark.parametrize("source", ["str(path)", "path.read_bytes()"], ids=["path", "bytes"])
def test_fragmented_mp4_without_read_progress_does_not_hang(tmp_path, source):
    path = tmp_path / "fragmented.mp4"
    ffmpeg("-i", ASSET, "-c", "copy", "-movflags", "frag_keyframe+empty_moov", path)
    data = bytearray(path.read_bytes())
    offset = data.index(b"trun")
    assert int.from_bytes(data[offset + 4:offset + 8], "big") & 1
    data[offset + 12:offset + 16] = bytes.fromhex("80000000")
    path.write_bytes(data)
    script = f"""
from pathlib import Path
import pytest
from video_reader import PyVideoReader
path = Path({str(path)!r})
with pytest.raises(RuntimeError):
    PyVideoReader({source})
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="uses /proc to inject a descriptor failure")
@pytest.mark.parametrize("operation", ["iterate", "count"])
def test_iteration_and_count_surface_io_errors(tmp_path, operation):
    path = tmp_path / "input.mp4"
    ffmpeg("-stream_loop", "9", "-i", ASSET, "-c", "copy", "-movflags", "+faststart", path)
    # Isolate closing FFmpeg's descriptor from the test runner and its other readers.
    script = f"""
import os
from pathlib import Path
import pytest
from video_reader import PyVideoReader
reader = PyVideoReader({str(path)!r}, threads=1)
descriptors = []
for fd in Path('/proc/self/fd').iterdir():
    try:
        if os.readlink(fd) == {str(path)!r}:
            descriptors.append(int(fd.name))
    except FileNotFoundError:
        pass
assert len(descriptors) == 1, descriptors
os.close(descriptors[0])
# Some demuxers reject a partial packet before surfacing the I/O errno.
with pytest.raises(RuntimeError):
    if {operation!r} == 'iterate':
        for frame in reader:
            pass
    else:
        reader.count_actual_frames()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("operation", ["constructor", "count"])
def test_constructor_and_count_allow_python_thread_progress(tmp_path, operation):
    path = tmp_path / "hour.mp4"
    ffmpeg("-stream_loop", "359", "-i", ASSET, "-c", "copy", "-movflags", "+faststart", path)
    script = f"""
import sys
import threading
from pathlib import Path
from video_reader import PyVideoReader
data = Path({str(path)!r}).read_bytes()
reader = PyVideoReader({str(ASSET)!r}, threads=1)
operation = (lambda: PyVideoReader(data, threads=1)) if {operation!r} == 'constructor' else reader.count_actual_frames
ready, stop = threading.Event(), threading.Event()
ticks = [0]
def observe():
    ready.set()
    while not stop.wait(.001):
        ticks[0] += 1
thread = threading.Thread(target=observe)
thread.start()
ready.wait()
# Prevent post-call Python scheduling from masquerading as progress during native work.
sys.setswitchinterval(10)
try:
    progress = []
    for _ in range(5):
        before = ticks[0]
        operation()
        progress.append(ticks[0] - before)
    assert any(progress), progress
finally:
    stop.set()
    thread.join()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
