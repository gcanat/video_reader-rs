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


def packet_count(path):
    """Video packets FFmpeg demuxes, one framecrc line each."""
    output = ffmpeg("-i", path, "-map", "0:v:0", "-c", "copy", "-f", "framecrc", "-").stdout.decode()
    return sum(not line.startswith("#") for line in output.splitlines())


def corrupt_packets(path, indices=None):
    """Overwrite the leading NAL length of the given video packets, or of all of them.

    Returns the presentation indices of the corrupted packets.
    """
    if not shutil.which("ffprobe"):
        pytest.skip("requires the FFprobe executable")
    packets = [line.split(",") for line in subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts,pos", "-of", "csv=p=0", path],
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout.split()]
    order = sorted(int(pts) for pts, _ in packets)
    indices = range(len(packets)) if indices is None else indices
    data = bytearray(path.read_bytes())
    for index in indices:
        position = int(packets[index][1])
        data[position:position + 4] = b"\xff" * 4
    path.write_bytes(data)
    return [order.index(int(packets[index][0])) for index in indices]


@pytest.mark.parametrize("in_memory", [False, True], ids=["path", "bytes"])
def test_frame_dropping_filters_drain_decoder_output(tmp_path, in_memory):
    if b"libvpx-vp9" not in ffmpeg("-encoders").stdout:
        pytest.skip("requires the libvpx-vp9 encoder")
    path = tmp_path / "vp9.webm"
    ffmpeg("-i", ASSET, "-c:v", "libvpx-vp9", "-b:v", "200k", "-deadline", "realtime", "-cpu-used", "8", path)
    # Frame threads are what queue extra output, so don't leave the count to the host.
    frames = np.asarray(list(PyVideoReader(str(path), threads=4)))
    source = path.read_bytes() if in_memory else str(path)
    selected = list(PyVideoReader(source, threads=4, filter="select='not(mod(n\\,3))'"))
    np.testing.assert_array_equal(selected, frames[::3], strict=True)
    assert sum(1 for _ in PyVideoReader(source, threads=4, filter="fps=1")) == 10
    np.testing.assert_array_equal(PyVideoReader(source, threads=4).decode(), frames, strict=True)


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("in_memory", [False, True], ids=["path", "bytes"])
def test_truncated_mp4_iteration_keeps_decodable_frames(tmp_path, in_memory, threads):
    path = tmp_path / "truncated.mp4"
    ffmpeg("-stream_loop", "9", "-i", ASSET, "-c", "copy", "-movflags", "+faststart", path)
    data = path.read_bytes()
    path.write_bytes(data[:len(data) * 85 // 100])
    reference = ffmpeg("-i", path, "-f", "null", "-", "-progress", "pipe:1", "-nostats")
    expected = int([line.split(b"=")[1] for line in reference.stdout.splitlines() if line.startswith(b"frame=")][-1])
    assert expected > 700
    source = path.read_bytes() if in_memory else str(path)
    # With frame threads, the partial last packet is reported late.
    reader = PyVideoReader(source, threads=threads)
    assert sum(1 for _ in reader) == expected
    assert reader.count_actual_frames() == expected


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


def test_apng_with_invalid_chunk_keeps_readable_frames(tmp_path):
    path = tmp_path / "animation.apng"
    ffmpeg("-i", ASSET, "-vf", "scale=64:48", "-plays", "1", path)
    data = bytearray(path.read_bytes())
    chunks = [index for index in range(len(data)) if data[index:index + 4] == b"fcTL"]
    length = chunks[len(chunks) // 2] - 4
    assert data[length:length + 4] == (26).to_bytes(4, "big")
    data[length:length + 4] = (27).to_bytes(4, "big")
    path.write_bytes(data)
    expected = packet_count(path)
    assert expected >= len(chunks) // 2
    reader = PyVideoReader(str(path), threads=1)
    assert len(reader) == expected
    assert sum(1 for _ in reader) == expected


@pytest.mark.parametrize("mode", ["black", "skip"])
def test_batches_skip_only_the_corrupt_packet(tmp_path, mode):
    path = tmp_path / "corrupt.mp4"
    path.write_bytes(ASSET.read_bytes())
    corrupted = corrupt_packets(path, [50])
    decodable = np.asarray(list(PyVideoReader(str(path), threads=1)))
    assert len(decodable) == 99
    reader = PyVideoReader(str(path), threads=1, oob_mode=mode)
    batch = reader.get_batch(list(range(100)), with_fallback=True)
    lost = ~batch.any(axis=(1, 2, 3))
    # B-frames reorder output, so only PTS puts the gap where the corrupt packet was.
    assert np.flatnonzero(lost).tolist() == (corrupted if mode == "black" else [])
    np.testing.assert_array_equal(batch[~lost], decodable, strict=True)
    # Chunks resume with frames queued past the corrupt packet.
    reader = PyVideoReader(str(path), threads=1, oob_mode=mode)
    chunks = [reader.get_batch(list(range(start, start + 10)), with_fallback=True) for start in range(0, 100, 10)]
    np.testing.assert_array_equal(np.concatenate(chunks), batch, strict=True)


def test_iteration_raises_when_no_packet_decodes(tmp_path):
    path = tmp_path / "corrupt.mp4"
    path.write_bytes(ASSET.read_bytes())
    corrupt_packets(path)
    reader = PyVideoReader(str(path), threads=1)
    assert len(reader) == 100
    with pytest.raises(RuntimeError, match="Invalid data"):
        list(reader)


def test_playlist_recovers_from_damaged_segment(tmp_path):
    segments = [tmp_path / f"segment{index}.ts" for index in range(3)]
    for segment in segments:
        ffmpeg("-stream_loop", "9", "-i", ASSET, "-c", "copy", "-f", "mpegts", segment)
    # The playlist's own position stays put while the nested demuxer resyncs.
    data = bytearray(segments[1].read_bytes())
    middle = len(data) // 2 // 188 * 188
    data[middle:middle + 262144] = bytes(262144)
    segments[1].write_bytes(data)
    path = tmp_path / "playlist.ffconcat"
    path.write_text("ffconcat version 1.0\n" + "".join(f"file {segment.name}\n" for segment in segments))
    expected = packet_count(path)
    assert expected > 2500
    assert len(PyVideoReader(str(path), threads=1)) == expected


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
