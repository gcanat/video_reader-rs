import shutil
import subprocess

import numpy as np
import pytest
from video_reader import PyVideoReader


def write_video(path, color_range, matrix="bt709", y=64, u=128, v=128, height=48):
    if shutil.which("ffmpeg") is None:
        pytest.skip("requires the ffmpeg executable")
    raw = bytes([y]) * (64 * height) + bytes([u]) * (32 * height // 2) + bytes([v]) * (32 * height // 2)
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-f", "rawvideo", "-pixel_format", "yuv420p",
        "-video_size", f"64x{height}", "-framerate", "1", "-color_range", color_range,
        "-colorspace", matrix, "-i", "pipe:0", "-c:v", "ffv1",
        "-color_range", color_range, "-colorspace", matrix, str(path),
    ], input=raw, check=True, capture_output=True, timeout=10)
    return str(path)


@pytest.fixture(scope="module", params=["tv", "pc"])
def range_video(request, tmp_path_factory):
    color_range = request.param
    path = tmp_path_factory.mktemp(color_range) / "input.mkv"
    return write_video(path, color_range), color_range


@pytest.mark.parametrize("read", [
    pytest.param(lambda reader: reader.decode(), id="decode"),
    pytest.param(lambda reader: np.asarray(reader.decode_fast()), id="decode-fast"),
    pytest.param(lambda reader: reader.get_batch([0]), id="batch"),
    pytest.param(lambda reader: reader.get_batch([0], with_fallback=True), id="batch-fallback"),
    pytest.param(lambda reader: np.asarray(list(reader)), id="iteration"),
])
def test_rgb_uses_filtered_range(range_video, read):
    path, input_range = range_video
    output_range = "pc" if input_range == "tv" else "tv"
    reader = PyVideoReader(
        path, threads=1,
        filter=f"scale=in_range={input_range}:out_range={output_range},format=yuv420p",
    )
    # Neutral Y=64 becomes full-range RGB=56 from limited input, or remains 64 from full input.
    expected = np.full((1, 48, 64, 3), 56 if input_range == "tv" else 64, dtype=np.uint8)
    np.testing.assert_array_equal(read(reader), expected, strict=True)


def test_grayscale_uses_filtered_range(range_video):
    path, input_range = range_video
    output_range = "pc" if input_range == "tv" else "tv"
    reader = PyVideoReader(
        path, threads=1,
        filter=f"scale=in_range={input_range}:out_range={output_range},format=yuv420p",
    )
    expected = np.full((1, 48, 64), 56 if input_range == "tv" else 64, dtype=np.uint8)
    np.testing.assert_array_equal(reader.decode_gray(), expected, strict=True)


def test_bt2020_metadata_overrides_resolution(tmp_path):
    path = write_video(tmp_path / "bt2020.mkv", "pc", "bt2020nc", y=128, u=160, v=160)
    tall_path = write_video(tmp_path / "bt2020-tall.mkv", "pc", "bt2020nc", y=128, u=160, v=160, height=1100)
    reader = PyVideoReader(path, threads=1, filter="format=yuv420p")
    assert reader.get_info()["color_space"] == "BT2020NCL"
    # Identical coded colors must stay identical when only the resolution changes.
    reference = PyVideoReader(tall_path, threads=1, filter="format=yuv420p").decode()[0, 0, 0]
    expected = np.empty((1, 48, 64, 3), dtype=np.uint8)
    expected[:] = reference
    np.testing.assert_array_equal(reader.decode(), expected, strict=True)
