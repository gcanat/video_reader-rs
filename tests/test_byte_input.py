import gc
import shutil
import subprocess
import sys
import weakref
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from video_reader import PyVideoReader

ASSETS = Path(__file__).resolve().parents[1] / "assets"
VIDEO = str(ASSETS / "input.mp4")


@pytest.fixture(scope="module")
def data():
    return (ASSETS / "input.mp4").read_bytes()


@pytest.fixture(scope="module")
def reference():
    return PyVideoReader(VIDEO, threads=1)


@pytest.fixture(scope="module")
def frames(reference):
    return reference.get_batch(list(range(len(reference))), with_fallback=True)


@pytest.fixture(params=[bytes, bytearray, BytesIO], ids=lambda factory: factory.__name__)
def source_factory(request):
    return request.param


def test_metadata_matches_file(source_factory, data, reference, frames):
    reader = PyVideoReader(source_factory(data), threads=1)
    assert len(reader) == len(frames)
    assert reader.get_shape() == list(frames.shape[:3])
    assert reader.get_pts() == reference.get_pts()
    assert reader.get_info() == reference.get_info()


def test_slices_and_repeated_seeks_match_file(source_factory, data, frames, reference):
    reader = PyVideoReader(source_factory(data), threads=1)
    for key in (slice(-3, -1), slice(None, None, -1), slice(2, 2), slice(8, 2, -2), slice(0, 3)):
        np.testing.assert_array_equal(reader[key], frames[key], strict=True)
        assert reader.get_pts(key) == reference.get_pts()[key]


def test_batch_and_single_frame_match_file(source_factory, data, frames):
    indices = [len(frames) - 1, 0, 5, 5, 1]
    reader = PyVideoReader(source_factory(data), threads=1)
    np.testing.assert_array_equal(reader.get_batch(indices), frames[indices], strict=True)
    np.testing.assert_array_equal(reader[-1], frames[-1], strict=True)


def test_decode_and_iteration_match_file(source_factory, data, frames):
    np.testing.assert_array_equal(PyVideoReader(source_factory(data), threads=1).decode(), frames, strict=True)
    np.testing.assert_array_equal(list(PyVideoReader(source_factory(data), threads=1)), frames, strict=True)


def test_async_decode_and_resize_match_file(source_factory, data):
    options = dict(threads=1, resize_shorter_side=120)
    expected = PyVideoReader(VIDEO, **options).decode_fast()
    np.testing.assert_array_equal(PyVideoReader(source_factory(data), **options).decode_fast(), expected, strict=True)


def test_bytesio_uses_full_contents_and_preserves_position(data, frames):
    stream = BytesIO(data)
    stream.seek(len(data))
    reader = PyVideoReader(stream, threads=1)
    assert stream.tell() == len(data)
    np.testing.assert_array_equal(reader[:3], frames[:3], strict=True)


def test_bytesio_can_be_changed_and_closed_after_construction(data, frames):
    stream = BytesIO(data)
    reader = PyVideoReader(stream, threads=1)
    stream.write(b"\x00" * len(data))
    stream.close()
    np.testing.assert_array_equal(reader[-3:], frames[-3:], strict=True)


def test_bytearray_is_snapshotted(data, frames):
    mutable = bytearray(data)
    reader = PyVideoReader(mutable, threads=1)
    mutable[:] = b"\x00" * len(mutable)
    np.testing.assert_array_equal(reader[:3], frames[:3], strict=True)


@pytest.mark.skipif(not hasattr(sys, "getrefcount"), reason="requires CPython reference counts")
def test_reader_keeps_bytes_alive_until_destruction(frames):
    data = (ASSETS / "input.mp4").read_bytes()
    references = sys.getrefcount(data)
    reader = PyVideoReader(data, threads=1)
    assert sys.getrefcount(data) == references + 1
    np.testing.assert_array_equal(reader[-3:], frames[-3:], strict=True)
    del reader
    assert sys.getrefcount(data) == references


def test_bytes_subclass_does_not_create_uncollectable_cycle(data, frames):
    class Blob(bytes):
        pass

    class Marker:
        pass

    blob = Blob(data)
    blob.marker = Marker()
    marker = weakref.ref(blob.marker)
    blob.reader = PyVideoReader(blob, threads=1)
    np.testing.assert_array_equal(blob.reader[-3:], frames[-3:], strict=True)
    del blob
    gc.collect()
    assert marker() is None


def test_filename_keyword_still_accepts_paths(frames):
    reader = PyVideoReader(filename=VIDEO, threads=1)
    np.testing.assert_array_equal(reader[:3], frames[:3], strict=True)


@pytest.mark.parametrize("invalid", [b"", b"not a video", (ASSETS / "audio_only.mp3").read_bytes()], ids=["empty", "invalid", "audio"])
def test_invalid_media_raises(source_factory, invalid):
    with pytest.raises(RuntimeError):
        PyVideoReader(source_factory(invalid), threads=1)


def test_memory_input_rejects_network_protocols():
    sdp = (
        b"v=0\no=- 0 0 IN IP4 127.0.0.1\ns=Protocol test\n"
        b"c=IN IP4 127.0.0.1\nt=0 0\nm=video 59138 RTP/AVP 96\n"
        b"a=rtpmap:96 H264/90000\n"
    )
    # A missing whitelist can block in the constructor, so isolate it with a timeout.
    script = f"""
from io import BytesIO
import pytest
from video_reader import PyVideoReader
for source in (bytes, bytearray, BytesIO):
    with pytest.raises(RuntimeError):
        PyVideoReader(source({sdp!r}))
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "not on whitelist" in result.stderr


def test_memory_input_rejects_local_file_references(tmp_path):
    shutil.copyfile(VIDEO, tmp_path / "victim.mp4")
    script = """
from io import BytesIO
import pytest
from video_reader import PyVideoReader
for source in (bytes, bytearray, BytesIO):
    with pytest.raises(RuntimeError):
        PyVideoReader(source(b'ffconcat version 1.0\\nfile victim.mp4\\n')).decode()
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "not on whitelist" in result.stderr


@pytest.mark.parametrize("options", [
    {"filter": "nosuchfilter=1"},
    {"resize_shorter_side": 100, "target_width": 64, "target_height": 48},
], ids=["filter", "resize-conflict"])
def test_decoder_errors_do_not_suggest_invalid_bytes(source_factory, data, options):
    with pytest.raises(RuntimeError) as error:
        PyVideoReader(source_factory(data), **options)
    assert "Bytes must contain" not in str(error.value)


@pytest.mark.parametrize("in_memory", [False, True], ids=["path", "bytes"])
@pytest.mark.parametrize("use_slice", [False, True], ids=["batch", "slice"])
def test_sequential_batch_after_iteration_uses_requested_indices(data, frames, in_memory, use_slice):
    reader = PyVideoReader(data if in_memory else VIDEO, threads=1)
    reader.get_batch(list(range(10)), with_fallback=True)
    for _ in range(5):
        next(reader)
    actual = reader[10:20] if use_slice else reader.get_batch(list(range(10, 20)), with_fallback=True)
    np.testing.assert_array_equal(actual, frames[10:20], strict=True)


@pytest.mark.parametrize("in_memory", [False, True], ids=["path", "bytes"])
def test_iteration_can_restart_after_random_access_drains_decoder(data, frames, in_memory):
    reader = PyVideoReader(data if in_memory else VIDEO, threads=1)
    np.testing.assert_array_equal(reader[-1], frames[-1], strict=True)
    # Iteration continues at the current position and rewinds on exhaustion.
    assert list(reader) == []
    np.testing.assert_array_equal(list(reader), frames, strict=True)


def test_empty_batch_keeps_a_stale_cursor_invalid(frames):
    reader = PyVideoReader(VIDEO, threads=1)
    for _ in range(5):
        next(reader)
    assert len(reader.get_batch([], with_fallback=False)) == 0
    np.testing.assert_array_equal(
        reader.get_batch(list(range(10, 20)), with_fallback=True), frames[10:20], strict=True
    )


def test_iteration_restarts_after_partial_pass_and_count(frames):
    reader = PyVideoReader(VIDEO, threads=1)
    for _ in range(len(frames) - 1):
        next(reader)
    assert reader.count_actual_frames() == len(frames)
    np.testing.assert_array_equal(list(reader), frames, strict=True)


def test_invalid_input_type_raises(data):
    with open(VIDEO, "rb") as file:
        for source in (None, 123, object(), [1, 2, 3], Path(VIDEO), memoryview(data), file):
            name = type(source).__name__
            with pytest.raises(TypeError, match=f"must be a str, bytes, bytearray, or io.BytesIO, got {name}"):
                PyVideoReader(source, oob_mode="invalid", log_level="trace")


def test_surrogate_path_preserves_unicode_error():
    with pytest.raises(UnicodeEncodeError):
        PyVideoReader("caf\udce9.mp4")


def test_nul_path_and_filter_raise_without_aborting(tmp_path):
    script = f"""
from pathlib import Path
import pytest
from video_reader import PyVideoReader
for source, options in (
    ('a\\x00b.mp4', {{}}),
    ({VIDEO!r}, {{'filter': 'scale=w=64:h=64\\x00x'}}),
    (Path({VIDEO!r}).read_bytes(), {{'filter': 'scale=w=64:h=64\\x00x'}}),
):
    with pytest.raises(ValueError, match='embedded null byte'):
        PyVideoReader(source, **options)
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_bytes_path_error_explains_input_types():
    with pytest.raises(RuntimeError, match="pass filesystem paths as str"):
        PyVideoReader(VIDEO.encode())


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires the FFmpeg executable")
def test_damaged_transport_stream_recovers_after_demuxer_errors(tmp_path, frames):
    path = tmp_path / "damaged.ts"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-stream_loop", "9", "-i", VIDEO, "-c", "copy", "-f", "mpegts", str(path)],
        check=True, capture_output=True, timeout=30,
    )
    data = bytearray(path.read_bytes())
    middle = len(data) // 2 // 188 * 188
    data[middle:middle + 131072] = bytes(131072)
    path.write_bytes(data)
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-c", "copy", "-f", "framecrc", "-"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    # One line per packet. FFmpeg 6.1 and 7.0 omit frame= from -progress for stream copy.
    expected = sum(not line.startswith("#") for line in result.stdout.splitlines())
    assert expected > len(frames) * 8, "FFmpeg must recover packets beyond the damaged region"
    for source in (str(path), bytes(data)):
        reader = PyVideoReader(source, threads=1)
        assert len(reader) == expected
        assert len(reader.get_pts()) == expected
        decoded = reader.decode()
        assert len(decoded) == expected
        assert np.all(np.any(decoded != 0, axis=(1, 2, 3))), "decode must not pad a lost tail with black frames"
        count = 0
        for index, frame in enumerate(reader):
            np.testing.assert_array_equal(frame, decoded[index], strict=True)
            count += 1
        assert count == expected
        assert reader.count_actual_frames() == expected
        np.testing.assert_array_equal(reader.get_batch([expected - 1], with_fallback=True)[0], decoded[-1], strict=True)
        fast = reader.decode_fast()
        assert len(fast) == expected
        for actual, reference in zip(fast, decoded):
            np.testing.assert_array_equal(actual, reference, strict=True)


def test_closed_bytesio_raises(data):
    stream = BytesIO(data)
    stream.close()
    with pytest.raises(ValueError):
        PyVideoReader(stream)
