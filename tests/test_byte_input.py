import gc
import shutil
import subprocess
import sys
import tempfile
import unittest
import weakref
from io import BytesIO
from pathlib import Path

import numpy as np
from video_reader import PyVideoReader


ASSETS = Path(__file__).resolve().parents[1] / "assets"
SOURCES = (bytes, bytearray, BytesIO)


class ByteInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = (ASSETS / "input.mp4").read_bytes()
        cls.path = str(ASSETS / "input.mp4")
        reference = PyVideoReader(cls.path, threads=1)
        cls.frames = reference.get_batch(list(range(len(reference))), with_fallback=True)
        cls.timestamps = reference.get_pts()
        cls.info = reference.get_info()

    def test_metadata_matches_file(self):
        for source in SOURCES:
            with self.subTest(source=source.__name__):
                reader = PyVideoReader(source(self.data), threads=1)
                self.assertEqual(len(reader), len(self.frames))
                self.assertEqual(reader.get_shape(), list(self.frames.shape[:3]))
                self.assertEqual(reader.get_pts(), self.timestamps)
                self.assertEqual(reader.get_info(), self.info)

    def test_slices_and_repeated_seeks_match_file(self):
        for source in SOURCES:
            reader = PyVideoReader(source(self.data), threads=1)
            for key in (slice(-3, -1), slice(None, None, -1), slice(2, 2), slice(8, 2, -2), slice(0, 3)):
                with self.subTest(source=source.__name__, key=key):
                    actual = reader[key]
                    self.assertEqual(actual.shape, self.frames[key].shape)
                    self.assertEqual(actual.dtype, self.frames.dtype)
                    np.testing.assert_array_equal(actual, self.frames[key])
                    self.assertEqual(reader.get_pts(key), self.timestamps[key])

    def test_batch_and_single_frame_match_file(self):
        indices = [len(self.frames) - 1, 0, 5, 5, 1]
        for source in SOURCES:
            with self.subTest(source=source.__name__):
                reader = PyVideoReader(source(self.data), threads=1)
                np.testing.assert_array_equal(reader.get_batch(indices), self.frames[indices])
                np.testing.assert_array_equal(reader[-1], self.frames[-1])

    def test_decode_and_iteration_match_file(self):
        for source in SOURCES:
            with self.subTest(source=source.__name__):
                np.testing.assert_array_equal(PyVideoReader(source(self.data), threads=1).decode(), self.frames)
                np.testing.assert_array_equal(list(PyVideoReader(source(self.data), threads=1)), self.frames)

    def test_async_decode_and_resize_match_file(self):
        options = dict(threads=1, resize_shorter_side=120)
        expected = PyVideoReader(self.path, **options).decode_fast()
        for source in SOURCES:
            with self.subTest(source=source.__name__):
                np.testing.assert_array_equal(PyVideoReader(source(self.data), **options).decode_fast(), expected)

    def test_bytesio_uses_full_contents_and_preserves_position(self):
        stream = BytesIO(self.data)
        stream.seek(len(self.data))
        reader = PyVideoReader(stream, threads=1)
        self.assertEqual(stream.tell(), len(self.data))
        np.testing.assert_array_equal(reader[:3], self.frames[:3])

    def test_bytesio_can_be_changed_and_closed_after_construction(self):
        stream = BytesIO(self.data)
        reader = PyVideoReader(stream, threads=1)
        stream.write(b"\x00" * len(self.data))
        stream.close()
        np.testing.assert_array_equal(reader[-3:], self.frames[-3:])

    def test_bytearray_is_snapshotted(self):
        data = bytearray(self.data)
        reader = PyVideoReader(data, threads=1)
        data[:] = b"\x00" * len(data)
        np.testing.assert_array_equal(reader[:3], self.frames[:3])

    @unittest.skipUnless(hasattr(sys, "getrefcount"), "requires CPython reference counts")
    def test_reader_keeps_bytes_alive_until_destruction(self):
        data = (ASSETS / "input.mp4").read_bytes()
        references = sys.getrefcount(data)
        reader = PyVideoReader(data, threads=1)
        self.assertEqual(sys.getrefcount(data), references + 1)
        np.testing.assert_array_equal(reader[-3:], self.frames[-3:])
        del reader
        self.assertEqual(sys.getrefcount(data), references)

    def test_bytes_subclass_does_not_create_uncollectable_cycle(self):
        class Blob(bytes):
            pass

        class Marker:
            pass

        blob = Blob(self.data)
        blob.marker = Marker()
        marker = weakref.ref(blob.marker)
        blob.reader = PyVideoReader(blob, threads=1)
        np.testing.assert_array_equal(blob.reader[-3:], self.frames[-3:])
        del blob
        gc.collect()
        self.assertIsNone(marker())

    def test_filename_keyword_still_accepts_paths(self):
        reader = PyVideoReader(filename=self.path, threads=1)
        np.testing.assert_array_equal(reader[:3], self.frames[:3])

    def test_invalid_media_raises(self):
        for data in (b"", b"not a video", (ASSETS / "audio_only.mp3").read_bytes()):
            for source in SOURCES:
                with self.subTest(source=source.__name__, size=len(data)):
                    with self.assertRaises(RuntimeError):
                        PyVideoReader(source(data), threads=1)

    def test_memory_input_rejects_network_protocols(self):
        sdp = (
            b"v=0\no=- 0 0 IN IP4 127.0.0.1\ns=Protocol test\n"
            b"c=IN IP4 127.0.0.1\nt=0 0\nm=video 59138 RTP/AVP 96\n"
            b"a=rtpmap:96 H264/90000\n"
        )
        # A missing whitelist can block in the constructor, so isolate it with a timeout.
        script = f"""
from io import BytesIO
from video_reader import PyVideoReader
for source in (bytes, bytearray, BytesIO):
    try:
        PyVideoReader(source({sdp!r}))
    except RuntimeError:
        pass
    else:
        raise AssertionError('network protocol accepted')
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("not on whitelist", result.stderr)

    def test_invalid_input_type_raises(self):
        with open(self.path, "rb") as file:
            for source in (None, 123, object(), [1, 2, 3], Path(self.path), memoryview(self.data), file):
                name = type(source).__name__
                with self.subTest(source=name):
                    with self.assertRaisesRegex(TypeError, f"must be a str, bytes, bytearray, or io.BytesIO, got {name}"):
                        PyVideoReader(source, oob_mode="invalid", log_level="trace")

    def test_surrogate_path_preserves_unicode_error(self):
        with self.assertRaises(UnicodeEncodeError):
            PyVideoReader("caf\udce9.mp4")

    def test_nul_path_raises_without_aborting(self):
        script = """
from video_reader import PyVideoReader
try:
    PyVideoReader('a\\x00b.mp4')
except ValueError as error:
    assert 'embedded null byte' in str(error)
else:
    raise AssertionError('NUL path accepted')
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_bytes_path_error_explains_input_types(self):
        with self.assertRaisesRegex(RuntimeError, "pass filesystem paths as str"):
            PyVideoReader(self.path.encode())

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires the FFmpeg executable")
    def test_damaged_transport_stream_recovers_after_demuxer_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "damaged.ts"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-stream_loop", "9", "-i", self.path, "-c", "copy", "-f", "mpegts", str(path)],
                check=True, capture_output=True, timeout=30,
            )
            data = bytearray(path.read_bytes())
            middle = len(data) // 2 // 188 * 188
            data[middle:middle + 131072] = bytes(131072)
            path.write_bytes(data)
            result = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-c", "copy", "-f", "null", "-",
                 "-progress", "pipe:1", "-nostats"],
                check=True, capture_output=True, text=True, timeout=30,
            )
            expected = int([line.split("=")[1] for line in result.stdout.splitlines() if line.startswith("frame=")][-1])
            self.assertGreater(expected, len(self.frames) * 8, "FFmpeg must recover packets beyond the damaged region")
            for source in (str(path), bytes(data)):
                with self.subTest(source=type(source).__name__):
                    reader = PyVideoReader(source, threads=1)
                    self.assertEqual(len(reader), expected)
                    self.assertEqual(len(reader.get_pts()), expected)

    def test_closed_bytesio_raises(self):
        stream = BytesIO(self.data)
        stream.close()
        with self.assertRaises(ValueError):
            PyVideoReader(stream)
