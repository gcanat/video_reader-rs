import gc
import unittest
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

    def test_reader_keeps_temporary_bytes_alive(self):
        reader = PyVideoReader((ASSETS / "input.mp4").read_bytes(), threads=1)
        gc.collect()
        np.testing.assert_array_equal(reader[-3:], self.frames[-3:])

    def test_filename_keyword_still_accepts_paths(self):
        reader = PyVideoReader(filename=self.path, threads=1)
        np.testing.assert_array_equal(reader[:3], self.frames[:3])

    def test_invalid_media_raises(self):
        for data in (b"", b"not a video", (ASSETS / "audio_only.mp3").read_bytes()):
            for source in SOURCES:
                with self.subTest(source=source.__name__, size=len(data)):
                    with self.assertRaises(RuntimeError):
                        PyVideoReader(source(data), threads=1)

    def test_invalid_input_type_raises(self):
        for source in (None, 123, object(), [1, 2, 3]):
            with self.subTest(source=type(source).__name__):
                with self.assertRaises(TypeError):
                    PyVideoReader(source)

    def test_closed_bytesio_raises(self):
        stream = BytesIO(self.data)
        stream.close()
        with self.assertRaises(ValueError):
            PyVideoReader(stream)
