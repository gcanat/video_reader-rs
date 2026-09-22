import unittest
from pathlib import Path

import numpy as np
from video_reader import PyVideoReader


VIDEO = str(Path(__file__).resolve().parents[1] / "assets" / "input.mp4")


class SlicingTests(unittest.TestCase):
    def test_slices_match_python_sequences(self):
        reference = PyVideoReader(VIDEO)
        frames = reference.get_batch(list(range(len(reference))), with_fallback=True)
        timestamps = reference.get_pts()
        slices = [
            slice(-3, -1),
            slice(None, None, -1),
            slice(-1, None, -7),
            slice(None, -3),
            slice(3, 17, 2),
            slice(8, 2, -2),
            slice(-10**100, 10**100, 33),
            slice(10**100, -10**100, -33),
            slice(None, None, 10**100),
            slice(None, None, -10**100),
            slice(10, 5),
            slice(2, 2),
            slice(5, 10, -1),
            slice(200, None),
            slice(None, -200),
        ]
        for key in slices:
            with self.subTest(key=key, method="__getitem__"):
                np.testing.assert_array_equal(PyVideoReader(VIDEO)[key], frames[key])
            with self.subTest(key=key, method="get_pts"):
                self.assertEqual(PyVideoReader(VIDEO).get_pts(key), timestamps[key])

    def test_invalid_slices_raise_python_errors(self):
        for key, error in [
            (slice(None, None, 0), ValueError),
            (slice(0.5, None), TypeError),
            (slice(None, 5.0), TypeError),
            (slice(None, None, 1.0), TypeError),
        ]:
            reader = PyVideoReader(VIDEO)
            for method in [reader.__getitem__, reader.get_pts]:
                with self.subTest(key=key, method=method.__name__):
                    with self.assertRaises(error):
                        method(key)
