from pathlib import Path

import numpy as np
import pytest
from video_reader import PyVideoReader


VIDEO = str(Path(__file__).resolve().parents[1] / "assets" / "input.mp4")


@pytest.fixture(scope="module")
def reference_outputs():
    reader = PyVideoReader(VIDEO)
    return {
        "__getitem__": reader.get_batch(list(range(len(reader))), with_fallback=True),
        "get_pts": reader.get_pts(),
    }


@pytest.mark.parametrize("method", ["__getitem__", "get_pts"])
@pytest.mark.parametrize(
    "key",
    [
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
    ],
)
def test_slices_match_python_sequences(key, method, reference_outputs):
    actual = getattr(PyVideoReader(VIDEO), method)(key)
    np.testing.assert_array_equal(actual, reference_outputs[method][key])


@pytest.mark.parametrize("method", ["__getitem__", "get_pts"])
@pytest.mark.parametrize(
    "key, error",
    [
        (slice(None, None, 0), ValueError),
        (slice(0.5, None), TypeError),
        (slice(None, 5.0), TypeError),
        (slice(None, None, 1.0), TypeError),
    ],
)
def test_invalid_slices_raise_python_errors(key, error, method):
    reader = PyVideoReader(VIDEO)
    with pytest.raises(error):
        getattr(reader, method)(key)
