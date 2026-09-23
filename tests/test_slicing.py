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
        pytest.param(slice(-3, -1), id="negative-bounds"),
        pytest.param(slice(None, None, -1), id="reverse"),
        pytest.param(slice(-1, None, -7), id="reverse-step"),
        pytest.param(slice(None, -3), id="negative-stop"),
        pytest.param(slice(3, 17, 2), id="forward-step"),
        pytest.param(slice(8, 2, -2), id="reverse-range"),
        pytest.param(slice(-10**100, 10**100, 33), id="huge-bounds"),
        pytest.param(slice(10**100, -10**100, -33), id="huge-reverse-bounds"),
        pytest.param(slice(None, None, 10**100), id="huge-step"),
        pytest.param(slice(None, None, -10**100), id="huge-negative-step"),
        pytest.param(slice(10, 5), id="start-after-stop"),
        pytest.param(slice(2, 2), id="equal-bounds"),
        pytest.param(slice(5, 10, -1), id="empty-reverse-range"),
        pytest.param(slice(200, None), id="start-past-end"),
        pytest.param(slice(None, -200), id="stop-before-start"),
    ],
)
def test_slices_match_python_sequences(key, method, reference_outputs):
    actual = getattr(PyVideoReader(VIDEO), method)(key)
    expected = reference_outputs[method][key]
    if method == "get_pts":
        assert isinstance(actual, list)
        assert actual == expected
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("method", ["__getitem__", "get_pts"])
@pytest.mark.parametrize(
    "key, error",
    [
        pytest.param(slice(None, None, 0), ValueError, id="zero-step"),
        pytest.param(slice(0.5, None), TypeError, id="float-start"),
        pytest.param(slice(None, 5.0), TypeError, id="float-stop"),
        pytest.param(slice(None, None, 1.0), TypeError, id="float-step"),
    ],
)
def test_invalid_slices_raise_python_errors(key, error, method):
    reader = PyVideoReader(VIDEO)
    with pytest.raises(error):
        getattr(reader, method)(key)
