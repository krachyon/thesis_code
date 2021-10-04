import pytest
import numpy as np

from thesis_lib.testdata_helpers import lowpass
from thesis_lib.util import centroid, center_of_image


@pytest.mark.parametrize('xsize', [51, 50, 101])
@pytest.mark.parametrize('ysize', [51, 50, 101])
def test_lowpass(xsize, ysize):
    img = np.ones((ysize, xsize))
    transform = lowpass(5)

    assert np.all(np.isclose(centroid(transform(img)), center_of_image(img)))
