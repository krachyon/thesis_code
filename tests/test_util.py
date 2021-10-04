import pytest
from thesis_lib.util import estimate_fwhm, center_of_index, center_of_image, centered_grid, centered_grid_quadratic, centroid

from astropy.modeling.functional_models import Gaussian2D
import numpy as np
from photutils import EPSFModel


class TestEstimateFWHM:
    @classmethod
    def setup_class(cls):
        cls.model = Gaussian2D(x_stddev=4,y_stddev=4, x_mean=50, y_mean=50)
        cls.data = np.zeros((101,101))
        cls.model.render(cls.data)

        cls.epsf = EPSFModel(data=cls.data)
        cls.epsf_os2 = EPSFModel(data=cls.data, oversampling=2)

    def test_estimate_fwhm_basic(self):
        est = estimate_fwhm(self.epsf)
        assert np.isclose(est, self.model.x_fwhm)
        assert np.isclose(est, self.model.y_fwhm)

    def test_estimate_fwhm_oversampling(self):
        est = estimate_fwhm(self.epsf_os2)
        assert np.isclose(est, self.model.x_fwhm/2)
        assert np.isclose(est, self.model.y_fwhm/2)


def test_center_of_index():
    # odd, middle pixel
    assert center_of_index(3) == 1
    # even, between pixels
    assert center_of_index(4) == 1.5
    #vectorization
    assert np.array_equal(center_of_index((3, 4, 5)), [1, 1.5, 2])


def test_center_of_image():
    img = np.zeros((5,5))
    assert center_of_image(img) == (2, 2)
    img = np.zeros((5,7))
    xcenter, ycenter = center_of_image(img)
    assert xcenter == 3
    assert ycenter == 2


def test_centered_grid_quadratic():
    size = 11
    y, x = centered_grid_quadratic(size)
    assert x.shape == (11,11)
    assert y.shape == (11,11)
    assert np.mean(y) == 0
    assert np.mean(x) == 0


def test_centered_grid():
    y, x = centered_grid((11, 15))
    assert y.shape == (11,15)
    assert x.shape == (11,15)
    assert np.mean(x) == 0
    assert np.mean(y) == 0


def test_centroid():
    y, x = centered_grid_quadratic(51)
    img = Gaussian2D()(x,y)
    assert np.all(np.isclose(centroid(img), center_of_image(img)))

    y, x = centered_grid((51, 71))
    img = Gaussian2D()(x, y)
    assert np.all(np.isclose(centroid(img), center_of_image(img)))