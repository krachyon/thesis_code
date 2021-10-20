import pytest
from thesis_lib.util import estimate_fwhm, center_of_index, center_of_image, centered_grid, centered_grid_quadratic, centroid, copying_lru_cache

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


def test_centroid_sym():
    y, x = centered_grid_quadratic(51)
    img = Gaussian2D()(x, y)
    assert np.all(np.isclose(centroid(img), center_of_image(img)))

    y, x = centered_grid((51, 71))
    img = Gaussian2D()(x, y)
    assert np.all(np.isclose(centroid(img), center_of_image(img)))


@pytest.mark.parametrize('xy', [(23, 28), (-10.3, 8.7), (-11, -7.2), (8.45, -5.98), (-30, -34), (-30, 34)])
def test_centroid_asym(xy):
    y_grid, x_grid = centered_grid_quadratic(101)
    xy = np.array(xy)
    xy_expected = xy + center_of_image(x_grid)

    img = Gaussian2D(x_mean=xy[0], y_mean=xy[1])(x_grid, y_grid)
    assert np.all(np.isclose(centroid(img), xy_expected))


def test_copying_lru():
    calls = []

    @copying_lru_cache()
    def get_list(i):
        calls.append(0)
        return list(range(i))

    get_list(5)
    result_list=get_list(5)

    assert len(calls) == 1
    result_list[0]=[100]

    result_list = get_list(5)
    assert result_list[0] != 100

    calls = []
    @copying_lru_cache(maxsize=2, typed=False)
    def get_square(foo):
        calls.append(0)
        return foo**2

    get_square(2)
    get_square(3)
    get_square(4)
    get_square(2)
    assert len(calls) == 4

    calls = []
    @copying_lru_cache(maxsize=2, typed=True)
    def get_square(foo):
        calls.append(0)
        return foo**2

    get_square(2)
    get_square(3)
    get_square(2.1)
    get_square(2)
    assert len(calls) == 4

