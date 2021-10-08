import anisocado
import numba
import numpy as np
from astropy.convolution import Kernel2D
from astropy.modeling.functional_models import Gaussian2D

from thesis_lib.scopesim_helper import max_pixel_coord
from thesis_lib.util import centered_grid


def gauss2d(σ_x=1., σ_y=1., a=1.):
    @numba.njit(fastmath=True)
    def inner(x: np.ndarray, y: np.ndarray):
        return a * np.exp(-x ** 2 / (2 * σ_x ** 2) + -y ** 2 / (2 * σ_y ** 2))

    return inner


# TODO this seems to break for std=0
def lowpass(std=5):
    def transform(data):
        y, x = centered_grid(data.shape)
        return data * Gaussian2D(x_stddev=std, y_stddev=std)(x, y)
    return transform


def expmag(N):
    dist = np.log(np.random.exponential(1, N))
    mag_target = 21
    dist_shift = dist - np.median(dist) + mag_target
    return dist_shift


def make_anisocado_kernel(shift=(0, 14), wavelength=2.15, pixel_count=max_pixel_coord.value):
    """
    Get a convolvable Kernel from anisocado to approximate field constant MICADO PSF
    :param shift: how far away from center are we?
    :param wavelength: PSF for what wavelength?
    :param pixel_count: How large the side lenght of the image should be
    :return: Convolution kernel
    """
    count = pixel_count + 1 if pixel_count % 2 == 0 else pixel_count
    count = int(count)
    hdus = anisocado.misc.make_simcado_psf_file(
        [shift], [wavelength], pixelSize=0.004, N=count)
    image = hdus[2]
    kernel = np.squeeze(image.data)
    return Kernel2D(array=kernel)
