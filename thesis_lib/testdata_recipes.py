# all generators defined here should return a source table with the following columns
# x,y are in pixel scale
# TODO maybe enforce with astropy.units
# TODO calculate flux as well

from typing import Tuple, Optional, Callable, Union

import numpy as np
from astropy.modeling.functional_models import Gaussian2D
from tqdm import tqdm
import os

import scopesim_templates
from astropy.convolution import Gaussian2DKernel, Kernel2D, convolve_fft
from astropy.table import Table
from photutils import FittableImageModel

from .saturation_model import SaturationModel, read_scopesim_linearity
from .scopesim_helper import to_pixel_scale, pixel_scale, setup_optical_train, make_anisocado_model, filter_name, \
    pixel_to_mas, max_pixel_coord, make_psf, cancel_psf_pixel_shift
from .util import flux_to_magnitude, magnitude_to_flux
from .astrometry_types import X,Y,FLUX,MAGNITUDE, INPUT_TABLE_NAMES
from .config import Config

COLUMN_NAMES = (INPUT_TABLE_NAMES[X],
                INPUT_TABLE_NAMES[Y],
                INPUT_TABLE_NAMES[FLUX],
                INPUT_TABLE_NAMES[MAGNITUDE])


def scopesim_grid(N1d: int = 16,
                  seed: int = 1000,
                  border=64,
                  perturbation: float = 0.,
                  magnitude=lambda N: N * [18],
                  psf_transform=lambda x: x,
                  custom_subpixel_psf=None) \
        -> Tuple[np.ndarray, Table]:
    """
    Use scopesim to create a regular grid of stars
    :param N1d:  Grid of N1d x N1d Stars will be generated
    :param seed: initalize RNG for predictable results
    :param border: how many pixels on the edge to leave empty
    :param perturbation: perturb each star position with a uniform random pixel offset
    :return: image and input catalogue
    """
    np.random.seed(seed)

    N = N1d ** 2
    spectral_types = ['A0V'] * N

    y = pixel_to_mas(np.tile(np.linspace(border, max_pixel_coord.value - border, N1d), reps=(N1d, 1)))
    x = y.T.copy()
    x += np.random.uniform(0, perturbation * pixel_scale.value, x.shape)
    y += np.random.uniform(0, perturbation * pixel_scale.value, y.shape)

    m = np.array(magnitude(N))

    source = scopesim_templates.basic.stars.stars(filter_name=filter_name,
                                                  amplitudes=m,
                                                  spec_types=spectral_types,
                                                  x=x.ravel(), y=y.ravel())
    if custom_subpixel_psf:
        detector = setup_optical_train(custom_subpixel_psf=custom_subpixel_psf)
    else:
        detector = setup_optical_train(psf_effect=make_psf(transform=psf_transform))

    detector.observe(source, random_seed=seed, update=True)
    observed_image = detector.readout()[0][1].data

    table = Table((cancel_psf_pixel_shift(to_pixel_scale(x)).ravel(),
                   cancel_psf_pixel_shift(to_pixel_scale(y)).ravel(),
                   magnitude_to_flux(m), m), names=COLUMN_NAMES)
    return observed_image, table


max_count = 20000
noise = 57


def _gaussian_cluster_coords(N: int = 1000,
                             seed: int = 9999,
                             magnitude=lambda N: np.random.normal(21, 2, N)
                             ):
    np.random.seed(seed)
    x = np.random.normal(0, 1, N)
    y = np.random.normal(0, 1, N)
    m = magnitude(N)
    x_in_px = to_pixel_scale(x)
    y_in_px = to_pixel_scale(y)

    mask = (m > 14) * (0 < x_in_px) * (x_in_px < 1024) * (0 < y_in_px) * (y_in_px < 1024)
    x = x[mask]
    y = y[mask]
    m = m[mask]

    assert (len(x) == len(y) and len(m) == len(x))

    return x, y, m


def gaussian_cluster_modeladd(N: int = 1000,
                              seed: int = 9999,
                              magnitude=lambda N: np.random.normal(21, 2, N),
                              psf_model: Optional[Callable] = None,
                              saturation: bool = True):
    xs, ys, ms = _gaussian_cluster_coords(N, seed, magnitude)

    img = np.zeros((1024, 1024))
    if not psf_model:
        psf_model = make_anisocado_model()

    normalize = np.sum(psf_model.oversampling)

    fluxes = 6.55328584e+11 * 10 ** (-0.4 * ms) * 15  # the 15 is pulled from thin air
    xs = to_pixel_scale(xs)
    ys = to_pixel_scale(ys)

    for x, y, f in tqdm(list(zip(xs, ys, fluxes))):
        psf_model.x_0 = x
        psf_model.y_0 = y
        psf_model.flux = f * normalize  # TODO That's not really the correct value for mag->flux

        psf_model.render(img)

    # taken from statistics on empty scopesim image
    img = np.random.poisson(img) + np.random.normal(3164.272322010335, 58, img.shape)
    if saturation:
        scopesim_wdir = Config.instance().scopesim_working_dir
        img = SaturationModel(read_scopesim_linearity(os.path.join(scopesim_wdir, 'MICADO/FPA_linearity.dat'))).evaluate(img)

    table = Table((xs.ravel(), ys.ravel(), fluxes, ms), names=COLUMN_NAMES)
    return img, table


def gaussian_cluster(N: int = 1000,
                     seed: int = 9999,
                     magnitude=lambda N: np.random.normal(21, 2, N),
                     psf_transform=lambda x: x,
                     custom_subpixel_psf=None) -> Tuple[np.ndarray, Table]:
    """
    Emulates custom cluster creation from initial simcado script.
    Stars with gaussian position and magnitude distribution
    :param N: how many stars
    :param seed: RNG initializer
    :param psf_transform: function modifying the psf array
    :return: image and input catalogue
    """
    x, y, m = _gaussian_cluster_coords(N, seed, magnitude)

    Nprime = len(x)
    filter_name = 'MICADO/filters/TC_filter_K-cont.dat'  # TODO: how to make system find this?

    # That's what scopesim seemed to use for all stars.
    spectral_types = ['A0V'] * Nprime

    source = scopesim_templates.basic.stars.stars(filter_name=filter_name,
                                                  amplitudes=m,
                                                  spec_types=spectral_types,
                                                  x=x, y=y)

    if custom_subpixel_psf:
        detector = setup_optical_train(custom_subpixel_psf=custom_subpixel_psf)
    else:
        detector = setup_optical_train(psf_effect=make_psf(transform=psf_transform))

    detector.observe(source, random_seed=seed, update=True)
    observed_image = detector.readout()[0][1].data

    table = Table((cancel_psf_pixel_shift(to_pixel_scale(x)).ravel(),
                   cancel_psf_pixel_shift(to_pixel_scale(y)).ravel(),
                   magnitude_to_flux(m), m), names=COLUMN_NAMES)
    return observed_image, table


def scopesim_cluster(seed: int = 9999, custom_subpixel_psf=None) -> Tuple[np.ndarray, Table]:
    """
    Use the scopesim_template to create an image of a star cluster that matches the interfaces of the
    other functions here
    :param seed: RNG initializer
    :return: image and input catalogue
    """
    source = scopesim_templates.basic.stars.cluster(mass=1000,  # Msun
                                                    distance=50000,  # parsec
                                                    core_radius=0.3,  # parsec
                                                    seed=seed)

    detector = setup_optical_train(psf_effect=make_psf(),custom_subpixel_psf=custom_subpixel_psf)

    detector.observe(source, random_seed=seed, update=True)
    observed_image = detector.readout()[0][1].data

    source_table = source.fields[0]
    xs = to_pixel_scale(source_table['x']).ravel()
    ys = to_pixel_scale(source_table['y']).ravel()
    ms = source_table['weight']  # TODO these don't really correspond, do they?
    fluxes = magnitude_to_flux(ms)
    return_table = Table((cancel_psf_pixel_shift(xs), cancel_psf_pixel_shift(ys), fluxes, ms), names=COLUMN_NAMES)

    return observed_image, return_table


def convolved_grid(N1d: int = 16,
                   border: int = 64,
                   size: int = 1024,
                   kernel: Union[Kernel2D, None] = Gaussian2DKernel(x_stddev=1, x_size=201, y_size=201),
                   perturbation: float = 0.,
                   seed: int = 1000) -> Tuple[np.ndarray, Table]:
    """
    Place point sources on a regular image grid and convolve with a kernel to simulate PSF.
    No noise, distortion etc.

    :param N1d:  Grid of N1d x N1d Stars will be generated
    :param border: how many pixels on the edge to leave empty
    :param kernel: What to convolve the point sources with
    :param perturbation: random uniform offset to star positions
    :param seed: RNG initializer
    :return: image, input catalogue
    """
    # Kernel should always be an odd image or else we introduce some shift in the image
    np.random.seed(seed)

    data = np.zeros((size, size))

    idx_float = np.linspace(0 + border, size - border, N1d)
    x_float = np.tile(idx_float, reps=(N1d, 1))
    y_float = x_float.T.copy()  # just a view of x_float
    # these two modify same array...
    x_float += np.random.uniform(0, perturbation, x_float.shape)
    y_float += np.random.uniform(0, perturbation, y_float.shape)

    x, x_frac = np.divmod(x_float, 1)
    y, y_frac = np.divmod(y_float, 1)
    x, y = x.astype(int), y.astype(int)
    # Y U so ugly sometimes PEP8?
    data[y, x] = (1 - x_frac) * (1 - y_frac)
    # noinspection PyRedundantParentheses
    data[y + 1, x] = (1 - x_frac) * (y_frac)
    # noinspection PyRedundantParentheses
    data[y, x + 1] = (x_frac) * (1 - y_frac)
    data[y + 1, x + 1] = y_frac * x_frac

    if kernel is not None:
        # noinspection PyTypeChecker
        data = convolve_fft(data, kernel)
    # TODO the no-zeros seem like an awful hack
    data = data / np.max(data) + 0.001  # normalize and add tiny offset to have no zeros in data
    fluxes = np.ones(x.size)
    table = Table((x_float.ravel(), y_float.ravel(), fluxes, flux_to_magnitude(fluxes)), names=COLUMN_NAMES)
    return data, table


def model_add_grid(model: FittableImageModel,
                   N1d: int = 16,
                   flux_func=lambda N: N * [15000],
                   noise_σ=0,
                   size: int = 1024,
                   border: int = 64,
                   perturbation: float = 0.,
                   seed: int = 1000):
    """

    :param model: Assumed to be centered around x,y = (0,0)
    :param N1d:
    :param size:
    :param border:
    :param perturbation:
    :param seed:
    :return:
    """
    np.random.seed(seed)

    # list of sources to generate, shape: [[y0,y1...], [x0,x1...]]
    yx_sources = np.mgrid[0 + border:size - border:N1d * 1j, 0 + border:size - border:N1d * 1j].transpose(
        (1, 2, 0)).reshape(-1, 2)
    yx_sources += np.random.uniform(0, perturbation, (N1d ** 2, 2))

    fluxes = flux_func(N1d ** 2)

    # Too much memory...
    ## magic: Marry (2, size, size) to (2, N1D) to allow broadcasting
    ##    (2, 1,   size, size)
    ##    (2, N1d, 1   , 1)
    ## -> (2, N1d, size, size)
    # ysxs = yx_template[:, None, :, :] - yx_sources[..., None, None]

    data = np.zeros((size, size))

    normalize = np.sum(model.oversampling) / np.max(model.data)

    for (y, x), flux in zip(yx_sources, fluxes):
        model.x_0 = x
        model.y_0 = y
        model.flux = flux * normalize
        model.render(data)

    data = np.random.poisson(data) + np.random.normal(0, noise_σ, data.shape)

    table = Table((yx_sources[:, 1], yx_sources[:, 0], fluxes, flux_to_magnitude(fluxes)), names=COLUMN_NAMES)

    return data, table


def single_star_image(seed: int = 9999, custom_subpixel_psf=None) -> Tuple[np.ndarray, Table]:
    """
    TODO This should return a table as well...
    """

    x = np.array([0.])
    y = np.array([0.])
    m = np.array([16.])

    filter_name = 'MICADO/filters/TC_filter_K-cont.dat'  # TODO: how to make system find this?
    spectral_types = ['A0V']

    source = scopesim_templates.basic.stars.stars(filter_name=filter_name,
                                                  amplitudes=m,
                                                  spec_types=spectral_types,
                                                  x=x, y=y)

    detector = setup_optical_train(custom_subpixel_psf=custom_subpixel_psf)

    detector.observe(source, random_seed=seed, update=True)
    observed_image = detector.readout()[0][1].data

    return observed_image


def empty_image(seed: int = 1000) -> Tuple[np.ndarray, Table]:
    source = scopesim_templates.basic.misc.empty_sky()
    detector = setup_optical_train()
    detector.observe(source, random_seed=seed, update=True)
    observed_image = detector.readout()[0][1].data
    return observed_image, Table(names=COLUMN_NAMES)


def scopesim_groups(N1d: int = 16,
                    seed: int = 1000,
                    border=64,
                    jitter=0.,
                    magnitude=lambda N: N * [18],
                    group_size=2,
                    group_radius=5,
                    psf_transform=lambda x: x,
                    custom_subpixel_psf=None) \
        -> Tuple[np.ndarray, Table]:
    np.random.seed(seed)

    N = N1d ** 2
    spectral_types = ['A0V'] * N * group_size

    x_center, y_center = (np.mgrid[border:max_pixel_coord.value - border:N1d * 1j,
                          border:max_pixel_coord.value - border:N1d * 1j] - max_pixel_coord.value / 2) * pixel_scale.value

    x, y = [], []

    θ = np.linspace(0, 2 * np.pi, group_size, endpoint=False)

    for x_c, y_c in zip(x_center.ravel(), y_center.ravel()):
        θ_perturbed = θ + np.random.uniform(0, 2 * np.pi)
        x_angle_offset = np.cos(θ_perturbed) * group_radius
        y_angle_offset = np.sin(θ_perturbed) * group_radius
        x_jitter = np.random.uniform(-jitter / 2, jitter / 2, len(θ))
        y_jitter = np.random.uniform(-jitter / 2, jitter / 2, len(θ))
        x += list(x_c + (x_angle_offset + x_jitter) * pixel_scale.value)
        y += list(y_c + (y_angle_offset + y_jitter) * pixel_scale.value)

    m = np.array(magnitude(N * group_size))

    source = scopesim_templates.basic.stars.stars(filter_name=filter_name,
                                                  amplitudes=m,
                                                  spec_types=spectral_types,
                                                  x=x, y=y)
    if custom_subpixel_psf:
        detector = setup_optical_train(custom_subpixel_psf=custom_subpixel_psf)
    else:
        detector = setup_optical_train(psf_effect=make_psf(transform=psf_transform))

    detector.observe(source, random_seed=seed, updatephotometry_iterations=True)
    observed_image = detector.readout()[0][1].data

    table = Table((cancel_psf_pixel_shift(to_pixel_scale(x)).ravel(),
                   cancel_psf_pixel_shift(to_pixel_scale(y)).ravel(),
                   magnitude_to_flux(m), m), names=COLUMN_NAMES)
    return observed_image, table


def one_source_testimage():
    ygrid, xgrid = np.indices((41, 41))
    model = Gaussian2D(x_mean=20, y_mean=20, x_stddev=1.5, y_stddev=1.5)
    return model(xgrid, ygrid), Table(([20], [20], [1], [1]), names=COLUMN_NAMES)


def multi_source_testimage():
    img = np.zeros((301, 301))
    xs = [50, 50, 120.1, 100, 20]
    ys = [50,57,130,20.3,100.8]
    fluxes = [10,20,20,10,30]
    model = Gaussian2D(x_mean=0, y_mean=0, x_stddev=1.5, y_stddev=1.5)

    for x, y, f in zip(xs, ys, fluxes):
        model.x_mean = x
        model.y_mean = y
        model.amplitude = f
        model.render(img)

    tab = Table((xs, ys, fluxes, flux_to_magnitude(fluxes)), names=COLUMN_NAMES)

    return img, tab