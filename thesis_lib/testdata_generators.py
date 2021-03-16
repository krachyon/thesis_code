import multiprocessing
from collections import defaultdict
from os.path import exists, join
from typing import Callable, Tuple, Union

import anisocado
import numpy as np
import scopesim_templates
from astropy.convolution import AiryDisk2DKernel, Gaussian2DKernel, Kernel2D, convolve_fft
from astropy.io import fits
from astropy.io.fits import PrimaryHDU
from astropy.modeling.functional_models import Gaussian2D
from astropy.table import Table

from .config import Config
from .scopesim_helper import setup_optical_train, pixel_scale, pixel_count, filter_name, to_pixel_scale, make_psf

# all generators defined here should return a source table with the following columns
# x,y are in pixel scale
# TODO maybe enforce with astropy.units
names = ('x', 'y', 'm')


def scopesim_grid(N1d: int = 16,
                  seed: int = 1000,
                  border=64,
                  perturbation: float = 0.,
                  magnitude=lambda N: N * [18],
                  psf_transform=lambda x: x) \
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

    y = (np.tile(np.linspace(border, pixel_count - border, N1d), reps=(N1d, 1)) - pixel_count / 2) * pixel_scale
    x = y.T
    x += np.random.uniform(0, perturbation * pixel_scale, x.shape)
    y += np.random.uniform(0, perturbation * pixel_scale, y.shape)

    m = np.array(magnitude(N))

    source = scopesim_templates.basic.stars.stars(filter_name=filter_name,
                                                  amplitudes=m,
                                                  spec_types=spectral_types,
                                                  x=x.ravel(), y=y.ravel())
    detector = setup_optical_train(make_psf(transform=psf_transform))

    detector.observe(source, random_seed=seed, updatephotometry_iterations=True)
    observed_image = detector.readout()[0][1].data

    table = Table((to_pixel_scale(x).ravel(), to_pixel_scale(y).ravel(), m), names=names)
    return observed_image, table


def gaussian_cluster(N: int = 1000,
                     seed: int = 9999,
                     magnitude=lambda N: np.random.normal(21, 2, N),
                     psf_transform=lambda x: x) -> Tuple[np.ndarray, Table]:
    """
    Emulates custom cluster creation from initial simcado script.
    Stars with gaussian position and magnitude distribution
    :param N: how many stars
    :param seed: RNG initializer
    :param psf_transform: function modifying the psf array
    :return: image and input catalogue
    """
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
    Nprime = len(x)
    filter_name = 'MICADO/filters/TC_filter_K-cont.dat'  # TODO: how to make system find this?

    ## TODO: random spectral types, adapt this to a realistic cluster distribution or maybe just use
    ## scopesim_templates.basic.stars.cluster
    # random_spectral_types = np.random.choice(get_spectral_types(), Nprime)

    # That's what scopesim seemed to use for all stars.
    spectral_types = ['A0V'] * Nprime

    source = scopesim_templates.basic.stars.stars(filter_name=filter_name,
                                                  amplitudes=m,
                                                  spec_types=spectral_types,
                                                  x=x, y=y)

    detector = setup_optical_train(make_psf(transform=psf_transform))

    detector.observe(source, random_seed=seed, update=True)
    observed_image = detector.readout()[0][1].data

    table = Table((to_pixel_scale(x).ravel(), to_pixel_scale(y).ravel(), m), names=names)
    return observed_image, table


def scopesim_cluster(seed: int = 9999) -> Tuple[np.ndarray, Table]:
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
    detector = setup_optical_train()

    detector.observe(source, random_seed=seed, update=True)
    observed_image = detector.readout()[0][1].data

    table = source.fields[0]
    table['x'] = to_pixel_scale(table['x']).ravel()
    table['y'] = to_pixel_scale(table['y']).ravel()
    table['m'] = table['weight']  # TODO these don't really correspond, do they?

    return observed_image, table


def convolved_grid(N1d: int = 16,
                   border: int = 64,
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

    size = 1024
    data = np.zeros((size, size))

    idx_float = np.linspace(0 + border, size - border, N1d)
    x_float = np.tile(idx_float, reps=(N1d, 1))
    y_float = x_float.T
    x_float += np.random.uniform(0, perturbation, x_float.shape)
    y_float += np.random.uniform(0, perturbation, y_float.shape)

    x, x_frac = np.divmod(x_float, 1)
    y, y_frac = np.divmod(y_float, 1)
    x, y = x.astype(int), y.astype(int)
    # Y U so ugly sometimes PEP8?
    data[y, x] = (1 - x_frac) * (1 - y_frac)
    # noinspection PyRedundantParentheses
    data[y + 1, x] = (x_frac) * (1 - y_frac)
    # noinspection PyRedundantParentheses
    data[y, x + 1] = (1 - x_frac) * (y_frac)
    data[y + 1, x + 1] = y_frac * x_frac

    if kernel is not None:
        # noinspection PyTypeChecker
        data = convolve_fft(data, kernel)
    data = data / np.max(data) + 0.001  # normalize and add tiny offset to have no zeros in data

    table = Table((x_float.ravel(), y_float.ravel(), np.ones(x.size)), names=names)
    return data, table


def make_anisocado_kernel(shift=(0, 14), wavelength=2.15, pixel_count=pixel_count):
    """
    Get a convolvable Kernel from anisocado to approximate field constant MICADO PSF
    :param shift: how far away from center are we?
    :param wavelength: PSF for what wavelength?
    :param pixel_count: How large the side lenght of the image should be
    :return: Convolution kernel
    """
    count = pixel_count + 1 if pixel_count % 2 == 0 else pixel_count
    hdus = anisocado.misc.make_simcado_psf_file(
        [shift], [wavelength], pixelSize=0.004, N=count)
    image = hdus[2]
    kernel = np.squeeze(image.data)
    return Kernel2D(array=kernel)


def make_single_star_image(seed: int = 9999) -> Tuple[np.ndarray, Table]:
    """
    Emulates custom cluster creation from initial simcado script.
    Stars with gaussian position and magnitude distribution
    :param seed: RNG initializer
    :return: image and input catalogue
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

    detector = setup_optical_train()

    detector.observe(source, random_seed=seed, update=True)
    observed_image = detector.readout()[0][1].data

    return observed_image


def scopesim_groups(N1d: int = 16,
                    seed: int = 1000,
                    border=64,
                    jitter=0.,
                    magnitude=lambda N: N * [18],
                    group_size=2,
                    group_radius=5,
                    psf_transform=lambda x: x) \
        -> Tuple[np.ndarray, Table]:
    np.random.seed(seed)

    N = N1d ** 2
    spectral_types = ['A0V'] * N * group_size

    x_center, y_center = (np.mgrid[border:pixel_count - border:N1d * 1j,
                          border:pixel_count - border:N1d * 1j] - pixel_count / 2) * pixel_scale

    x, y = [], []

    θ = np.linspace(0, 2 * np.pi, group_size, endpoint=False)

    for x_c, y_c in zip(x_center.ravel(), y_center.ravel()):
        θ_perturbed = θ + np.random.uniform(0, 2 * np.pi)
        x_angle_offset = np.cos(θ_perturbed) * group_radius
        y_angle_offset = np.sin(θ_perturbed) * group_radius
        x_jitter = np.random.uniform(-jitter / 2, jitter / 2, len(θ))
        y_jitter = np.random.uniform(-jitter / 2, jitter / 2, len(θ))
        x += list(x_c + (x_angle_offset + x_jitter) * pixel_scale)
        y += list(y_c + (y_angle_offset + y_jitter) * pixel_scale)

    m = np.array(magnitude(N * group_size))

    source = scopesim_templates.basic.stars.stars(filter_name=filter_name,
                                                  amplitudes=m,
                                                  spec_types=spectral_types,
                                                  x=x, y=y)
    detector = setup_optical_train(make_psf(transform=psf_transform))

    detector.observe(source, random_seed=seed, updatephotometry_iterations=True)
    observed_image = detector.readout()[0][1].data

    table = Table((to_pixel_scale(x).ravel(), to_pixel_scale(y).ravel(), m), names=names)
    return observed_image, table


def lowpass(std=5):
    def transform(data):
        x, y = data.shape
        # psf array is offset by one, hence need the -1 after coordinates
        return data * Gaussian2D(x_stddev=std, y_stddev=std)(*np.mgrid[-x / 2:x / 2:x * 1j, -y / 2:y / 2:y * 1j] - 1)

    return transform


# make sure concurrent calls with the same filename don't tread on each other's toes.
# only generate/write file once
filename_locks = defaultdict(lambda: multiprocessing.Lock())


def read_or_generate_image(recipe: Callable[[], Tuple[np.ndarray, Table]],
                           filename_base: str,
                           directory: str = Config.instance().image_folder):
    """
    For the 'recipe' either generate and write the image+catalogue or read existing output from disk
    :param directory: where to put/read data
    :param filename_base: what the files are called, minus extension
    :param recipe: function generating your image and catalogue
    :return: image, input_catalogue
    """
    image_name = join(directory, filename_base + '.fits')
    table_name = join(directory, filename_base + '.dat')
    with filename_locks[filename_base]:
        if exists(image_name) and exists(table_name):
            img = fits.open(image_name)[0].data
            table = Table.read(table_name, format='ascii.ecsv')
        else:
            img, table = recipe()
            PrimaryHDU(img).writeto(image_name, overwrite=True)
            table.write(table_name, format='ascii.ecsv')

    return img, table


def read_or_generate_helper(recipe: Callable[[], Tuple[np.ndarray, Table]],
                            filename_base: str,
                            directory: str = Config.instance().image_folder):
    """
    For the 'recipe' either generate and write the image+catalogue or read existing output from disk
    :param directory: where to put/read data
    :param filename_base: what the files are called, minus extension
    :param recipe: function generating your image and catalogue
    :return: image, input_catalogue
    """
    image_name = join(directory, filename_base + '.fits')
    with filename_locks[filename_base]:
        if exists(image_name):
            img = fits.open(image_name)[0].data
        else:
            img = recipe()
            PrimaryHDU(img).writeto(image_name, overwrite=True)

    return img


# predefined recipes
kernel_size = 201  # should be enough for getting reasonable results
misc_images = {
    'gauss_cluster_N1000': lambda: gaussian_cluster(N=1000),
    'gauss_cluster_N1000_low': lambda: gaussian_cluster(N=1000, psf_transform=lowpass()),
    'scopesim_cluster': lambda: scopesim_cluster(),
    'gauss_grid_16_sigma1_perturb_0': lambda: convolved_grid(N1d=16),
    'gauss_grid_16_sigma1_perturb_2': lambda: convolved_grid(N1d=16, perturbation=2.),
    'gauss_grid_16_sigma5_perturb_2':
        lambda: convolved_grid(N1d=16, perturbation=2.,
                               kernel=Gaussian2DKernel(x_stddev=5, x_size=kernel_size, y_size=kernel_size)),
    'airy_grid_16_radius2_perturb_0':
        lambda: convolved_grid(N1d=16, perturbation=0.,
                               kernel=AiryDisk2DKernel(radius=2, x_size=kernel_size, y_size=kernel_size)),
    'airy_grid_16_radius2_perturb_2':
        lambda: convolved_grid(N1d=16, perturbation=2.,
                               kernel=AiryDisk2DKernel(radius=2, x_size=kernel_size, y_size=kernel_size)),
    'airy_grid_16_radius5_perturb_2':
        lambda: convolved_grid(N1d=16, perturbation=2.,
                               kernel=AiryDisk2DKernel(radius=5, x_size=kernel_size, y_size=kernel_size)),
    'anisocado_grid_16_perturb_0':
        lambda: convolved_grid(N1d=16, perturbation=0.,
                               kernel=make_anisocado_kernel()),
    'anisocado_grid_16_perturb_2':
        lambda: convolved_grid(N1d=16, perturbation=2.,
                               kernel=make_anisocado_kernel()),
    'grid_16_no_convolve_perturb2':
        lambda: convolved_grid(N1d=16, perturbation=2., kernel=None),
    'scopesim_grid_16_perturb0':
        lambda: scopesim_grid(N1d=16, perturbation=0.),
    'scopesim_grid_16_perturb2':
        lambda: scopesim_grid(N1d=16, perturbation=2.),
}


def expmag(N):
    dist = np.log(np.random.exponential(1, N))
    mag_target = 21
    dist_shift = dist - np.median(dist) + mag_target
    return dist_shift


lowpass_images = {
    # 'scopesim_grid_16_perturb2_low':
    #     lambda: scopesim_grid(N1d=16, perturbation=2., psf_transform=lowpass()),
    # 'scopesim_grid_16_perturb2_low_mag20':
    #     lambda: scopesim_grid(N1d=16, perturbation=2., psf_transform=lowpass(), magnitude=lambda N: N * [20]),
    # 'scopesim_grid_30_perturb2_low_mag22':
    #     lambda: scopesim_grid(N1d=30, perturbation=2., psf_transform=lowpass(), magnitude=lambda N: N * [22]),
    # 'scopesim_grid_30_perturb2_low_mag20':
    #     lambda: scopesim_grid(N1d=30, perturbation=2., psf_transform=lowpass(), magnitude=lambda N: N * [20]),
    'scopesim_grid_30_perturb2_low_mag18-24':
        lambda: scopesim_grid(N1d=30, perturbation=2., psf_transform=lowpass(),
                              magnitude=lambda N: np.random.uniform(18, 24, N)),
    'scopesim_groups_16_perturb_2_low_radius_7':
        lambda: scopesim_groups(N1d=16, jitter=2., psf_transform=lowpass(), magnitude=lambda N: N * [20],
                                group_radius=7, group_size=2),
    'scopesim_groups_16_perturb_2_low_radius_5':
        lambda: scopesim_groups(N1d=16, jitter=2., psf_transform=lowpass(), magnitude=lambda N: N * [20],
                                group_radius=10, group_size=5),
    'gausscluster_N2000_mag22':
        lambda: gaussian_cluster(2000, magnitude=lambda N: np.random.normal(22, 2, N), psf_transform=lowpass()),
    'gausscluster_N4000_expmag21':
        lambda: gaussian_cluster(4000, magnitude=expmag, psf_transform=lowpass())
}

helpers = {
    'anisocado_psf':
        lambda: make_anisocado_kernel().array,
    'single_star_image':
        lambda: make_single_star_image()
}
