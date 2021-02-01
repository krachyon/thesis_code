import numpy as np

import scopesim_templates

from astropy.io import fits
from astropy.io.fits import PrimaryHDU
from astropy.table import Table

from scopesim_helper import setup_optical_train, pixel_scale, pixel_count, filter_name, to_pixel_scale
from astropy.convolution import AiryDisk2DKernel, Gaussian2DKernel, Kernel2D, convolve_fft

from os.path import exists, join

from config import Config
from typing import Callable, Tuple, Union

import anisocado

# all generators defined here should return a source table with the following columns
# x,y are in pixel scale
# TODO maybe enforce with astropy.units
names = ('x', 'y', 'm')


def scopesim_grid(N1d: int = 16, seed: int = 1000, border=64, perturbation: float = 0.) \
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

    N = N1d**2
    spectral_types = ['A0V'] * N

    y = (np.tile(np.linspace(border, pixel_count-border, N1d), reps=(N1d, 1)) - pixel_count/2) * pixel_scale
    x = y.T
    x += np.random.uniform(0, perturbation*pixel_scale, x.shape)
    y += np.random.uniform(0, perturbation*pixel_scale, y.shape)

    m = np.array(N*[18])

    source = scopesim_templates.basic.stars.stars(filter_name=filter_name,
                                                  amplitudes=m,
                                                  spec_types=spectral_types,
                                                  x=x.ravel(), y=y.ravel())
    detector = setup_optical_train()

    detector.observe(source, random_seed=seed, updatephotometry_iterations=True)
    observed_image = detector.readout()[0][1].data

    table = Table((to_pixel_scale(x).ravel(), to_pixel_scale(y).ravel(), m), names=names)
    return observed_image, table


def gaussian_cluster(N: int = 1000, seed: int = 9999) \
        -> Tuple[np.ndarray, Table]:
    """
    Emulates custom cluster creation from initial simcado script.
    Stars with gaussian position and magnitude distribution
    :param seed: RNG initializer
    :return: image and input catalogue
    """
    # TODO could use more parameters e.g for magnitudes
    np.random.seed(seed)
    N = 1000
    x = np.random.normal(0, 1, N)
    y = np.random.normal(0, 1, N)
    m = np.random.normal(19, 2, N)
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

    detector = setup_optical_train()

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
    data = np.zeros((size,  size))

    idx_float = np.linspace(0+border, size-border, N1d)
    x_float = np.tile(idx_float, reps=(N1d, 1))
    y_float = x_float.T
    x_float += np.random.uniform(0, perturbation, x_float.shape)
    y_float += np.random.uniform(0, perturbation, y_float.shape)

    x, x_frac = np.divmod(x_float, 1)
    y, y_frac = np.divmod(y_float, 1)
    x, y = x.astype(int), y.astype(int)
    data[x, y]     = (1-x_frac) * (1-y_frac)
    data[x+1, y]   = (x_frac)   * (1-y_frac)
    data[x, y+1]   = (1-x_frac) * (y_frac)
    data[x+1, y+1] = y_frac     * x_frac

    if kernel is not None:
        # type: ignore
        data = convolve_fft(data, kernel)
    data = data/np.max(data) + 0.001  # normalize and add tiny offset to have no zeros in data

    table = Table((x_float.ravel(), y_float.ravel(), np.ones(x.size)), names=names)
    return data, table


def make_anisocado_kernel(shift=(0, 14), wavelength=2.15):
    """
    Get a convolvable Kernel from anisocado to approximate field constant MICADO PSF
    :param shift: how far away from center are we?
    :param wavelength: PSF for what wavelength?
    :return: Convolution kernel
    """
    count = pixel_count + 1 if pixel_count%2==0 else pixel_count
    hdus = anisocado.misc.make_simcado_psf_file(
        [shift], [wavelength], pixelSize=0.004, N=pixel_count)
    image = hdus[2]
    kernel = np.squeeze(image.data)
    return Kernel2D(array=kernel)


kernel_size = 201  # this should be enough
# name : generator Callable[[], Tuple[np.ndarray, Table]]
images = {
    'gauss_cluster_N1000': lambda: gaussian_cluster(N=1000),
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


def read_or_generate(filename: str, config=Config.instance()):
    """
    For the 'recipes' defined in the 'images' dictionary either generate and write the image/catalogue
    or read existing image/catalogue from disk
    :param filename: where to write/read the image from/to
    :param config: Configuration object
    :return: image, input_catalogue
    """
    try:
        generator = images[filename]
    except KeyError:
        print(f'No generator for {filename} defined')
        raise
    image_name = join(config.output_folder, filename + '.fits')
    table_name = join(config.output_folder, filename + '.dat')

    if exists(image_name) and exists(table_name):
        img = fits.open(image_name)[0].data
        table = Table.read(table_name, format='ascii.ecsv')
    else:
        img, table = generator()
        PrimaryHDU(img).writeto(image_name, overwrite=True)
        table.write(table_name, format='ascii.ecsv')

    return img, table
