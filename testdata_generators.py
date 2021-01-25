import numpy as np

import scopesim_templates

from astropy.io import fits
from astropy.io.fits import PrimaryHDU
from astropy.table import Table

from scopesim_helper import setup_optical_train, pixel_scale, pixel_count, filter_name
from astropy.convolution import AiryDisk2DKernel, Gaussian2DKernel, Kernel2D, convolve_fft

from os.path import exists, join

from config import Config
from typing import Callable, Tuple

from scopesim_helper import to_pixel_scale

# all generators defined here should return a source table with the following columns
# x,y are in pixel scale
# TODO maybe enforce with astropy.units
names = ('x', 'y', 'm')

def scopesim_grid(N1d: int = 16, seed: int = 1000, border=64, perturbation: float = 0.) \
        -> Tuple[np.ndarray, Table]:
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

    table = Table((to_pixel_scale(x), to_pixel_scale(y), m), names=names)
    return observed_image, table


def gaussian_cluster(N: int = 1000, seed: int = 9999) \
        -> Tuple[np.ndarray, Table]:
    """
    Emulates custom cluster creation from initial script
    :param seed:
    :return:
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

    table = Table((to_pixel_scale(x), to_pixel_scale(y), m), names=names)
    return observed_image, table


def scopesim_cluster(seed: int = 9999) -> Tuple[np.ndarray, Table]:
    source = scopesim_templates.basic.stars.cluster(mass=1000,  # Msun
                                                    distance=50000,  # parsec
                                                    core_radius=0.3,  # parsec
                                                    seed=seed)
    detector = setup_optical_train()

    detector.observe(source, random_seed=seed, update=True)
    observed_image = detector.readout()[0][1].data

    table = source.fields
    table['x'] = to_pixel_scale(table['x'])
    table['y'] = to_pixel_scale(table['y'])


    return observed_image, table


def convolved_grid(N1d: int = 16,
                   border: int = 64,
                   kernel: Kernel2D = Gaussian2DKernel(x_stddev=1),
                   perturbation: float = 0.,
                   seed: int = 1000) -> Tuple[np.ndarray, Table]:

    np.random.seed(seed)

    size = 1024
    data = np.zeros((size,  size))

    idx_float = np.linspace(0+border, size-border, N1d)
    x_float = np.tile(idx_float, reps=(N1d,1))
    y_float = x_float.T
    x_float += np.random.uniform(0,perturbation, x_float.shape)
    y_float += np.random.uniform(0, perturbation, y_float.shape)

    x, x_frac = np.divmod(x_float, 1)
    y, y_frac = np.divmod(y_float, 1)
    x, y = x.astype(int), y.astype(int)
    data[x, y]     = (1-x_frac) * (1-y_frac)
    data[x+1, y]   = (x_frac)   * (1-y_frac)
    data[x, y+1]   = (1-x_frac) * (y_frac)
    data[x+1, y+1] = y_frac     * x_frac

    # type: ignore
    data = convolve_fft(data, kernel)
    data = data/np.max(data) + 0.001  # normalize and add tiny offset to have no zeros in data

    table = Table((x, y, np.ones(len(x))), names=names)
    return data, table


# name : generator Callable[[], Tuple[np.ndarray, Table]]
images = {
    'gauss_cluster_N1000': lambda: gaussian_cluster(N=1000),
    'scopesim_cluster': lambda: scopesim_cluster()
          }


def read_or_generate(filename: str,):
    try:
        generator = images[filename]
    except KeyError:
        print(f'No generator for {filename} defined')
        raise
    image_name = join(Config.output_folder, filename + '.fits')
    table_name = join(Config.output_folder, filename + '.dat')

    if exists(image_name) and exists(table_name):
        img = fits.open(image_name)[0].data
        table = Table.read(table_name, format='ascii')
    else:
        img, table = generator()
        PrimaryHDU(img).writeto(image_name, overwrite=True)
        table.write(table_name, format='ascii')

    return img, table
