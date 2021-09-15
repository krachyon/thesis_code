from os import mkdir
from os.path import exists, join
from typing import Callable, Tuple, Optional

import multiprocess
import numpy as np
from astropy.convolution import AiryDisk2DKernel, Gaussian2DKernel
from astropy.io.fits import PrimaryHDU
from astropy.table import Table

from .config import Config
from .scopesim_helper import make_anisocado_model
from .testdata_helpers import make_anisocado_kernel, lowpass, expmag
from .testdata_recipes import gaussian_cluster, convolved_grid, scopesim_cluster, scopesim_grid, scopesim_groups, \
    empty_image, single_star_image, test_dummy_img
from .util import getdata_safer, work_in

# make sure concurrent calls with the same filename don't tread on each other's toes.
# only generate/write file once

manager = multiprocess.Manager()
file_locks = manager.dict()
#file_locks = dict()

def get_lock(filename_base):
    lock: multiprocess.Lock = file_locks.setdefault(filename_base, manager.Lock())
    return lock


def read_or_generate_image(filename_base: str,
                           config: Config = Config.instance(),
                           recipe: Optional[Callable[[], Tuple[np.ndarray, Table]]] = None):
    """
    For the 'recipe' either generate and write the image+catalogue or read existing output from disk
    :param directory: where to put/read data
    :param filename_base: what the files are called, minus extension
    :param recipe: function generating your image and catalogue
    :return: image, input_catalogue
    """

    if not exists(config.image_folder):
        mkdir(config.image_folder)
    image_name = join(config.image_folder, filename_base + '.fits')
    table_name = join(config.image_folder, filename_base + '.dat')
    lock = get_lock(filename_base)
    with lock:
        if exists(image_name) and exists(table_name):
            img = getdata_safer(image_name)
            table = Table.read(table_name, format='ascii.ecsv')
        else:
            with work_in(config.scopesim_working_dir):
                if not recipe:
                    recipe = predefined_images[filename_base]
                img, table = recipe()
            img = img.astype(np.float64, order='C', copy=False)
            PrimaryHDU(img).writeto(image_name, overwrite=True)
            table.write(table_name, format='ascii.ecsv')

    return img, table


def read_or_generate_helper(filename_base: str,
                            config: Config = Config.instance(),
                            recipe: Optional[Callable[[], np.ndarray]] = None):
    """
    For the 'recipe' either generate and write the image+catalogue or read existing output from disk
    :param directory: where to put/read data
    :param filename_base: what the files are called, minus extension
    :param recipe: function generating your image and catalogue
    :return: image, input_catalogue
    """
    if not exists(config.image_folder):
        mkdir(config.image_folder)
    image_name = join(config.image_folder, filename_base + '.fits')
    lock = get_lock(filename_base)
    with lock:
        if exists(image_name):
            img = getdata_safer(image_name)
        else:
            with work_in(config.scopesim_working_dir):
                img = recipe().astype(np.float64, order='C', copy=False)
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
    'empty_image':
        lambda: empty_image(),
    'testdummy':
        lambda: test_dummy_img()
}

lowpass_images = {
    'scopesim_grid_30_perturb2_lowpass_mag18-24':
        lambda: scopesim_grid(N1d=30, perturbation=2., psf_transform=lowpass(),
                              magnitude=lambda N: np.random.uniform(18, 24, N)),
    'scopesim_groups_16_perturb_2_lowpass_radius_7':
        lambda: scopesim_groups(N1d=16, jitter=2., psf_transform=lowpass(), magnitude=lambda N: N * [20],
                                group_radius=7, group_size=2),
    'scopesim_groups_16_perturb_2_lowpass_radius_5':
        lambda: scopesim_groups(N1d=16, jitter=2., psf_transform=lowpass(), magnitude=lambda N: N * [20],
                                group_radius=10, group_size=5),
    'gausscluster_N2000_lowpass_mag22':
        lambda: gaussian_cluster(2000, magnitude=lambda N: np.random.normal(22, 2, N), psf_transform=lowpass()),
    'gausscluster_N4000_lowpass_expmag21':
        lambda: gaussian_cluster(4000, magnitude=expmag, psf_transform=lowpass())
}

benchmark_images = {
    'scopesim_grid_16_perturb2_mag18_24':
        lambda: scopesim_grid(N1d=16, perturbation=2.,
                              magnitude=lambda N: np.random.uniform(18, 24, N)),
    'scopesim_grid_16_perturb2_lowpass_mag18_24':
        lambda: scopesim_grid(N1d=16, perturbation=2.,
                              magnitude=lambda N: np.random.uniform(18, 24, N), psf_transform=lowpass()),
    'gausscluster_N2000_mag22':
        lambda: gaussian_cluster(2000, magnitude=lambda N: np.random.normal(22, 2, N)),
    'gausscluster_N2000_mag22_lowpass':
        lambda: gaussian_cluster(2000, magnitude=lambda N: np.random.normal(22, 2, N), psf_transform=lowpass()),
    'gausscluster_N2000_mag22_subpixel':
        lambda: gaussian_cluster(2000, magnitude=lambda N: np.random.normal(22, 2, N),
                                 custom_subpixel_psf=make_anisocado_model()),
    'gausscluster_N2000_mag22_lowpass_subpixel':
        lambda: gaussian_cluster(2000, magnitude=lambda N: np.random.normal(22, 2, N),
                                 custom_subpixel_psf=make_anisocado_model(lowpass=5)),
}

helpers = {
    'anisocado_psf':
        lambda: make_anisocado_kernel().array,
    'single_star_image':
        lambda: single_star_image()
}

predefined_images = benchmark_images | misc_images | lowpass_images
