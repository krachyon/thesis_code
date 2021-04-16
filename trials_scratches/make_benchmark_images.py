# can be installed from github.com/krachyon/thesis_code
from thesis_lib.testdata_generators import read_or_generate_image, scopesim_grid, gaussian_cluster, lowpass
from thesis_lib.scopesim_helper import download

import tempfile
import os
import numpy as np

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
}

if __name__ == '__main__':
    # download scopesim stuff to temporary location
    old_dir = os.path.abspath(os.getcwd())
    with tempfile.TemporaryDirectory() as dir:
        os.chdir(dir)
        download(ask=False)

        for fname, recipe in benchmark_images.items():
            read_or_generate_image(recipe, fname, os.path.join(old_dir, 'test_images'))


def diy():
    """
    Examples of how to generate images yourself
    :return:
    """

    # generate Image for immediate use with python, won't use existing file:
    # see also available recipes in testdata_generators
    image, input_table = scopesim_grid()  # choose args as needed

    # generate different versions of same image for statistics
    for i in range (10):
        recipe = lambda: scopesim_grid(N1d=16, perturbation=2., magnitude=lambda N: np.random.uniform(18, 24, N))
        fname = f'scopesim_grid_16_perturb2_mag18_24_{i}'
        image, input_table = read_or_generate_image(recipe, fname, 'test_images')
