import scopesim_templates
import scopesim
from scopesim_cluster import setup_optical_train
from astropy.io.fits import PrimaryHDU

import numpy as np


def make_grid_image(N1d, seed: int = 1000) -> np.ndarray:
    np.random.seed(seed)

    N = N1d**2
    spectral_types = ['A0V'] * N
    filter_name = 'MICADO/filters/TC_filter_K-cont.dat'  # TODO: how to make system find this?
    pixel_scale = 0.004
    pixel_count = 1024

    start = N1d
    stop = pixel_count
    step = pixel_count//N1d - 1

    y = (np.tile(np.linspace(32, pixel_count-32, N1d), reps=(N1d, 1)) - pixel_count/2) * pixel_scale
    x = y.T

    m = np.array(N*[18])

    source = scopesim_templates.basic.stars.stars(filter_name=filter_name,
                                                amplitudes=m,
                                                spec_types=spectral_types,
                                                x=x.ravel(), y=y.ravel())
    detector = setup_optical_train()

    detector.observe(source, random_seed=seed, update=True)
    observed_image = detector.readout()[0][1].data

    PrimaryHDU(observed_image).writeto(f'output_files/grid_{N1d:d}.fits', overwrite=True)


#make_grid_image(8, 0)
make_grid_image(16, 0)
make_grid_image(15, 0)

