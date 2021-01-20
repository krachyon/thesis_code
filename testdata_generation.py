import scopesim_templates
import scopesim
from scopesim_cluster import setup_optical_train
from astropy.io.fits import PrimaryHDU
from astropy.convolution import AiryDisk2DKernel, Gaussian2DKernel, Kernel2D, convolve_fft

import numpy as np


def make_grid_image(N1d, seed: int = 1000) -> None:
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


def make_convolved_grid(N1d, border=64, kernel: Kernel2D = Gaussian2DKernel(x_stddev=1),
                        perturbation: float = 0.) -> np.ndarray:
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


    data = convolve_fft(data, kernel)
    return data/np.max(data) + 0.01


#make_grid_image(8, 0)
#make_grid_image(16, 0)
#make_grid_image(15, 0)

size = 1024
np.random.seed(1000)

gauss_15 = make_convolved_grid(15, kernel=Gaussian2DKernel(x_stddev=2, x_size=size, y_size=size))
PrimaryHDU(gauss_15).writeto(f'output_files/grid_gauss_15.fits', overwrite=True)
gauss_16 = make_convolved_grid(16, kernel=Gaussian2DKernel(x_stddev=2, x_size=size, y_size=size))
PrimaryHDU(gauss_16).writeto(f'output_files/grid_gauss_16.fits', overwrite=True)

airy_15 = make_convolved_grid(15, kernel=AiryDisk2DKernel(radius=2, x_size=size, y_size=size))
PrimaryHDU(airy_15).writeto(f'output_files/grid_airy_15.fits', overwrite=True)
airy_16 = make_convolved_grid(16, kernel=AiryDisk2DKernel(radius=2, x_size=size, y_size=size))
PrimaryHDU(airy_16).writeto(f'output_files/grid_airy_16.fits', overwrite=True)

airy_16_perturbed = \
    make_convolved_grid(16, kernel=AiryDisk2DKernel(radius=2, x_size=size, y_size=size), perturbation=4.)
PrimaryHDU(airy_16_perturbed).writeto(f'output_files/grid_airy_16_perturbed.fits', overwrite=True)
