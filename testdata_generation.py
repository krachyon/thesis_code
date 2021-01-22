import scopesim_templates
import scopesim
from scopesim_cluster import setup_optical_train
from astropy.io.fits import PrimaryHDU

import numpy as np

border = 64





size = 1024
np.random.seed(1000)

def generate():
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

    airy_15_large = make_convolved_grid(15, kernel=AiryDisk2DKernel(radius=10, x_size=size, y_size=size))
    PrimaryHDU(airy_15_large).writeto(f'output_files/grid_airy_15_large.fits', overwrite=True)

    import anisocado
    hdus = anisocado.misc.make_simcado_psf_file(
        [(0, 14)], [2.15], pixelSize=0.004, N=size)
    image = hdus[2]
    kernel = np.squeeze(image.data)
    anisocado_15 = make_convolved_grid(15, kernel=kernel, perturbation=0)
    PrimaryHDU(anisocado_15).writeto(f'output_files/grid_anisocado_15.fits', overwrite=True)
    anisocado_16 = make_convolved_grid(16, kernel=kernel, perturbation=1.2)
    PrimaryHDU(anisocado_16).writeto(f'output_files/grid_anisocado_16.fits', overwrite=True)


if __name__ == '__main__':
    # generate()
    # make_grid_image(8, 0)
    # make_grid_image(16, 0)
    # make_grid_image(15, 0)
    # make_grid_image(16, 0, perturbation=1.2)
    make_grid_image(25, 0, perturbation=5.)

