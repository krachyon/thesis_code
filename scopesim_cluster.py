import scopesim
import scopesim_templates
import numpy as np
import pyckles
import os
import astropy.table
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import anisocado
import tempfile

# globals
pixel_scale = 0.004  # TODO get this from scopesim?


def get_spectral_types() -> List[astropy.table.Row]:
    pickles_lib = pyckles.SpectralLibrary("pickles", return_style="synphot")
    return list(pickles_lib.table['name'])


def make_scopesim_cluster(seed: int = 9999) -> scopesim.Source:
    return scopesim_templates.basic.stars.cluster(mass=1000,  # Msun
                                                  distance=50000,  # parsec
                                                  core_radius=0.3,  # parsec
                                                  seed=seed)


def make_simcado_cluster(seed: int = 9999) -> scopesim.Source:
    np.random.seed(seed)
    N = 1000
    x = np.random.normal(0, 1, N)
    y = np.random.normal(0, 1, N)
    m = np.random.normal(19, 2, N)
    x_in_px = x / pixel_scale + 512 + 1
    y_in_px = y / pixel_scale + 512 + 1

    mask = (m > 14) * (0 < x_in_px) * (x_in_px < 1024) * (0 < y_in_px) * (y_in_px < 1024)
    x = x[mask]
    y = y[mask]
    m = m[mask]

    assert (len(x) == len(y) and len(m) == len(x))
    Nprime = len(x)
    filter = "MICADO/filters/TC_filter_K-cont.dat"  # TODO: how to make system find this?
    ## TODO: random spectral types, adapt this to a realistic cluster distribution or maybe just use
    ## scopesim_templates.basic.stars.cluster
    # random_spectral_types = np.random.choice(get_spectral_types(), Nprime)
    spectral_types = ["A0V"] * Nprime

    return scopesim_templates.basic.stars.stars(filter_name=filter, amplitudes=m, spec_types=spectral_types, x=x, y=y)


def download() -> None:
    # Warning: This will download everything to your working directory, so run this in dedicated workspace dira
    scopesim.download_package(["locations/Armazones",
                               "telescopes/ELT",
                               "instruments/MICADO"])


def generate_psf(psf_wavelength: float = 2.15, shift: Tuple[int] = (0, 14), N: int = 512):
    hdus = anisocado.misc.make_simcado_psf_file(
        [shift], [psf_wavelength], pixelSize=pixel_scale, N=N)
    image = hdus[2]
    image.data = np.squeeze(image.data)  # remove leading dimension, we're only looking at a single picture, not a stack
    image.writeto('temp.fits', overwrite=True)

    tmp_psf = anisocado.AnalyticalScaoPsf(N=N, wavelength=psf_wavelength)
    strehl = tmp_psf.strehl_ratio

    return scopesim.effects.FieldConstantPSF(
                                            name="anisocado_psf",
                                            filename='temp.fits',
                                            wavelength=psf_wavelength,
                                            psf_side_length=N,
                                            strehl_ratio=strehl,)
    # convolve_mode=''

def main():
    ## uncomment if directory not populated:
    # download()
    psf_effect = generate_psf()
    cluster = make_simcado_cluster()

    micado = scopesim.OpticalTrain("MICADO")

    # the previous psf had that optical element so put it in the same spot.
    # Todo This way of looking it up is pretty stupid. Is there a better way?
    element_idx = [element.meta['name'] for element in micado.optics_manager.optical_elements].index('default_ro')

    micado.optics_manager.add_effect(psf_effect, ext=element_idx)

    # disable old psf
    # TODO why is there no remove_effect with a similar interface? Why do I need to go through a dictionary attached to
    # TODO would be nice if Effect Objects where frozen, e.g. with the dataclas decorator. Used ".included" first and
    # was annoyed that it wasn't working...
    micado['relay_psf'].include = False

    micado.effects.pprint(max_lines=100, max_width=300)

    micado.observe(cluster)
    hdus = micado.readout()[0]

    plt.figure(figsize=(10, 10))
    plt.imshow(hdus[1].data, norm=LogNorm(), vmax=1E5)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
