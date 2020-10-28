import scopesim
import scopesim_templates
import numpy as np
import pyckles
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# globals
pixel_scale = 0.004  # TODO get this from scopesim?


def get_spectral_types():
    pickles_lib = pyckles.SpectralLibrary("pickles", return_style="synphot")
    return list(pickles_lib.table['name'])


def make_scopesim_cluster(seed=9999):
    return scopesim_templates.basic.stars.cluster(mass=1000,  # Msun
                                                  distance=50000,  # parsec
                                                  core_radius=0.3,  # parsec
                                                  seed=seed)


def make_simcado_cluster(seed=9999):
    np.random.seed(seed)
    N = 1000
    x = np.random.normal(0, 1, N)
    y = np.random.normal(0, 1, N)
    m = np.random.normal(19, 2, N)
    x_in_px = x / pixel_scale + 512 + 1
    y_in_px = y / pixel_scale + 512 + 1

    mask = (m > 14) * (0 < x_in_px) * (x_in_px < 1024) * (0 < y_in_px) * (y_in_px < 1024)
    x = x[mask];
    y = y[mask];
    m = m[mask]

    assert (len(x) == len(y) and len(m) == len(x))
    Nprime = len(x)
    filter = "MICADO/filters/TC_filter_K-cont.dat"  # TODO: how to make system find this?
    # TODO: random spectral types, not sure what original does
    spectral_types = np.random.choice(get_spectral_types(), Nprime)

    return scopesim_templates.basic.stars.stars(filter_name=filter, amplitudes=m, spec_types=spectral_types, x=x, y=y)


def download():
    # Warning: This will download everything to your working directory, so run this in dedicated workspace dira
    scopesim.download_package(["locations/Armazones",
                               "telescopes/ELT",
                               "instruments/MICADO"])


def main():
    ## uncomment if directory not populated:
    # download()

    micado = scopesim.OpticalTrain("MICADO")
    cluster = make_simcado_cluster()
    micado.observe(cluster)
    hdus = micado.readout()[0]

    plt.figure(figsize=(10, 10))
    plt.imshow(hdus[1].data, norm=LogNorm(), vmax=1E5)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
