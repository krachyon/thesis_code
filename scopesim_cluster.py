import os
from typing import List, Tuple

import anisocado
import astropy.table
from astropy.io.fits import PrimaryHDU
import matplotlib.pyplot as plt
import numpy as np
import pyckles
import scopesim
import scopesim_templates
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
import multiprocessing
import tempfile


plt.ion()


# globals
pixel_scale = 0.004  # TODO get this from scopesim?
psf_name = 'anisocado_psf'
N_simulation = 3


def get_spectral_types() -> List[astropy.table.Row]:
    pickles_lib = pyckles.SpectralLibrary('pickles', return_style='synphot')
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
    filter_name = 'MICADO/filters/TC_filter_K-cont.dat'  # TODO: how to make system find this?
    ## TODO: random spectral types, adapt this to a realistic cluster distribution or maybe just use
    ## scopesim_templates.basic.stars.cluster
    # random_spectral_types = np.random.choice(get_spectral_types(), Nprime)

    # That's what scopesim seemed to use for all stars.ü
    spectral_types = ['A0V'] * Nprime

    return scopesim_templates.basic.stars.stars(filter_name=filter_name,
                                                amplitudes=m,
                                                spec_types=spectral_types,
                                                x=x, y=y)


def download() -> None:
    if not os.path.exists('MICADO'):
        # TODO is it really necessary to always throw shit into the current wdir?
        print('''Simcado Data missing. Do you want to download?
        Attention: Will write into current working dir!''')
        choice = input('[y/N] ')
        if choice == 'y' or choice == 'Y':
            scopesim.download_package(["locations/Armazones",
                                       "telescopes/ELT",
                                       "instruments/MICADO"])
        else:
            exit(-1)


# noinspection PyPep8Naming
def make_psf(psf_wavelength: float = 2.15, shift: Tuple[int] = (0, 14), N: int = 512) -> scopesim.effects.Effect:
    hdus = anisocado.misc.make_simcado_psf_file(
        [shift], [psf_wavelength], pixelSize=pixel_scale, N=N)
    image = hdus[2]
    image.data = np.squeeze(image.data)  # remove leading dimension, we're only looking at a single picture, not a stack
    filename = tempfile.NamedTemporaryFile('w', suffix='.fits').name
    image.writeto(filename)


    # noinspection PyTypeChecker
    tmp_psf = anisocado.AnalyticalScaoPsf(N=N, wavelength=psf_wavelength)
    strehl = tmp_psf.strehl_ratio

    # Todo: passing a filename that does not end in .fits causes a weird parsing error
    return scopesim.effects.FieldConstantPSF(
        name=psf_name,
        filename=filename,
        wavelength=psf_wavelength,
        psf_side_length=N,
        strehl_ratio=strehl, )
    # convolve_mode=''


def gauss(x, a, x0, σ):
    return a * np.exp(-(x - x0) ** 2 / (2 * σ ** 2))


def fit_gaussian_to_psf(psf: np.ndarray, plot=False) -> np.ndarray:
    """Take a (centered) psf image and attempt to fit a gaussian through
    :return: tuple of a, x0, σ
    """
    # TODO does it really make sense to fit only a 1-d slice, not the whole image?

    assert (psf.ndim == 2)
    assert (psf.shape[0] == psf.shape[1])

    size = psf.shape[0]
    # get horizontal rows of pixels through center
    psf_slice_h = psf[size // 2, :]
    # psf_slice_w = psf[:, size//2]

    x_vals = np.arange(-size // 2, size // 2, 1)

    popt, pcov = curve_fit(gauss, x_vals, psf_slice_h)

    if plot:
        plt.figure()
        plt.plot(x_vals, psf_slice_h, 'o')
        x_dense = np.linspace(-size // 2, size // 2, size * 10)
        plt.plot(x_dense, gauss(x_dense, *popt))

    return popt


def do_photometry(image: np.ndarray, σ_psf: float) -> Tuple[astropy.table.Table, np.ndarray]:
    """
    Find stars in an image

    :param image: The image data you want to find stars in
    :param σ_psf: expected deviation of PSF
    :return: tuple result table, residual image
    """
    from photutils.detection import IRAFStarFinder
    from photutils.psf import DAOGroup
    from photutils.psf import IntegratedGaussianPRF
    from photutils.background import MMMBackground
    from photutils.background import MADStdBackgroundRMS
    from photutils.psf import BasicPSFPhotometry

    from astropy.modeling.fitting import LevMarLSQFitter
    from astropy.stats import gaussian_sigma_to_fwhm

    bkgrms = MADStdBackgroundRMS()

    std = bkgrms(image)

    iraffind = IRAFStarFinder(threshold=3 * std, sigma_radius=σ_psf,
                              fwhm=σ_psf * gaussian_sigma_to_fwhm,
                              minsep_fwhm=2, roundhi=5.0, roundlo=-5.0,
                              sharplo=0.0, sharphi=2.0)
    daogroup = DAOGroup(0.1 * σ_psf * gaussian_sigma_to_fwhm)

    mmm_bkg = MMMBackground()

    # my_psf = AiryDisk2D(x_0=0., y_0=0.,radius=airy_minimum)
    # psf_model = prepare_psf_model(my_psf, xname='x_0', yname='y_0', fluxname='amplitude',renormalize_psf=False)
    psf_model = IntegratedGaussianPRF(sigma=σ_psf)
    # psf_model = AiryDisk2D(radius = airy_minimum)#prepare_psf_model(AiryDisk2D,xname ="x_0",yname="y_0")
    # psf_model = Moffat2D([amplitude, x_0, y_0, gamma, alpha])

    # photometry = IterativelySubtractedPSFPhotometry(finder=iraffind, group_maker=daogroup,
    #                                                bkg_estimator=mmm_bkg, psf_model=psf_model,
    #                                                fitter=LevMarLSQFitter(),
    #                                                niters=2, fitshape=(11,11))
    photometry = BasicPSFPhotometry(finder=iraffind, group_maker=daogroup,
                                    bkg_estimator=mmm_bkg, psf_model=psf_model,
                                    fitter=LevMarLSQFitter(), aperture_radius=11.0,
                                    fitshape=(11, 11))

    result_table = photometry.do_photometry(image)
    return result_table, photometry.get_residual_image()


def write_ds9_regionfile(x_y_data: np.ndarray, filename: str) -> None:
    assert (x_y_data.ndim == 2)
    assert (x_y_data.shape[1] == 2)

    with open(filename, 'w') as f:
        f.write("# Region file format: DS9 version 3.0\n")
        f.write(
            "global color=blue font=\"helvetica 10 normal\" select=1 edit=1 move=1 delete=1 include=1 fixed=0 source\n")
        for row in x_y_data:
            # +1 for ds9 one-based indexing...
            f.write(f"image;circle( {row[0] + 1:f}, {row[1] + 1:f}, 1.5)\n")
        f.close()


def setup_optical_train() -> scopesim.OpticalTrain:
    psf_effect = make_psf()

    micado = scopesim.OpticalTrain('MICADO')

    # the previous psf had that optical element so put it in the same spot.
    # Todo This way of looking up the index is pretty stupid. Is there a better way?
    element_idx = [element.meta['name'] for element in micado.optics_manager.optical_elements].index('default_ro')

    micado.optics_manager.add_effect(psf_effect, ext=element_idx)

    # disable old psf
    # TODO - why is there no remove_effect with a similar interface? Why do I need to go through a dictionary attached to
    # TODO   a different class?
    # TODO - would be nice if Effect Objects where frozen, e.g. with the dataclas decorator. Used ".included" first and
    # TODO   was annoyed that it wasn't working...
    micado['relay_psf'].include = False
    micado['micado_ncpas_psf'].include = False

    return micado


def observation_and_photometry(astronomical_object: scopesim.Source, seed: int) \
        -> Tuple[np.ndarray, np.ndarray, astropy.table.Table]:

    np.random.seed(seed)

    detector = setup_optical_train()
    detector.observe(astronomical_object, random_seed=seed, update=True)
    observed_image = detector.readout()[0][1].data

    _, _, σ = fit_gaussian_to_psf(detector[psf_name].data)
    photometry_result, residual_image = do_photometry(observed_image, σ)
    photometry_result.sigma_input = σ

    return observed_image, residual_image, photometry_result


# TODO: return types
# TODO: difference between two observations
# TODO: SNR vs Sigma plot

def main(verbose=True, output=True):
    download()

    cluster = make_simcado_cluster()
    stars_in_cluster = len(cluster.meta['x'])  # TODO again, stupid way of looking this information up...

    micado = setup_optical_train()  # this can't be pickled and not be used in multiprocessing, so create independently
    if verbose:
        micado.effects.pprint(max_lines=100, max_width=300)
    psf_effect = micado[psf_name]

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    args = [(cluster, i) for i in range(N_simulation)]

    results = pool.starmap(observation_and_photometry, args)
    # debuggable version
    # import itertools
    # results = list(itertools.starmap(observation_and_photometry, args))

    if output:
        if not os.path.exists('output_files'):
            os.mkdir('output_files')
        PrimaryHDU(psf_effect.data).writeto('output_files/psf.fits', overwrite=True)

    fit_gaussian_to_psf(psf_effect.data, plot=True)
    plt.savefig(f'output_files/psf_fit.png')

    plt.figure()
    plt.imshow(psf_effect.data, norm=LogNorm(), vmax=1E5)
    plt.colorbar()
    plt.title('PSF')
    plt.savefig(f'output_files/psf.png')

    for i, (observed_image, residual_image, photometry_result) in enumerate(results):

        x_y_data = np.vstack((photometry_result['x_fit'], photometry_result['y_fit'])).T

        if verbose:
            print(f"Got σ={photometry_result.sigma_input} for the PSF")
            print(f'found {len(x_y_data)} out of {stars_in_cluster} stars with photometry')

        if output:
            PrimaryHDU(observed_image).writeto(f'output_files/observed_{i:02d}.fits', overwrite=True)
            PrimaryHDU(residual_image).writeto(f'output_files/residual_{i:02d}.fits', overwrite=True)

            write_ds9_regionfile(x_y_data, f'output_files/centroids_{i:02d}.reg')

        # Visualization

        plt.figure(figsize=(10, 10))
        plt.imshow(observed_image, norm=LogNorm(), vmax=1E5)
        plt.scatter(x_y_data[:, 0], x_y_data[:, 1], marker="o", lw=1, color=None, edgecolors='red')
        plt.title('observed')
        plt.colorbar()
        plt.savefig(f'output_files/obsverved_{i:02d}.png')

        plt.figure(figsize=(10, 10))
        plt.imshow(residual_image, norm=LogNorm(), vmax=1E5)
        plt.title('residual')
        plt.colorbar()
        plt.savefig(f'output_files/residual{i:02d}.png')




if __name__ == '__main__':
    main()
    plt.show()
