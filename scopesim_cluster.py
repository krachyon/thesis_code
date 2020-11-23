import copy
import multiprocessing
import os
import tempfile
from typing import List, Tuple

import anisocado
import astropy.table
import matplotlib.pyplot as plt
import numpy as np
import pyckles
import scopesim
import scopesim_templates
from astropy.io.fits import PrimaryHDU
from astropy.table import Table, Row
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

plt.ion()

# globals
pixel_scale = 0.004  # TODO get this from scopesim?
psf_name = 'anisocado_psf'
N_simulation = 12
output_folder = 'output_files'


def get_spectral_types() -> List[Row]:
    pickles_lib = pyckles.SpectralLibrary('pickles', return_style='synphot')
    return list(pickles_lib.table['name'])


def make_scopesim_cluster(seed: int = 9999) -> scopesim.Source:
    return scopesim_templates.basic.stars.cluster(mass=1000,  # Msun
                                                  distance=50000,  # parsec
                                                  core_radius=0.3,  # parsec
                                                  seed=seed)


@np.vectorize
def to_pixel_scale(pos):
    """
    convert position of objects from arcseconds to pixel coordinates
    :param pos:
    :return:
    """
    return pos / pixel_scale + 512 + 1


def make_simcado_cluster(seed: int = 9999) -> scopesim.Source:
    """
    Emulates custom cluster creation from initial script
    :param seed:
    :return:
    """
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

    # That's what scopesim seemed to use for all stars.
    spectral_types = ['A0V'] * Nprime

    return scopesim_templates.basic.stars.stars(filter_name=filter_name,
                                                amplitudes=m,
                                                spec_types=spectral_types,
                                                x=x, y=y)


def download() -> None:
    """
    get scopesim file if not present
    :return:
    """
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
    """
    create a psf effect for scopesim to be as close as possible to how an anisocado PSF is used in simcado
    :param psf_wavelength:
    :param shift:
    :param N: ? Size of kernel?
    :return: effect object you can plug into OpticalTrain
    """
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


def do_photometry(image: np.ndarray, σ_psf: float) -> Tuple[Table, np.ndarray]:
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
    """
    Create a DS9 region file from a list of coordinates
    :param x_y_data: set of x-y coordinate pairs
    :param filename: where to write to
    :return:
    """
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
    """
    Create a Micado optical train with custom PSF
    :return: OpticalTrain object
    """
    psf_effect = make_psf()

    micado = scopesim.OpticalTrain('MICADO')

    # the previous psf had that optical element so put it in the same spot.
    # Todo This way of looking up the index is pretty stupid. Is there a better way?
    element_idx = [element.meta['name'] for element in micado.optics_manager.optical_elements].index('default_ro')

    micado.optics_manager.add_effect(psf_effect, ext=element_idx)

    # disable old psf
    # TODO - why is there no remove_effect with a similar interface? Why do I need to go through a dictionary attached to
    # TODO   a different class?
    # TODO - would be nice if Effect Objects where frozen, e.g. with the dataclass decorator. Used ".included" first and
    # TODO   was annoyed that it wasn't working...
    micado['relay_psf'].include = False
    micado['micado_ncpas_psf'].include = False

    return micado


# TODO: This does not work, there's a constant shift between the object and the image
def match_observation_to_source(astronomical_object: scopesim.Source, photometry_result: Table) \
        -> Table:
    """
    Find the nearest source for a fitted astrometry result and add this information to the photometry_result table
    :param astronomical_object: The source object that was observed with scopesim
    :param photometry_result: What the photometry found out about it
    :return: table modified with extra information
    """
    from scipy.spatial import cKDTree
    assert (len(astronomical_object.fields) == 1)  # Can't handle multiple fields

    x_y = np.array((astronomical_object.fields[0]['x'], astronomical_object.fields[0]['y'])).T
    x_y_pixel = to_pixel_scale(x_y)
    lookup_tree = cKDTree(x_y_pixel)

    photometry_result['x_orig'] = np.nan
    photometry_result['y_orig'] = np.nan
    photometry_result['offset'] = np.nan

    seen_indices = set()
    for row in photometry_result:

        dist, index = lookup_tree.query((row['x_fit'], row['y_fit']))
        if index in seen_indices:
            print('Warning: multiple match for source')  # TODO make this message more useful/use warning module
        seen_indices.add(index)
        row['x_orig'] = x_y_pixel[index, 0]
        row['y_orig'] = x_y_pixel[index, 1]
        row['offset'] = dist

    return photometry_result


def match_observations(observations: List[Table]):
    """
    given a set of astrometric measurements, take the first as reference and determine difference in centroid position
    by nearest neighbour search.
    :param observations: Multiple astrometric measurments of same object
    :return: None, modifies table inplace
    """
    from scipy.spatial import cKDTree  # O(n log n) instead of O(n**2)  lookup speed
    assert (len(observations) != 0)

    ref_obs = observations[0]
    x_y = np.array((ref_obs['x_fit'], ref_obs['y_fit'])).T
    lookup_tree = cKDTree(x_y)

    for photometry_result in observations:
        photometry_result['x_ref'] = np.nan
        photometry_result['y_ref'] = np.nan
        photometry_result['ref_index'] = -1
        photometry_result['offset'] = np.nan

        for row in photometry_result:
            dist, index = lookup_tree.query((row['x_fit'], row['y_fit']))
            row['x_ref'] = x_y[index, 0]
            row['y_ref'] = x_y[index, 1]
            row['ref_index'] = index
            row['offset'] = dist


def observation_and_photometry(astronomical_object: scopesim.Source, seed: int) \
        -> Tuple[np.ndarray, np.ndarray, Table, float]:
    """
    Observe an object with scopesim and perform photometry on resulting image. Deterministic wrt. to given random seed
    :param astronomical_object: what to observe
    :param seed: for making randomness repeatable
    :return: Tuple [observed image, residual image after photometry, photometry data, sigma of assumed psf]
    """

    np.random.seed(seed)

    detector = setup_optical_train()
    detector.observe(astronomical_object, random_seed=seed, update=True)
    observed_image = detector.readout()[0][1].data

    _, _, σ = fit_gaussian_to_psf(detector[psf_name].data)
    photometry_result, residual_image = do_photometry(observed_image, σ)

    # photometry_result = match_observation_to_source(astronomical_object, photometry_result)

    return observed_image, residual_image, photometry_result, σ


def plot_images(results: List[Tuple[np.ndarray, np.ndarray, Table, float]]) -> None:
    for i, (observed_image, residual_image, photometry_result, sigma) in enumerate(results):

        x_y_data = np.vstack((photometry_result['x_fit'], photometry_result['y_fit'])).T

        if verbose:
            print(f"Got σ={sigma} for the PSF")
            print(f'found {len(x_y_data)} out of {stars_in_cluster} stars with photometry')

        if output:
            PrimaryHDU(observed_image).writeto(f'{output_folder}/observed_{i:02d}.fits', overwrite=True)
            PrimaryHDU(residual_image).writeto(f'{output_folder}/residual_{i:02d}.fits', overwrite=True)

            write_ds9_regionfile(x_y_data, f'{output_folder}/centroids_{i:02d}.reg')

        # Visualization

        plt.figure(figsize=(10, 10))
        plt.imshow(observed_image, norm=LogNorm(), vmax=1E5)
        plt.scatter(x_y_data[:, 0], x_y_data[:, 1], marker="o", lw=1, color=None, edgecolors='red')
        plt.title('observed')
        plt.colorbar()
        if output:
            plt.savefig(f'{output_folder}/obsverved_{i:02d}.png')

        plt.figure(figsize=(10, 10))
        plt.imshow(residual_image, norm=LogNorm(), vmax=1E5)
        plt.title('residual')
        plt.colorbar()
        if output:
            plt.savefig(f'{output_folder}/residual{i:02d}.png')


def plot_deviation(photometry_results: List[Table]) -> None:
    match_observations(photometry_results)

    stacked = astropy.table.vstack(photometry_results)

    # only select stars that are seen in all observations
    stacked = stacked.group_by('ref_index').groups.filter(lambda tab, keys: len(tab) == len(photometry_results))

    # evil table magic to combine information for objects that where matched to single reference

    means = stacked.group_by('ref_index').groups.aggregate(np.mean)
    stds = stacked.group_by('ref_index').groups.aggregate(np.std)

    plt.close('all')

    plt.figure()
    plt.plot(-2.5 * np.log10(means['flux_fit']), stds['offset'], 'o')
    plt.title('magnitude vs std-deviation of position')
    plt.savefig(f'{output_folder}/magnitude_vs_std.png')

    plt.figure()
    plt.plot(-2.5 * np.log10(stacked['flux_fit']), stacked['offset'], '.')
    plt.title('spread of measurements')
    plt.xlabel('magnitude')
    plt.ylabel('measured deviation')
    plt.savefig(f'{output_folder}/magnitude_vs_spread.png')

    rms = np.sqrt(np.mean(stacked['offset'] ** 2))
    print(f'Total RMS of offset between values: {rms}')


verbose = True
output = True

download()

cluster = make_simcado_cluster()
stars_in_cluster = len(cluster.meta['x'])  # TODO again, stupid way of looking this information up...

micado = setup_optical_train()  # this can't be pickled and not be used in multiprocessing, so create independently
if verbose:
    micado.effects.pprint(max_lines=100, max_width=300)
psf_effect = micado[psf_name]

# Weird lockup when cluster is shared between processes...
args = [(copy.deepcopy(cluster), i) for i in range(N_simulation)]

with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    results = pool.starmap(observation_and_photometry, args)

# debuggable version
# import itertools
# results = list(itertools.starmap(observation_and_photometry, args))

if output:
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    PrimaryHDU(psf_effect.data).writeto(f'{output_folder}/psf.fits', overwrite=True)

    fit_gaussian_to_psf(psf_effect.data, plot=True)
    plt.savefig(f'{output_folder}/psf_fit.png')

    plt.figure()
    plt.imshow(psf_effect.data, norm=LogNorm(), vmax=1E5)
    plt.colorbar()
    plt.title('PSF')
    plt.savefig(f'{output_folder}/psf.png')

    plot_images(results)

photometry_results = [photometry_result for observed_image, residual_image, photometry_result, sigma in results]

if output:
    plot_deviation(photometry_results)

script_has_run = True
