import itertools

from photometry import make_epsf_fit
from testdata_generators import read_or_generate_image, gaussian_cluster, read_or_generate_helper
import testdata_generators

from config import Config

from photometry import make_stars_guess, make_epsf_fit, do_photometry_epsf, FWHM_estimate, cut_edges
# show how epsf looks for given image/parameters

from plots_and_sanitycheck import plot_image_with_source_and_measured, plot_input_vs_photometry_positions, \
    save, concat_star_images

from scopesim_helper import download
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import util
from collections import namedtuple

from astropy.stats import sigma_clipped_stats
from photutils.detection import IRAFStarFinder, DAOStarFinder
from photutils.background import MADStdBackgroundRMS
import photutils
import numpy as np
import astropy

from itertools import starmap

from astropy.modeling.fitting import LevMarLSQFitter
from astropy.io.fits import PrimaryHDU

from photutils.psf import DAOGroup, BasicPSFPhotometry

PhotometryResult = namedtuple('PhotometryResult',
                              ('image', 'input_table', 'result_table', 'epsf', 'star_guesses'))


# would be nicer to already pass the image here instead of the name but that would mean that generation
# happens in the main process
def photometry_full(filename='gauss_cluster_N1000', config=Config.instance()):
    """
    apply EPSF fitting photometry to a testimage

    :param filename: must be found in testdata_generators.images
    :param config: instance of Config containing all processing parameters
    :return: PhotometryResult, (image, input_table, result_table, epsf, star_guesses)
    """
    image, input_table = read_or_generate_image(filename, config)

    mean, median, std = sigma_clipped_stats(image, sigma=config.clip_sigma)
    threshold = median + config.threshold_factor * std

    finder = DAOStarFinder(threshold=threshold, fwhm=config.fwhm_guess)

    star_guesses = make_stars_guess(image,
                                    finder,
                                    cutout_size=config.cutout_size)

    epsf = make_epsf_fit(star_guesses,
                         iters=config.epsfbuilder_iters,
                         oversampling=config.oversampling,
                         smoothing_kernel=config.smoothing,
                         epsf_guess=config.epsf_guess)

    if config.use_catalogue_positions:
        guess_table = input_table.copy()
        guess_table = cut_edges(guess_table, config.cutout_size, image.shape[0])
        guess_table.rename_columns(['x', 'y'], ['x_0', 'y_0'])
    else:
        guess_table = None

    result_table = do_photometry_epsf(image, epsf, finder, initial_guess=guess_table, config=config)



    plot_filename = os.path.join(config.output_folder, filename+'_photometry_vs_sources')
    plot_image_with_source_and_measured(image, input_table, result_table, output_path=plot_filename)

    if len(result_table) != 0:
        plot_filename = os.path.join(config.output_folder, filename + '_measurement_offset')
        plot_input_vs_photometry_positions(input_table, result_table, output_path=plot_filename)
    else:
        print(f"No sources found for {filename} with {config}")

    plt.figure()
    plt.imshow(epsf.data)
    save(os.path.join(config.output_folder, filename+'_epsf'), plt.gcf())

    plt.figure()
    plt.imshow(concat_star_images(star_guesses))
    save(os.path.join(config.output_folder, filename+'_star_guesses'), plt.gcf())

    return PhotometryResult(image, input_table, result_table, epsf, star_guesses)


def cheating_astrometry(filename: str, psf: np.ndarray, config: Config):
    """
    Evaluate the maximum achievable precision of the EPSF fitting approach by using a hand-defined psf
    :param filename:
    :param psf:
    :param config:
    :return:
    """
    origin = np.array(psf.shape)/2
    # type: ignore
    epsf = photutils.psf.EPSFModel(psf, flux=1, origin=origin, oversampling=1, normalize=False)
    epsf = photutils.psf.prepare_psf_model(epsf, renormalize_psf=False)

    image, input_table = read_or_generate_image(filename, config)

    mean, median, std = sigma_clipped_stats(image, sigma=config.clip_sigma)
    threshold = median + config.threshold_factor * std

    fwhm = FWHM_estimate(epsf.psfmodel)

    finder = DAOStarFinder(threshold=threshold, fwhm=fwhm)

    grouper = DAOGroup(config.separation_factor*fwhm)

    shape = (epsf.psfmodel.shape/epsf.psfmodel.oversampling).astype(np.int64)

    epsf.fwhm = astropy.modeling.Parameter('fwhm', 'this is not the way to add this I think')
    epsf.fwhm.value = fwhm
    bkgrms = MADStdBackgroundRMS()


    photometry = BasicPSFPhotometry(
        finder=finder,
        group_maker=grouper,
        bkg_estimator=bkgrms,
        psf_model=epsf,
        fitter=LevMarLSQFitter(),
        fitshape=shape
    )

    guess_table = input_table.copy()
    guess_table = cut_edges(guess_table, 101, image.shape[0])
    guess_table.rename_columns(['x', 'y'], ['x_0', 'y_0'])

    # guess_table['x_0'] += np.random.uniform(-0.1, +0.1, size=len(guess_table['x_0']))
    # guess_table['y_0'] += np.random.uniform(-0.1, +0.1, size=len(guess_table['y_0']))

    result_table = photometry(image, guess_table)

    plot_filename = os.path.join(config.output_folder, filename+'_photometry_vs_sources')
    plot_image_with_source_and_measured(image, input_table, result_table, output_path=plot_filename)

    if len(result_table) != 0:
        plot_filename = os.path.join(config.output_folder, filename + '_measurement_offset')
        plot_input_vs_photometry_positions(input_table, result_table, output_path=plot_filename)
    else:
        print(f"No sources found for {filename} with {config}")

    plt.figure()
    plt.imshow(epsf.psfmodel.data)
    save(os.path.join(config.output_folder, filename+'_epsf'), plt.gcf())

    #star_guesses = make_stars_guess(image, finder, 51)
    #plt.figure()
    #plt.imshow(concat_star_images(star_guesses))
    #save(os.path.join(config.output_folder, filename+'_star_guesses'), plt.gcf())

    return PhotometryResult(image, input_table, result_table, epsf, None)


# if __name__ == '__main__':
#     download()
#     config = Config.instance()
#
#     # throw away border pixels to make psf fit into original image
#     psf = read_or_generate_helper('anisocado_psf', config)
#     # TODO why is the generated psf not centered?
#     psf = util.center_cutout_shift_1(psf, (101, 101))
#     psf = psf/psf.max()
#
#     test_images = ['scopesim_grid_16_perturb0', 'scopesim_grid_16_perturb2',
#                    'gauss_cluster_N1000', 'scopesim_cluster']
#
#     config.output_folder = 'output_cheating_astrometry'
#
#     args = [(image, psf, config) for image in test_images]
#
#     if not os.path.exists(config.image_folder):
#          os.mkdir(config.image_folder)
#     if not os.path.exists(config.output_folder):
#         os.mkdir(config.output_folder)
#
#     with mp.Pool(mp.cpu_count()) as pool:
#         result = list(starmap(cheating_astrometry, args))
#
#     plt.show()

if __name__ == '__main__':
    download()
    test_images = testdata_generators.images.keys()
    normal_config = Config.instance()

    gauss_config = Config()
    gauss_config.smoothing = util.make_gauss_kernel()
    gauss_config.output_folder = 'output_files_gaussian_smooth'

    init_guess_config = Config()
    init_guess_config.smoothing = util.make_gauss_kernel()
    init_guess_config.output_folder = 'output_files_initial_guess'
    init_guess_config.use_catalogue_positions = True
    init_guess_config.photometry_iterations = 1  # with known positions we know all stars on first iter

    args = itertools.product(test_images, [normal_config, gauss_config, init_guess_config])

    if not os.path.exists(normal_config.image_folder):
        os.mkdir(normal_config.image_folder)
    if not os.path.exists(normal_config.output_folder):
        os.mkdir(normal_config.output_folder)
    if not os.path.exists(gauss_config.output_folder):
        os.mkdir(gauss_config.output_folder)
    if not os.path.exists(init_guess_config.output_folder):
        os.mkdir(init_guess_config.output_folder)

    with mp.Pool(mp.cpu_count()) as pool:
        # call photometry_full(*args[0]), photometry_full(*args[1]) ...
        results = list(pool.starmap(photometry_full, args))


    plt.show()
    pass
