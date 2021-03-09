import pickle
from typing import Union, Callable, Tuple
import numpy as np
import itertools
from astropy.table import Table
from testdata_generators import read_or_generate_image, read_or_generate_helper
import testdata_generators

from config import Config

from photometry import run_photometry, PhotometryResult, cheating_astrometry

from plots_and_sanitycheck import plot_image_with_source_and_measured, plot_input_vs_photometry_positions, \
    save, concat_star_images, plot_deviation_vs_magnitude

from scopesim_helper import download
import os
import matplotlib.pyplot as plt
import multiprocess as mp  # not multiprocessing, this can pickle lambdas
import util

from itertools import starmap


def run_plots(photometry_result: PhotometryResult):
    image, input_table, result_table, epsf, star_guesses, config, filename = photometry_result
    
    plot_filename = os.path.join(config.output_folder, filename + '_photometry_vs_sources')
    plot_image_with_source_and_measured(image, input_table, result_table, output_path=plot_filename)

    if len(result_table) != 0:
        plot_filename = os.path.join(config.output_folder, filename + '_measurement_offset')
        plot_input_vs_photometry_positions(input_table, result_table, output_path=plot_filename)
        plot_filename = os.path.join(config.output_folder, filename + '_magnitude_v_offset')
        plot_deviation_vs_magnitude(input_table, result_table, output_path=plot_filename)
    else:
        print(f"No sources found for {filename} with {config}")

    plt.figure()
    plt.imshow(epsf.data)
    save(os.path.join(config.output_folder, filename + '_epsf'), plt.gcf())
    plt.figure()
    plt.imshow(concat_star_images(star_guesses))
    save(os.path.join(config.output_folder, filename + '_star_guesses'), plt.gcf())
    plt.close('all')


def photometry_with_plots(image_recipe: Callable[[], Tuple[np.ndarray, Table]],
                          image_name: str,
                          config=Config.instance()) -> Union[PhotometryResult, str]:
    """
    apply EPSF fitting photometry to a testimage
    :param image_recipe: function to generate test image
    :param image_name: name to cache image and print status/errors
    :param config: instance of Config containing all processing parameters
    :return: PhotometryResult
    """
    image, input_table = read_or_generate_image(image_recipe, image_name, config.image_folder)
    result = run_photometry(image, input_table, image_name, config)
    run_plots(result)
    return result


def cheating_astrometry_with_plots(image_recipe: Callable[[], Tuple[np.ndarray, Table]],
                                   image_name: str,
                                   psf: np.ndarray,
                                   config=Config.instance()) -> Union[PhotometryResult, str]:

    image, input_table = read_or_generate_image(image_recipe, image_name, config.image_folder)
    result = cheating_astrometry(image, input_table, psf, image_name, config)
    run_plots(result)
    return result


if __name__ == '__main__':
    download()
    normal_config = Config.instance()

    gauss_config = Config()
    gauss_config.smoothing = util.make_gauss_kernel()
    gauss_config.output_folder = 'output_files_gaussian_smooth'

    init_guess_config = Config()
    init_guess_config.smoothing = util.make_gauss_kernel()
    init_guess_config.output_folder = 'output_files_initial_guess'
    init_guess_config.use_catalogue_positions = True
    init_guess_config.photometry_iterations = 1  # with known positions we know all stars on first iter

    cheating_config = Config()
    cheating_config.output_folder = 'output_cheating_astrometry'

    lowpass_config = Config()
    lowpass_config.smoothing = util.make_gauss_kernel()
    lowpass_config.output_folder = 'output_files_lowpass'
    lowpass_config.use_catalogue_positions = True
    lowpass_config.photometry_iterations = 1  # with known positions we know all stars on first iter

    configs = [normal_config, gauss_config, init_guess_config, cheating_config, lowpass_config]
    for config in configs:
        if not os.path.exists(config.image_folder):
            os.mkdir(config.image_folder)
        if not os.path.exists(config.output_folder):
            os.mkdir(config.output_folder)

    # throw away border pixels to make psf fit into original image
    psf = read_or_generate_helper(
        testdata_generators.helpers['anisocado_psf'], 'anisocado_psf', normal_config.image_folder)
    # TODO why is the generated psf not centered?
    psf = util.center_cutout_shift_1(psf, (101, 101))
    psf = psf/psf.max()

    cheating_test_images = ['scopesim_grid_16_perturb0', 'scopesim_grid_16_perturb2',
                            'gauss_grid_16_sigma5_perturb_2', 'anisocado_grid_16_perturb_2',
                            'gauss_cluster_N1000']

    misc_args = [(recipe, name, c)
                 for name, recipe in testdata_generators.misc_images.items()
                 for c in (normal_config, gauss_config, init_guess_config)]
    cheat_args = [(testdata_generators.misc_images[name], name, psf, cheating_config)
                  for name in cheating_test_images]
    lowpass_args = [(recipe, name, c)
                    for name, recipe in testdata_generators.lowpass_images.items()
                    for c in (lowpass_config,)]

    # Honestly you'll have to change this yourself for your machine. Too much and you won't have enough memory
    with mp.Pool(10) as pool:
        # call photometry_full(*args[0]), photometry_full(*args[1]) ...
        future1 = pool.starmap_async(photometry_with_plots, misc_args)
        future2 = pool.starmap_async(cheating_astrometry_with_plots, cheat_args)
        future3 = pool.starmap_async(photometry_with_plots, lowpass_args)
        results = list(future1.get()) + list(future2.get()) + list(future3.get())

    # this is not going to scale very well
    with open('all_photometry_results.pickle', 'wb') as f:
        pickle.dump(results, f)
    plt.close('all')
    pass
