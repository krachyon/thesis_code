import os
import pickle
from typing import Union, Callable, Tuple, Optional

import matplotlib.pyplot as plt
import multiprocess as mp  # not multiprocessing, this can pickle lambdas
from multiprocess.pool import Pool as Pool_t
import numpy as np
from astropy.table import Table
import astropy.table

from .import testdata_generators
from .import util
from .config import Config
from .photometry import run_photometry, PhotometryResult, cheating_astrometry
from .plots_and_sanitycheck import plot_image_with_source_and_measured, plot_input_vs_photometry_positions, \
    save, concat_star_images, plot_deviation_vs_magnitude, plot_deviation_histograms
from .scopesim_helper import download
from .testdata_generators import read_or_generate_image, read_or_generate_helper


def run_plots(photometry_result: PhotometryResult):
    image, input_table, result_table, epsf, star_guesses, config, filename = photometry_result

    plot_filename = os.path.join(config.output_folder, filename + '_photometry_vs_sources')
    plot_image_with_source_and_measured(image, input_table, result_table, output_path=plot_filename)

    offsets = util.match_observation_to_source(input_table, result_table)

    if len(result_table) != 0:
        plot_filename = os.path.join(config.output_folder, filename + '_measurement_offset')
        plot_input_vs_photometry_positions(offsets, output_path=plot_filename)
        plot_filename = os.path.join(config.output_folder, filename + '_magnitude_v_offset')
        plot_deviation_vs_magnitude(offsets, output_path=plot_filename)
        plot_filename = os.path.join(config.output_folder, filename + '_histogram')
        plot_deviation_histograms(offsets, output_path=plot_filename)
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


def photometry_multi(image_recipe_template: Callable[[int], Callable[[], Tuple[np.ndarray, Table]]],
                     image_name_template: str,
                     n: int,
                     config=Config.instance(),
                     threads: Union[int, None]=None) -> Table:
    """
    apply EPSF fitting photometry to a testimage
    :param image_recipe: function to generate test image
    :param image_name: name to cache image and print status/errors
    :param config: instance of Config containing all processing parameters
    :return: table with results and matched input catalogue
    """

    def inner(i):
        image_recipe = image_recipe_template(i)
        image_name = image_name_template+f'_{i}'
        image, input_table = read_or_generate_image(image_recipe, image_name, config.image_folder)
        result = run_photometry(image, input_table, image_name, config)
        return util.match_observation_to_source(input_table, result.result_table)

    if threads:
        with mp.Pool(threads) as pool:
            partial_results = pool.map(inner, range(n))
    else:
        partial_results = list(map(inner, range(n)))

    matched_result = astropy.table.vstack(partial_results)

    plot_filename = os.path.join(config.output_folder, image_name_template + '_measurement_offset')
    plot_input_vs_photometry_positions(matched_result, output_path=plot_filename)
    plot_filename = os.path.join(config.output_folder, image_name_template + '_magnitude_v_offset')
    plot_deviation_vs_magnitude(matched_result, output_path=plot_filename)

    plot_filename = os.path.join(config.output_folder, image_name_template + '_histogram')
    plot_deviation_histograms(matched_result, output_path=plot_filename)
    plt.close('all')
    return matched_result




def cheating_astrometry_with_plots(image_recipe: Callable[[], Tuple[np.ndarray, Table]],
                                   image_name: str,
                                   psf: np.ndarray,
                                   config=Config.instance()) -> Union[PhotometryResult, str]:
    image, input_table = read_or_generate_image(image_recipe, image_name, config.image_folder)
    result = cheating_astrometry(image, input_table, psf, image_name, config)
    run_plots(result)
    return result


def main():
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
    lowpass_config.separation_factor = 0.1

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
    psf = psf / psf.max()

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

    def recipe_template(seed):
        return lambda: testdata_generators.scopesim_grid(seed=seed, N1d=30, perturbation=2.,
                                                         psf_transform=testdata_generators.lowpass(),
                                                         magnitude=lambda N: np.random.uniform(18, 24, N))

    # Honestly you'll have to change this yourself for your machine. Too much and you won't have enough memory
    n_threads = 10
    with mp.Pool(n_threads) as pool:
        # call photometry_full(*args[0]), photometry_full(*args[1]) ...
        futures = []
        results = []
        futures.append(pool.starmap_async(photometry_with_plots, misc_args))
        futures.append(pool.starmap_async(cheating_astrometry_with_plots, cheat_args))
        futures.append(pool.starmap_async(photometry_with_plots, lowpass_args))
        for future in futures:
            results += future.get()

    results += photometry_multi(recipe_template, 'mag18-24_grid', n=10, config=lowpass_config, threads=n_threads)

    # this is not going to scale very well
    with open('../all_photometry_results.pickle', 'wb') as f:
        pickle.dump(results, f)
    plt.close('all')
    pass


if __name__ == '__main__':
    main()