""" Stuff that was in other files before and is not used right now"""


from typing import List

from astropy.table import Table, Row
from scipy.spatial import cKDTree


def cut_close_stars(peak_table: Table, cutoff_dist: float) -> Table:
    peak_table['nearest'] = 0.
    x_y = np.array((peak_table['x'], peak_table['y'])).T
    lookup_tree = cKDTree(x_y)
    for row in peak_table:
        # find the second nearest neighbour, first one will be the star itself...
        dist, _ = lookup_tree.query((row['x'], row['y']), k=[2])
        row['nearest'] = dist[0]

    peak_table = peak_table[peak_table['nearest'] > cutoff_dist]
    return peak_table


def get_spectral_types() -> List[Row]:
    pickles_lib = pyckles.SpectralLibrary('pickles', return_style='synphot')
    return list(pickles_lib.table['name'])

import copy
import multiprocessing
import os
from typing import List, Tuple, Callable

import astropy.table
import matplotlib.pyplot as plt
import numpy as np
import pyckles
import scopesim
from astropy.io.fits import PrimaryHDU
from astropy.table import Table
from matplotlib.colors import LogNorm

# TODO CHANGEME

from photometry import do_photometry_basic
# TODO Try out cluster algorithms to avoid shifting stars too far if they don't exist in all photometry
#  why did the filter for length of result not eliminate all outliers?
# TODO Disable detector saturation to see if high fluxes get better
#


plt.ion()

# globals/configuration

N_simulation = 8

verbose = True
output = True


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


def match_observations_nearest(observations: Table) -> Table:
    """
    given a set of astrometric measurements, take the first as reference and determine difference in centroid position
    by nearest neighbour search.
    :param observations: Multiple astrometric measurments of same object
    :return: None, modifies tables inplace
    """
    from scipy.spatial import cKDTree  # O(n log n) instead of O(n**2)  lookup speed
    assert (len(observations) != 0)

    ref_obs = observations[observations['run_id'] == 0]
    x_y = np.array((ref_obs['x_fit'], ref_obs['y_fit'])).T
    lookup_tree = cKDTree(x_y)


    observations['x_ref'] = np.nan
    observations['y_ref'] = np.nan
    observations['ref_index'] = -1
    observations['offset'] = np.nan

    for row in observations:
        dist, index = lookup_tree.query((row['x_fit'], row['y_fit']))
        row['x_ref'] = x_y[index, 0]
        row['y_ref'] = x_y[index, 1]
        row['ref_index'] = index
        row['offset'] = dist

    # filter out everything with a distance greater than one pixel -> outliers and failed fits
    if verbose:
        print(f'removing {sum(observations["offset"] > 1)} photometric detections due to distance > 1px')
    observations.remove_rows(observations['offset'] > 1)

    return observations


def match_observations_clustering(observations: Table) -> Table:
    """
    given a set of astrometric measurements, take the first as reference and determine difference in centroid position
    with a cluster finder
    :param observations: Multiple astrometric measurments of same object
    :return: stacked Table
    """
    from sklearn.cluster import dbscan

    x_y = np.array((observations['x_fit'], observations['y_fit'])).T

    # parameters for cluster finder:
    # min_samples: discard everything that does not occcur in at least half of the photometry
    # eps: separation should be no more than a quarter pixel TODO: not sure about the universality of that...
    _, label = dbscan(x_y, min_samples=N_simulation / 2, eps=0.25)

    assert np.all(len(label) == len(x_y))
    observations['ref_index'] = label

    if verbose:
        print(f'{sum(label == -1)} astrometry detections could not be assigned')
    observations.remove_rows(observations['ref_index'] == -1)

    # TODO: Use the information to write back the centroid position to the table and then subtract from the fit
    #  x-y values to get the offset
    #  Problem: How to write it back, use a table join somehow?
    means = observations.group_by('ref_index').groups.aggregate(np.mean)[['ref_index', 'x_fit', 'y_fit', 'flux_fit']]
    means.rename_columns(['x_fit', 'y_fit', 'flux_fit'], ['x_fit_mean', 'y_fit_mean', 'flux_fit_mean'])
    observations = astropy.table.join(observations, means, keys='ref_index')
    observations['offset'] = np.sqrt(
        (observations['x_fit']-observations['x_fit_mean'])**2 +
        (observations['y_fit']-observations['y_fit_mean'])**2
    )
    return observations


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
    photometry_result, residual_image = do_photometry_basic(observed_image, σ)

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

        plt.close('all')


def plot_source_vs_photometry(image: np.ndarray, photometry_table: Table, source: scopesim.Source):
    x_y_observed = np.vstack((photometry_table['x_fit'], photometry_table['y_fit'])).T
    x_y_source = to_pixel_scale(np.array((source.fields[0]['x'], source.fields[0]['y'])).T)

    plt.figure(figsize=(15, 15))
    plt.imshow(image, norm=LogNorm(), vmax=1E5)
    plt.plot(x_y_observed[:, 0], x_y_observed[:, 1], 'o', fillstyle='none',
             markeredgewidth=1, markeredgecolor='red', label='photometry')
    plt.plot(x_y_source[:, 0], x_y_source[:, 1], '^', fillstyle='none',
             markeredgewidth=1, markeredgecolor='orange', label='source')
    plt.legend()
    plt.savefig(f'{output_folder}/source_vs_photometry.png', dpi=300)


def plot_photometry_centroids(image: np.ndarray, photometry_table: Table):
    plt.figure(figsize=(15, 15))
    plt.imshow(image, norm=LogNorm())
    x_y = np.vstack((photometry_table['x_fit'], photometry_table['y_fit'])).T
    plt.plot(x_y[:, 0], x_y[:, 1], 'o', fillstyle='none',
             markeredgewidth=1, markeredgecolor='red', label='photometry')
    plt.savefig(f'{output_folder}/photometry_centroids.png', dpi=300)



def plot_deviation(photometry_table: Table, match_method: Callable[[Table], Table] = match_observations_nearest) \
        -> None:

    matched = match_method(photometry_table)

    # only select stars that are seen in all observations
    # stacked = stacked.group_by('ref_index').groups.filter(lambda tab, keys: len(tab) == len(photometry_results))

    # evil table magic to combine information for objects that where matched to single reference

    means = matched.group_by('ref_index').groups.aggregate(np.mean)
    stds = matched.group_by('ref_index').groups.aggregate(np.std)

    name = match_method.__name__
    plt.figure()
    plt.plot(means['flux_fit'], stds['offset'], 'o')
    plt.axhline(0)
    plt.title(f'flux vs std-deviation of position, method: {name}')
    plt.xlabel('flux')
    plt.ylabel('std deviation over simulations')
    plt.savefig(f'{output_folder}/magnitude_vs_std_{name}.png')

    plt.figure()
    plt.plot(matched['flux_fit'], matched['offset'], '.')
    plt.title(f'spread of measurements, method: {name}')
    plt.xlabel('flux')
    plt.ylabel('measured deviation')
    plt.savefig(f'{output_folder}/magnitude_vs_spread_{name}.png')

    rms = np.sqrt(np.mean(matched['offset'] ** 2))
    print(f'Total RMS of offset between values: {rms}')

if __name__ == '__main__':
    download()

    cluster = make_simcado_cluster()
    # cluster = make_scopesim_cluster()
    stars_in_cluster = len(cluster.fields[0]['x'])  # TODO again, stupid way of looking this information up...

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



    photometry_results = [photometry_result for observed_image, residual_image, photometry_result, sigma in results]
    # make sure we can disentangle results later
    for i, table in enumerate(photometry_results):
        table['run_id'] = i

    photometry_table = astropy.table.vstack(photometry_results)


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

        plt.close('all')
        plot_images(results)
        plt.close('all')

        plot_source_vs_photometry(results[0][0], photometry_table, cluster)

        plt.close('all')
        plot_deviation(photometry_table, match_observations_nearest)
        plot_deviation(photometry_table, match_observations_clustering)

from astropy.io.fits import PrimaryHDU
from config import Config
from testdata_generators import *




def generate():
    # TODO CHANGEME
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
    # TODO CHANGEME
    # generate()
    # make_grid_image(8, 0)
    # make_grid_image(16, 0)
    # make_grid_image(15, 0)
    # make_grid_image(16, 0, perturbation=1.2)
    make_grid_image(25, 0, perturbation=5.)


def verify_methods_with_grid(filename='output_files/grid_16.fits'):
    # TODO CHANGEME
    img = fits.open(filename)[0].data

    epsf_fit = make_epsf_fit(img)
    epsf_combine = make_epsf_combine(img)

    table_fit = do_photometry_epsf(epsf_fit, img)
    table_combine = do_photometry_epsf(epsf_combine, img)

    plt.figure()
    plt.title('EPSF from fit')
    plt.imshow(epsf_fit.data+0.01, norm=LogNorm())

    plt.figure()
    plt.title('EPSF from image combination')
    plt.imshow(epsf_combine.data+0.01, norm=LogNorm())

    plt.figure()
    plt.title('EPSF internal fit')
    plt.imshow(img, norm=LogNorm())
    plt.plot(table_fit['x_fit'], table_fit['y_fit'], 'r.', alpha=0.7)

    plt.figure()
    plt.title('EPSF image combine')
    plt.imshow(img, norm=LogNorm())
    plt.plot(table_combine['x_fit'], table_combine['y_fit'], 'r.', alpha=0.7)

    return epsf_fit, epsf_combine, table_fit, table_combine

def get_cutout(image, position, cutout_size):
    assert position.shape == (2,)

    # clip values less than 0 to 0, greater does not matter, that's ignored
    low = np.clip(np.round(position - cutout_size/2+0.5).astype(int), 0, None)
    high = np.clip(np.round(position + cutout_size/2+0.5).astype(int), 0, None)

    # xy swap
    return image[low[1]:high[1], low[0]:high[0]]

def test_get_cutout():
    testimg = np.arange(0, 100).reshape(10, -1)

    assert np.all(get_cutout(testimg, np.array((0, 0)), 1) == np.array([[0]]))
    assert np.all(get_cutout(testimg, np.array((0, 0)), 2) == np.array([[0,1], [10,11]]))
    # goin of the edge to the upper right
    assert np.all(get_cutout(testimg, np.array((0, 0)), 3) == np.array([[0,1], [10,11]]))

    assert np.all(get_cutout(testimg, np.array((1, 1)), 1) == np.array([[11]]))

    assert np.all(get_cutout(testimg, np.array((0.9, 0.9)), 2) == np.array([[0,1], [10,11]]))
    assert np.all(get_cutout(testimg, np.array((1., 1.)), 2) == np.array([[0,1], [10,11]]))

    assert np.all(get_cutout(testimg, np.array((1.1, 1.1)), 2) == np.array([[11,12], [21,22]]))

    assert np.all(get_cutout(testimg, np.array((5,5)),1000) == testimg)


def run_plots(photometry_result: PhotometryResult):
    image, input_table, result_table, epsf, star_guesses, config, filename = photometry_result

    plot_filename = os.path.join(config.output_folder, filename + '_photometry_vs_sources')
    plot_image_with_source_and_measured(image, input_table, result_table, output_path=plot_filename)

    offsets = util.match_observation_to_source(input_table, result_table)

    if len(result_table) != 0:
        plot_filename = os.path.join(config.output_folder, filename + '_measurement_offset')
        plot_xy_deviation(offsets, output_path=plot_filename)
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





def cheating_astrometry_with_plots(image_recipe: Callable[[], Tuple[np.ndarray, Table]],
                                   image_name: str,
                                   psf: np.ndarray,
                                   config=Config.instance()) -> Union[PhotometryResult, str]:
    image, input_table = read_or_generate_image(image_recipe, image_name, config.image_folder)
    result = cheating_astrometry(image, input_table, psf, image_name, config)
    run_plots(result)
    return result

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
    try:
        image, input_table = read_or_generate_image(image_recipe, image_name, config.image_folder)
        result = run_photometry(image, input_table, image_name, config)
        run_plots(result)

    except Exception as ex:
        import traceback
        print('\033[93m##############\033[0m')
        print(f'error in photometry_with_plots({image_name}, {config})')
        print('\033[93m##############\033[0m')
        error = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        print(error)
        return error

    return result

if __name__ == '__main__':
    from astropy.io import fits

    # Original
    img = fits.open('output_files/observed_00.fits')[0].data
    #img = fits.open('output_files/grid_16_pertubation_1.2.fits')[0].data
    #epsf = make_epsf_combine(img)
    epsf = make_epsf_fit(img, iters=5)
    plt.imshow(epsf.data, norm=LogNorm())
    plt.show()
    # table_psf = do_photometry_epsf(epsf, img)
    # table_basic = do_photometry_basic(img,3)

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
        thesis_lib.testdata_definitions.helpers['anisocado_psf'], 'anisocado_psf', normal_config.image_folder)
    # TODO why is the generated psf not centered?
    psf = util.center_cutout_shift_1(psf, (101, 101))
    psf = psf / psf.max()

    cheating_test_images = ['scopesim_grid_16_perturb0', 'scopesim_grid_16_perturb2',
                            'gauss_grid_16_sigma5_perturb_2', 'anisocado_grid_16_perturb_2',
                            'gauss_cluster_N1000']

    misc_args = [(recipe, name, c)
                 for name, recipe in thesis_lib.testdata_definitions.misc_images.items()
                 for c in (normal_config, gauss_config, init_guess_config)]
    cheat_args = [(thesis_lib.testdata_definitions.misc_images[name], name, psf, cheating_config)
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
    #from .util import DebugPool
    #with DebugPool() as pool:
    with mp.Pool(n_threads) as pool:
        # call photometry_full(*args[0]), photometry_full(*args[1]) ...
        futures = []
        results = []
        futures.append(pool.starmap_async(photometry_with_plots, misc_args))
        futures.append(pool.starmap_async(cheating_astrometry_with_plots, cheat_args))
        futures.append(pool.starmap_async(photometry_with_plots, lowpass_args))
        for future in futures:
            results += future.get()

    results += photometry_multi(recipe_template, 'mag18-24_grid', n_images=10, config=lowpass_config, threads=n_threads)

    # this is not going to scale very well
    with open('../all_photometry_results.pickle', 'wb') as f:
        pickle.dump(results, f)
    plt.close('all')
    pass

def linspace_grid(start: float, stop: float, num: int):
    """
    Construct a 2D meshgrid analog to np.mgrid, but use linspace syntax instead of indexing
    """
    # complex step: use number of steps instead
    return np.mgrid[start:stop:1j*num, start:stop:1j*num]