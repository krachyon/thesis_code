import copy
import multiprocessing
import os
import tempfile
from typing import List, Tuple, Callable

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

