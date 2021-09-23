import matplotlib.figure
import multiprocess as mp
import numpy as np
import dill
import time
import matplotlib.pyplot as plt
import skopt
from skopt.space import Real, Integer, Categorical, Dimension
from typing import Callable, Any
from collections import namedtuple
from scipy.spatial import cKDTree
from numpy.lib.recfunctions import structured_to_unstructured
from typing import Optional, List

import thesis_lib.testdata_definitions
from photutils.detection import DAOStarFinder

from astropy.stats import sigma_clipped_stats

from thesis_lib.config import Config
from thesis_lib import testdata_generators
from thesis_lib import util
from thesis_lib.photometry import run_photometry
from thesis_lib.parameter_tuning import run_optimizer

from thesis_lib.astrometry_plots import plot_xy_deviation
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm


# def make_starfinder_objective(image_recipe: Callable, image_name: str) -> Callable:
#
#     image, input_table = testdata_generators.read_or_generate_image(image_recipe, image_name)
#     mean, median, std = sigma_clipped_stats(image)
#
#     xym_pixel = np.array((input_table['x'], input_table['y'])).T
#     lookup_tree = cKDTree(xym_pixel)
#
#
#     def starfinder_objective(threshold, fwhm, sigma_radius, roundlo, roundhi, sharplo, sharphi):
#         res_table = DAOStarFinder(threshold=median+std*threshold,
#                                   fwhm=fwhm,
#                                   sigma_radius=sigma_radius,
#                                   sharplo=sharplo,
#                                   sharphi=sharphi,
#                                   roundlo=roundlo,
#                                   roundhi=roundhi,
#                                   exclude_border=True
#                                   )(image)
#         if not res_table:
#             return 3 * len(input_table)
#
#         xys = structured_to_unstructured(np.array(res_table['xcentroid', 'ycentroid']))
#         seen_indices = set()
#         offsets = []
#         for xy in xys:
#             dist, index = lookup_tree.query(xy)
#             if dist > 2 or index in seen_indices:
#                 offsets.append(np.nan)
#             else:
#                 offsets.append(dist)
#             seen_indices.add(index)
#
#         offsets += [np.nan] * len(seen_indices - set(lookup_tree.indices))
#         offsets += [np.nan] * abs(len(input_table)-len(res_table))
#         offsets = np.array(offsets)
#         offsets -= np.nanmean(offsets)
#         offsets[np.isnan(offsets)] = 50.
#
#         return np.sqrt(np.sum(np.array(offsets)**2))
#
#     return starfinder_objective
from thesis_lib.parameter_tuning import make_starfinder_objective

if __name__ == '__main__':
    name = 'gausscluster_N2000_mag22'
    recipe = thesis_lib.testdata_definitions.benchmark_images[name]

    img, input_table = testdata_generators.read_or_generate_image(recipe, name)

    starfinder_obj = make_starfinder_objective(recipe, name)
    starfinder_dims = [
        Real(-6., 3, name='threshold'),
        Real(2., 8, name='fwhm'),
        Real(2., 5.5, name='sigma_radius'),
        Real(-15., 0., name='roundlo'),
        Real(0., 15., name='roundhi'),
        Real(-15, 0., name='sharplo'),
        Real(0., 10., name='sharphi')
    ]

    starfinder_optimizer = skopt.Optimizer(
        dimensions=starfinder_dims,
        n_jobs=mp.cpu_count(),
        random_state=1,
        base_estimator='RF',
        n_initial_points=1000,
        initial_point_generator='random'
    )

    starfinder_result = run_optimizer(starfinder_optimizer, starfinder_obj, n_evaluations=1400)
    x = starfinder_result.x

    print(list(zip(starfinder_result.space.dimension_names, starfinder_result.x)))
    mean, median, std = sigma_clipped_stats(img, sigma=3)
    threshold = median + x[0] * std

    finder = DAOStarFinder(threshold=threshold,
        fwhm=x[1],
        sigma_radius=x[2],
        roundlo=x[3],
        roundhi=x[4],
        sharplo=x[5],
        sharphi=x[6],
        exclude_border=True)

    result_table = finder(img)

    plt.imshow(img, norm=LogNorm(), cmap='inferno')
    plt.plot(input_table['x'], input_table['y'], 'o', fillstyle='none',
             markeredgewidth=0.5, markeredgecolor='red', label=f'reference N={len(input_table)}')
    plt.plot(result_table['xcentroid'], result_table['ycentroid'], 'g.', markersize=1, label=f'photometry N={len(result_table)}')
    plt.legend()
    plt.show()