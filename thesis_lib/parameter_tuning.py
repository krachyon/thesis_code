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


from photutils.detection import DAOStarFinder
from photutils import DAOGroup

from astropy.stats import sigma_clipped_stats

from .config import Config
from . import testdata_generators
from . import util
from .photometry import run_photometry, get_finder

from scipy.interpolate import griddata
from matplotlib.colors import LogNorm


def plot_shape(result: skopt.utils.OptimizeResult, dimensions=None) -> matplotlib.figure.Figure:
    """
    Try and visualize the shape of the function with 2D/1D cuts through the search space.
    Lower triangle is average of objective function at that position,
    Upper triangle is standard deviation of objective function values
    Diagonal is 1D plot of parameter vs objective
    :param result: OptimizeResult from running a skopt.Optimizer or helper function
    :param dimensions: Names for function parameters that where optimized
    :return:
    """

    x = np.array(result.x_iters)
    y = np.array(result.func_vals)
    if dimensions:
        n_dims = len(dimensions)
        assert n_dims <= result.space.n_dims
    else:
        dimensions = result.space.dimension_names
        n_dims = result.space.n_dims

        #dimensions = [f'$X_{i}$' for i in range(n_dims)]
    

    fig, ax = plt.subplots(n_dims, n_dims,
                           figsize=(2 * n_dims, 2 * n_dims))

    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.02, top=0.98,
                        hspace=0.1, wspace=0.05)


    img_points = 300

    for row in range(n_dims):
        for col in range(n_dims):
            if row == col:
                order = np.argsort(x[:,row])
                ax[row, row].plot(x[:,row][order], y[order], 'o')
                ax[row, row].plot(result.x[row], result.fun, 'x')
                ax[row, row].set_box_aspect(1)
            # lower triangle
            elif row > col:
                points = []
                avgs = []
                stds = []
                x0x1 = x[:, (col, row)]

                for entry in np.unique(x0x1, axis=0):
                    points.append(entry)
                    locations = np.all(x0x1 == entry, axis=1)
                    avgs.append(np.mean(y[locations]))
                    stds.append(np.std(y[locations]))

                x0_low, x0_high = np.min(x0x1[:, 0]), np.max(x0x1[:, 0])
                x1_low, x1_high = np.min(x0x1[:, 1]), np.max(x0x1[:, 1])

                X, Y = np.mgrid[x0_low:x0_high:img_points*1j, x1_low:x1_high:img_points*1j]

                avg_img = griddata(points, np.array(avgs), (X, Y), method='nearest').T
                avg_img -= np.min(avg_img)
                avg_img /= np.max(avg_img)
                avg_img[~np.isfinite(avg_img)] = np.nanmax(np.hstack((avg_img[avg_img<np.inf],[0.])))
                avg_img += 0.0001
                assert(np.all(np.isfinite(avg_img)))

                std_img = griddata(points, np.array(stds), (X, Y), method='nearest').T
                std_img -= np.min(std_img)
                std_img /= np.max(std_img)
                std_img[~np.isfinite(std_img)] = np.nanmax(np.hstack((std_img[std_img<np.inf],[0.])))
                std_img += 0.0001
                assert(np.all(np.isfinite(std_img)))

                n_ticks = 6

                xticks = list(np.linspace(0, img_points-1, n_ticks))
                yticks = list(np.linspace(0, img_points-1, n_ticks))

                xtick_labels = [f'{i:.2f}' for i in np.linspace(x0_low, x0_high, n_ticks)]
                ytick_labels = [f'{i:.2f}' for i in np.linspace(x1_low, x1_high, n_ticks)]

                la = ax[row, col]
                im = la.imshow(avg_img, origin='lower', norm=LogNorm())
                la.plot((result.x[col]-x0_low)/(x0_high-x0_low)*img_points, (result.x[row]-x1_low)/(x1_high-x1_low)*img_points, 'r*')
                la.set_xticks(xticks)
                la.set_yticks(yticks)
                la.set_xticklabels(xtick_labels)
                la.set_yticklabels(ytick_labels)
                la.set_xlabel(dimensions[col])
                la.set_ylabel(dimensions[row])
                #fig.colorbar(im, ax=ax[i, j])

                ua = ax[col, row]
                im = ua.imshow(std_img, origin='lower', norm=LogNorm())
                ua.plot((result.x[col]-x0_low)/(x0_high-x0_low)*img_points, (result.x[row]-x1_low)/(x1_high-x1_low)*img_points, 'r*')
                ua.set_xticks(xticks)
                ua.set_yticks(yticks)
                ua.set_xticklabels(xtick_labels)
                ua.set_yticklabels(ytick_labels)
                ua.set_xlabel(dimensions[col])
                ua.set_ylabel(dimensions[row])
                #fig.colorbar(im, ax=ax[j, i])
    return fig


def run_optimizer(optimizer: skopt.Optimizer, objective: Callable[[Any], float],
                  n_evaluations: int,
                  n_processes=None) -> skopt.utils.OptimizeResult:
    """

    :param optimizer:
    :param objective:
    :param n_evaluations:
    :param n_processes:
    :return:
    """

    Job = namedtuple('Job', ('result', 'args'))

    def not_ready(job: Job):
        return not job.result.ready()

    if n_processes is None:
        n_processes = mp.cpu_count()

    #from thesis_lib.util import DebugPool
    #with DebugPool() as p:
    with mp.Pool(n_processes) as p:
        jobs = []
        try:
            for i in range(n_evaluations):
                args = optimizer.ask()
                optimizer.update_next()
                print('\033[93m##############\033[0m')
                print(f'Evaluation #{i}')
                print(list(zip(optimizer.space.dimension_names, args)))
                print('\033[93m##############\033[0m')
                jobs.append(Job(p.apply_async(objective, args), args))
                for job in jobs:
                    if job.result.ready():
                        optimizer.tell(job.args, job.result.get())
                        jobs.remove(job)
                while sum(map(not_ready, jobs)) >= n_processes:
                    time.sleep(0.5)
            for job in jobs:
                optimizer.tell(job.args, job.result.get())
        except KeyboardInterrupt:
            pass

    return optimizer.get_result()


def make_starfinder_objective(image_recipe: Callable, image_name: str) -> Callable:

    image, input_table = testdata_generators.read_or_generate_image(image_recipe, image_name)
    mean, median, std = sigma_clipped_stats(image)

    xym_pixel = np.array((input_table['x'], input_table['y'])).T
    lookup_tree = cKDTree(xym_pixel)


    def starfinder_objective(threshold, fwhm, sigma_radius, roundlo, roundhi, sharplo, sharphi):
        res_table = DAOStarFinder(threshold=median+std*threshold,
                                  fwhm=fwhm,
                                  sigma_radius=sigma_radius,
                                  sharplo=sharplo,
                                  sharphi=sharphi,
                                  roundlo=roundlo,
                                  roundhi=roundhi,
                                  exclude_border=True
                                  )(image)
        if not res_table:
            return 3 * len(input_table)

        xys = structured_to_unstructured(np.array(res_table['xcentroid', 'ycentroid']))
        seen_indices = set()
        offsets = []
        for xy in xys:
            dist, index = lookup_tree.query(xy)
            if dist > 2 or index in seen_indices:
                offsets.append(np.nan)
            else:
                offsets.append(dist)
            seen_indices.add(index)

        offsets += [np.nan] * len(seen_indices - set(lookup_tree.indices))
        offsets += [np.nan] * abs(len(input_table)-len(res_table))
        offsets = np.array(offsets)
        offsets -= np.nanmean(offsets)
        offsets[np.isnan(offsets)] = 50.

        return np.sqrt(np.sum(np.array(offsets)**2))

    return starfinder_objective


def make_epsf_objective(config: Config, image_recipe: Callable, image_name: str) -> Callable:

    dimensions = [Integer(5, 40, name='cutout_size'),
                  Integer(1, 10, name='fitshape_half'),
                  Real(0.30, 3., name='sigma'),
                  Categorical([4, 5, 7, 10], name='iters'),
                  Categorical([1, 2, 4], name='oversampling')]

    def epsf_objective(cutout_size: int, fitshape_half: int, sigma: float, iters: int, oversampling: int, separation: float):

        config.oversampling = oversampling
        config.smoothing = util.make_gauss_kernel(sigma)
        config.fitshape = fitshape_half * 2 + 1
        config.cutout_size = cutout_size
        config.epsfbuilder_iters = iters
        config.separation_factor = separation

        image, input_table = testdata_generators.read_or_generate_image(image_recipe, image_name, config.image_folder)
        try:
            find_result = get_finder(image, config)(image)
            find_result.rename_columns(['xcentroid', 'ycentroid'], ['x_0', 'y_0'])
            group_table = DAOGroup(separation * config.fwhm_guess)(find_result)
            if any([len(i) > 10 for i in group_table.group_by('group_id').groups]):
                print('\n### too big group### \n')
                raise ValueError('too big group')

            result = run_photometry(image, input_table, image_name, config)
            result_table = util.match_observation_to_source(input_table, result.result_table)

            loss = np.sqrt(np.sum(result_table['offset']**2))
        except Exception as ex:
            loss = np.sqrt(np.sum(len(input_table)*[10**2]))
        finally:
            return loss

    return epsf_objective
