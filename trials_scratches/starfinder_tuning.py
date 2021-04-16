import multiprocess as mp
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import matplotlib.pyplot as plt
import dill
import os
from collections import namedtuple
import time
from astropy.stats import sigma_clipped_stats

from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_objective, plot_convergence, plot_evaluations
from scipy.spatial import cKDTree

from photutils.detection import IRAFStarFinder, DAOStarFinder

from thesis_lib.testdata_generators import read_or_generate_image, benchmark_images
from thesis_lib.util import DebugPool

from scipy.interpolate import griddata
from matplotlib.colors import LogNorm


def plot_shape(result, dimensions=None):

    space = result.space
    n_dims = space.n_dims

    x = np.array(result.x_iters)
    y = np.array(result.func_vals)
    if dimensions:
        assert len(dimensions) == n_dims
    else:
        dimensions = [f'$X_{i}$' for i in range(n_dims)]

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


result_filename = 'daofind_opt_full_threshfwhmsigma.pkl'
image_name = 'gausscluster_N2000_mag22'
image_recipe = benchmark_images[image_name]
img, ref_table = read_or_generate_image(image_recipe, image_name)

# for lowpass:
# roundlo = -32.6
# roundhi = 3.91
# sharplo = -21.70
# sharphi = 17.73
# threshold, fwhm, sigma_radius = [-0.5961880021208964, 2.820694454041728, 3.0288937031114287]

# for no lowpass:
# roundlo, roundhi, sharplo, sharphi = -10, 8.5, -11, 2.4
# thresh, fwhm, sigma_radius = -0.9985482815217563, 2.790086326795894, 4.015963504846637
# thresh, fwhm, sigma_radius = -2.942323484472957, 2.24827752258762, 4.367898600244883



dimensions = ('threshold', 'fwhm', 'sigma_radius')#, 'roundlo', 'roundhi', 'sharplo', 'sharphi'
dim_vals = [(-5., -0.5), (2., 3.7), (3., 4.5)]
n_procs = 10
n_initial = 800
n_eval = 1200

roundlo, roundhi, sharplo, sharphi = -10, 8.5, -11, 2.4
xym_pixel = np.array((ref_table['x'], ref_table['y'], ref_table['m'])).T
lookup_tree = cKDTree(xym_pixel[:, :2])  # only feed x and y to the lookup tree

mean, median, std = sigma_clipped_stats(img)


def objective(threshold, fwhm, sigma_radius):
    res_table = DAOStarFinder(threshold=median+std*threshold,
                              fwhm=fwhm,
                              sigma_radius=sigma_radius,
                              sharplo=sharplo,
                              sharphi=sharphi,
                              roundlo=roundlo,
                              roundhi=roundhi,
                              )(img)
    if not res_table:
        return 2000

    xys = structured_to_unstructured(np.array(res_table['xcentroid', 'ycentroid']))
    seen_indices = set()
    offsets = []
    for xy in xys:
        dist, index = lookup_tree.query(xy)
        if dist > 1 or index in seen_indices:
            offsets.append(np.nan)
        else:
            offsets.append(dist)
        seen_indices.add(index)

    offsets += [np.nan] * abs(len(res_table) - len(ref_table))
    offsets += [np.nan] * abs(len(seen_indices) - len(res_table))
    offsets = np.array(offsets)
    offsets -= np.nanmean(offsets)
    offsets[np.isnan(offsets)] = 1.

    return np.sqrt(np.sum(np.array(offsets)**2))


Job = namedtuple('Job', ('result', 'args'))


def not_ready(job: Job):
    return not job.result.ready()


if __name__ == '__main__':

    optimizer = Optimizer(
        dimensions=dim_vals,
        n_jobs=12,
        random_state=1,
        base_estimator='RF',
        n_initial_points=n_initial,
        initial_point_generator='random'
    )

    with mp.Pool(n_procs) as p:
    #with DebugPool() as p:
        jobs = []
        try:
            for i in range(n_eval):
                args = optimizer.ask()
                optimizer.update_next()
                print('#######')
                print(f'Evaluation #{i}')
                print(args)
                print('#######')
                jobs.append(Job(p.apply_async(objective, args), args))
                for job in jobs:
                    if job.result.ready():
                        optimizer.tell(job.args, job.result.get())
                        jobs.remove(job)
                while sum(map(not_ready, jobs)) >= n_procs:
                    time.sleep(0.5)
            for job in jobs:
                optimizer.tell(job.args, job.result.get())
        except KeyboardInterrupt:
            pass

    res = optimizer.get_result()

    with open(result_filename, 'wb') as f:
        dill.dump(optimizer, f)

    threshold, fwhm, sigma_radius = res.x
    res_table = DAOStarFinder(threshold=median+std*threshold,
                              fwhm=fwhm,
                              sigma_radius=sigma_radius,
                              sharplo=sharplo,
                              sharphi=sharphi,
                              roundlo=roundlo,
                              roundhi=roundhi
                              )(img)



    plt.ion()
    plt.imshow(img)
    plt.plot(res_table['xcentroid'], res_table['ycentroid'], 'ro', markersize=0.5)

    plot_evaluations(res, dimensions=dimensions)
    plt.figure()
    ax = plot_convergence(res)
    ax.set_yscale('log')

    plot_objective(res, sample_source='result', dimensions=dimensions)

    plot_shape(res, dimensions=dimensions)
    plt.show()


