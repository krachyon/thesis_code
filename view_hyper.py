import matplotlib.pyplot as plt
from thesis_lib import testdata_generators
from thesis_lib.photometry import run_photometry
from thesis_lib.config import Config
from thesis_lib import util
from thesis_lib.plots_and_sanitycheck import *

image_name = 'scopesim_grid_16_perturb2_mag18_24'
image_recipe = testdata_generators.benchmark_images[image_name]


def view_objective(cutout_size: int, fitshape_half: int, sigma: float, iters:int):
    config = Config()
    config.use_catalogue_positions = True
    config.photometry_iterations = 1
    config.oversampling = 1

    config.smoothing = util.make_gauss_kernel(sigma)
    config.fitshape = fitshape_half*2+1
    config.cutout_size = cutout_size
    config.epsfbuilder_iters=iters

    image, input_table = testdata_generators.read_or_generate_image(image_recipe, image_name, config.image_folder)
    result = run_photometry(image, input_table, image_name, config)
    result_table = util.match_observation_to_source(input_table, result.result_table)

    plot_input_vs_photometry_positions(result_table)
    plot_deviation_vs_magnitude(result_table)
    plot_image_with_source_and_measured(image,input_table,result_table)
    return result, result_table

from scipy.interpolate import griddata
from matplotlib.colors import LogNorm

def plot_evaluations(result, bins=20, dimensions=None):
    """Visualize the order in which points were sampled during optimization.

    This creates a 2-d matrix plot where the diagonal plots are histograms
    that show the distribution of samples for each search-space dimension.

    The plots below the diagonal are scatter-plots of the samples for
    all combinations of search-space dimensions.

    The order in which samples
    were evaluated is encoded in each point's color.

    A red star shows the best found parameters.

    Parameters
    ----------
    result : `OptimizeResult`
        The optimization results from calling e.g. `gp_minimize()`.

    bins : int, bins=20
        Number of bins to use for histograms on the diagonal.

    dimensions : list of str, default=None
        Labels of the dimension
        variables. `None` defaults to `space.dimensions[i].name`, or
        if also `None` to `['X_0', 'X_1', ..]`.

    plot_dims : list of str and int, default=None
        List of dimension names or dimension indices from the
        search-space dimensions to be included in the plot.
        If `None` then use all dimensions except constant ones
        from the search-space.

    Returns
    -------
    ax : `Matplotlib.Axes`
        A 2-d matrix of Axes-objects with the sub-plots.

    """
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
                ax[row, row].plot(x[:,row][order], y[order])
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
                std_img = griddata(points, np.array(stds), (X, Y), method='nearest').T

                n_ticks = 6

                xticks = list(np.linspace(0, img_points-1, n_ticks))
                yticks = list(np.linspace(0, img_points-1, n_ticks))

                xtick_labels = [f'{i:.2f}' for i in np.linspace(x0_low, x0_high, n_ticks)]
                ytick_labels = [f'{i:.2f}' for i in np.linspace(x1_low, x1_high, n_ticks)]

                la = ax[row, col]
                im = la.imshow(avg_img, origin='lower')
                breakpoint()
                la.plot(result.x[col]/x0_high*img_points, result.x[row]/x1_high*img_points, 'r*')
                la.set_xticks(xticks)
                la.set_yticks(yticks)
                la.set_xticklabels(xtick_labels)
                la.set_yticklabels(ytick_labels)
                la.set_xlabel(dimensions[col])
                la.set_ylabel(dimensions[row])
                #fig.colorbar(im, ax=ax[i, j])

                ua = ax[col, row]
                im = ua.imshow(std_img, origin='lower')
                ua.plot(result.x[col]/x0_high*img_points, result.x[row]/x1_high*img_points, 'r*')
                ua.set_xticks(xticks)
                ua.set_yticks(yticks)
                ua.set_xticklabels(xtick_labels)
                ua.set_yticklabels(ytick_labels)
                ua.set_xlabel(dimensions[col])
                ua.set_ylabel(dimensions[row])
                #fig.colorbar(im, ax=ax[j, i])




if __name__=='__main__':

    res, res_tab = view_objective(7, 15, 0.10553113688384372, 18)

    plt.show()