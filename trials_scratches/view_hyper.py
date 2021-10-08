import thesis_lib.testdata.definitions
from thesis_lib.testdata import generators
from thesis_lib.photometry import run_photometry
from thesis_lib.config import Config
from thesis_lib import util
from thesis_lib.astrometry.plots import *

image_name = 'gausscluster_N2000_mag22_lowpass'
image_recipe = thesis_lib.testdata.definitions.benchmark_images[image_name]


def view_objective(cutout_size: int, fitshape_half: int, sigma: float, iters:int):
    config = Config()
    config.use_catalogue_positions = True
    config.photometry_iterations = 1
    config.oversampling = 2

    config.smoothing = util.make_gauss_kernel(sigma)
    config.fitshape = fitshape_half*2+1
    config.cutout_size = cutout_size
    config.epsfbuilder_iters=iters

    image, input_table = generators.read_or_generate_image(image_recipe, image_name, config.image_folder)
    result = run_photometry(image, input_table, image_name, config)
    result_table = util.match_observation_to_source(input_table, result.result_table)

    plot_xy_deviation(result_table)
    plot_deviation_vs_magnitude(result_table)
    plot_image_with_source_and_measured(image, input_table, result_table)
    return result, result_table


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


if __name__=='__main__':

    res, res_tab = view_objective(7, 15, 0.10553113688384372, 18)

    plt.show()
