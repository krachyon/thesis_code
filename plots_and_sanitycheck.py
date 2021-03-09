import pickle
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.colors import LogNorm

import photutils
from util import flux_to_magnitude
from util import match_observation_to_source


def save(filename_base: str, figure: matplotlib.pyplot.figure):
    """Save a matplotlib figure as png and a pickle it to a mlpf file"""
    figure.savefig(filename_base + '.png')
    with open(filename_base + '.mplf', 'wb') as outfile:
        pickle.dump(figure, outfile)


def concat_star_images(stars: photutils.psf.EPSFStars) -> np.ndarray:
    """
    Create a large single image out of EPSFStars to verify cutouts
    :param stars:
    :return: The concatenated image
    """
    assert len(set(star.shape for star in stars)) == 1  # all stars need same shape
    N = int(np.ceil(np.sqrt(len(stars))))
    shape = stars[0].shape
    out = np.zeros(np.array(shape, dtype=int) * N)

    from itertools import product

    for row, col in product(range(N), range(N)):
        if (row + N * col) >= len(stars):
            continue
        xstart = row * shape[0]
        ystart = col * shape[1]

        xend = xstart + shape[0]
        yend = ystart + shape[1]
        i = row + N * col
        out[xstart:xend, ystart:yend] = stars[i].data
    return out


def plot_image_with_source_and_measured(image: np.ndarray, input_table: Table, result_table: Table,
                                        output_path: Optional[str] = None) -> matplotlib.pyplot.figure:
    """
    Plot catalogue and photometry positions on an image
    :param image:
    :param input_table: input catalogue for the image (columns 'x', 'y')
    :param result_table: output from photometry ('x_fit', 'y_fit')
    :param output_path: optionally save the figure to this base filename
    :return:
    """
    plt.figure()
    plt.imshow(image, norm=LogNorm())

    plt.plot(input_table['x'], input_table['y'], 'o', fillstyle='none',
             markeredgewidth=0.5, markeredgecolor='red', label=f'reference N={len(input_table)}')
    plt.plot(result_table['x_fit'], result_table['y_fit'], '.', markersize=1,
             markeredgecolor='orange', label=f'photometry N={len(result_table)}')
    plt.legend()

    if output_path:
        save(output_path, plt.gcf())
    return plt.gcf()


def plot_input_vs_photometry_positions(input_table: Table, result_table: Table,
                                       output_path: Optional[str] = None) -> matplotlib.pyplot.figure:
    """
    Plot the x and y offsets between two catalogues as scatterplot
    :param input_table: Table with columns 'x' and 'y'
    :param result_table: Table with columns 'x_fit' and 'y_fit'
    :param output_path: optionally save plot to this base filename
    :return:
    """

    plt.figure()

    offsets = match_observation_to_source(input_table, result_table)

    plt.title('offset between measured and input')
    plt.xlabel('x offsets [pixel]')
    plt.ylabel('y offsets [pixel]')
    xs = offsets['x_fit'] - offsets['x_orig']
    ys = offsets['y_fit'] - offsets['y_orig']
    plt.plot(xs, ys, '.', markersize=2, markeredgecolor='orange')
    text = f'$σ_x={np.std(xs):.3f}$\n$σ_y={np.std(ys):.3f}$'
    plt.annotate(text, xy=(0, 0.7), xycoords='axes fraction')

    if output_path:
        save(output_path, plt.gcf())
    return plt.gcf()


def plot_deviation_vs_magnitude(input_table: Table, result_table: Table,
                                output_path: Optional[str] = None) -> matplotlib.pyplot.figure:
    """
    Plot the absolute deviation between input and photometry as a function of magnitude
    :param input_table: Table with columns 'x' and 'y'
    :param result_table: Table with columns 'x_fit' and 'y_fit'
    :param output_path: optionally save plot to this base filename
    :return:
    """

    plt.figure()

    offsets = match_observation_to_source(input_table, result_table)
    dists = np.sqrt((offsets['x_fit'] - offsets['x_orig']) ** 2 + (offsets['y_fit'] - offsets['y_orig']) ** 2)
    magnitudes = flux_to_magnitude(offsets['flux_fit'])

    plt.title('Photometry Offset as function of magnitude')
    plt.xlabel('magnitude')
    plt.ylabel('distance offset to reference position [pixel]')
    plt.plot(magnitudes, dists, 'o', markersize=2, markerfacecolor='orange', markeredgewidth=0,
             label=f'$σ={np.std(dists):.3f}$')
    plt.legend()

    if output_path:
        save(output_path, plt.gcf())
    return plt.gcf()
