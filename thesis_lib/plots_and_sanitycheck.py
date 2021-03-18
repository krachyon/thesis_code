import pickle
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.colors import LogNorm

import photutils
from .util import flux_to_magnitude, match_observation_to_source


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


def plot_input_vs_photometry_positions(matched_table: Table,
                                       output_path: Optional[str] = None) -> matplotlib.pyplot.figure:
    """
    Plot the x and y offsets between two catalogues as scatterplot
    :param matched_table: Table with columns 'x_fit', 'y_fit', 'x_orig', 'y_orig', 'offset'
    :param output_path: optionally save plot to this base filename
    :return:
    """

    plt.figure()

    plt.title('offset between measured and input')
    plt.xlabel('x offsets [pixel]')
    plt.ylabel('y offsets [pixel]')
    xs = matched_table['x_fit'] - matched_table['x_orig']
    ys = matched_table['y_fit'] - matched_table['y_orig']
    plt.plot(xs, ys, '.', markersize=2, markeredgecolor='orange')
    text = f'$σ_x={np.std(xs):.3f}$\n$σ_y={np.std(ys):.3f}$'
    plt.annotate(text, xy=(0.01, 0.9), xycoords='axes fraction')

    if output_path:
        save(output_path, plt.gcf())
    return plt.gcf()


def plot_deviation_vs_magnitude(matched_table: Table,
                                output_path: Optional[str] = None) -> matplotlib.pyplot.figure:
    """
    Plot the absolute deviation between input and photometry as a function of magnitude
    :param matched_table: Table with columns 'x_fit', 'y_fit', 'x_orig', 'y_orig', 'offset'
    :param output_path: optionally save plot to this base filename
    :return:
    """

    plt.figure()

    dists = matched_table['offset']

    #magnitudes = flux_to_magnitude(matched_table['flux_fit'])
    magnitudes = matched_table['m']

    order = np.argsort(magnitudes)
    dists = dists[order]
    magnitudes = magnitudes[order]

    window_size = 41
    x_offset = int(window_size/2)
    dists_slide = np.lib.stride_tricks.sliding_window_view(dists, window_size)

    dists_mean = np.mean(dists_slide, axis=1)
    dists_std = np.std(dists_slide, axis=1)

    plt.title('Photometry Offset as function of magnitude')
    plt.xlabel('magnitude')
    plt.ylabel('distance offset to reference position [pixel]')
    plt.plot(magnitudes, dists, 'o', markersize=2, markerfacecolor='orange', markeredgewidth=0,
             label=f'$σ={np.std(dists):.3f}$')
    plt.plot(magnitudes[x_offset:-x_offset], dists_mean, color='blue', alpha=0.4)
    plt.fill_between(magnitudes[x_offset:-x_offset], dists_mean-dists_std, dists_mean+dists_std,
                     label=f'sliding average $\pm σ$; window_size={window_size}', alpha=0.2, color='blue')
    plt.plot(magnitudes[x_offset:-x_offset], dists_std, linewidth=1, color='green',
             label=f'sliding σ, window_size={window_size}')
    plt.legend()

    if output_path:
        save(output_path, plt.gcf())
    return plt.gcf()


def plot_deviation_histograms(matched_table: Table, output_path: Optional[str] = None) -> matplotlib.pyplot.figure:

    fig, axes = plt.subplots(2, 2, sharex='all')

    axes[0,0].hist(matched_table['offset'], bins=60)
    σ = np.nanstd(matched_table['offset'])
    axes[0,0].axvline(σ, label=f'σ={σ:.3f}', color='r')
    axes[0,0].set_title('distance offset input<->measured')
    axes[0,0].legend()
    axes[0,0].set_xlabel('[pixel]')

    # TODO m is the original value or the starfinder value;
    #  need to calculate from flux_fit
    magnitude_diff = np.array(matched_table['m'] - flux_to_magnitude(matched_table['flux_fit']))
    magnitude_diff = magnitude_diff - np.nanmean(magnitude_diff)
    axes[0,1].hist(magnitude_diff, bins=60)
    σ = np.nanstd(magnitude_diff)
    axes[0,1].axvline(σ, label=f'σ={σ:.3f}', color='r')
    axes[0,1].set_title('magnitude offset input<->measured')
    axes[0,1].legend()
    axes[0,1].set_xlabel('[mag]')

    axes[1,0].hist(matched_table['x_fit']-matched_table['x_orig'], bins=60)
    σ = np.nanstd(matched_table['x_fit']-matched_table['x_orig'])
    axes[1,0].axvline(σ, label=f'σ={σ:.3f}', color='r')
    axes[1,0].set_title('x offset input<->measured')
    axes[1,0].legend()
    axes[1,0].set_xlabel('[pixel]')

    axes[1,1].hist(matched_table['y_fit']-matched_table['y_orig'], bins=60)
    σ = np.nanstd(matched_table['y_fit'] - matched_table['y_orig'])
    axes[1,1].axvline(σ, label=f'σ={σ:.3f}', color='r')
    axes[1,1].set_title('y offset input<->measured')
    axes[1,1].legend()
    axes[1,1].set_xlabel('[pixel]')

    if output_path:
        save(output_path, plt.gcf())
    return plt.gcf()






