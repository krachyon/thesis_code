import os.path as p
import pickle
import warnings
import zstandard

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.colors import LogNorm

import photutils
from thesis_lib.util import flux_to_magnitude, concat_star_images
from . import wrapper
from .types import RESULT_TABLE_NAMES, INPUT_TABLE_NAMES, \
    X, Y, FLUX, \
    ResultTable


def save(filename_base: str, figure: matplotlib.pyplot.figure):
    """Save a matplotlib figure as png and a pickle it to a mplf file"""
    figure.savefig(filename_base + '.png')
    with zstandard.open(filename_base + '.mplf', 'wb') as outfile:
        pickle.dump(figure, outfile)


def plot_image_with_source_and_measured(image: np.ndarray,
                                        input_table: Table,
                                        result_table: Table) -> matplotlib.pyplot.figure:
    """
    Plot catalogue and photometry positions on an image
    :param image:
    :param input_table: input catalogue for the image (columns 'x', 'y')
    :param result_table: output from photometry ('x_fit', 'y_fit')
    :param output_path: optionally save the figure to this base filename
    :return:
    """
    fig = plt.figure()
    plt.imshow(image, norm=LogNorm())

    plt.plot(input_table[INPUT_TABLE_NAMES[X]], input_table[INPUT_TABLE_NAMES[Y]], 'o', fillstyle='none',
             markeredgewidth=0.5, markeredgecolor='red', label=f'reference N={len(input_table)}')
    plt.plot(result_table[RESULT_TABLE_NAMES[X]], result_table[RESULT_TABLE_NAMES[Y]], '.', markersize=1,
             markeredgecolor='orange', label=f'photometry N={len(result_table)}')
    plt.legend()

    return fig


def plot_xy_deviation(matched_table: ResultTable) -> matplotlib.pyplot.figure:
    """
    Plot the x and y offsets between two catalogues as scatterplot
    :param matched_table: Table with columns 'x_fit', 'y_fit', 'x_orig', 'y_orig', 'offset'
    :param output_path: optionally save plot to this base filename
    :return:
    """

    fig = plt.figure()

    plt.title('offset between measured and input position')
    plt.xlabel('x offsets [pixel]')
    plt.ylabel('y offsets [pixel]')
    xs = matched_table['x_offset']
    ys = matched_table['y_offset']
    plt.plot(xs, ys, '.', markersize=2, markeredgecolor='orange')
    text = f'$σ_x={np.std(xs):.3f}$\n$σ_y={np.std(ys):.3f}$'
    plt.annotate(text, xy=(0.01, 0.9), xycoords='axes fraction')

    return fig


def plot_deviation_vs_magnitude(matched_table: Table) -> matplotlib.pyplot.figure:
    """
    Plot the absolute deviation between input and photometry as a function of magnitude
    :param matched_table: Table with columns 'x_fit', 'y_fit', 'x_orig', 'y_orig', 'offset'
    :param output_path: optionally save plot to this base filename
    :return:
    """

    # TODO make names refer to types.
    # TODO refactor out calculations

    fig = plt.figure()

    dists = matched_table['offset']

    magnitudes = flux_to_magnitude(matched_table[RESULT_TABLE_NAMES[FLUX]])

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

    return fig


def plot_deviation_histograms(matched_table: Table) -> matplotlib.pyplot.figure:

    # TODO make names refer to types.
    # TODO refactor out calculations

    fig, axes = plt.subplots(2, 2, sharex='all')

    axes[0,0].hist(matched_table['offset'], bins=60)
    σ = np.nanstd(matched_table['offset'])
    axes[0,0].axvline(σ, label=f'σ={σ:.3f}', color='r')
    axes[0,0].set_title('distance offset input<->measured')
    axes[0,0].legend()
    axes[0,0].set_xlabel('[pixel]')


    magnitude_diff = np.array(flux_to_magnitude(matched_table['flux_0']) - flux_to_magnitude(matched_table['flux_fit']))
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

    return fig


def plot_epsf(epsf: photutils.EPSFModel) -> matplotlib.pyplot.figure:
    fig = plt.figure()
    plt.imshow(epsf.data)
    return fig


def plot_epsfstars(epsfstars: photutils.EPSFStars) -> matplotlib.pyplot.figure:
    fig = plt.figure()
    img = concat_star_images(epsfstars)
    plt.title('epfs stars')
    plt.imshow(img)
    return fig


def make_all_plots(astrometry_session: wrapper.Session, save_files: bool = False) -> None:
    """"""

    tables = astrometry_session.tables

    source_measured = plot_image_with_source_and_measured(astrometry_session.image,
                                                          tables.input_table,
                                                          tables.valid_result_table)
    xy_dev = plot_xy_deviation(tables.valid_result_table)

    dev_v_mag = plot_deviation_vs_magnitude(tables.valid_result_table)

    deviation_histograms = plot_deviation_histograms(tables.valid_result_table)
    epsf = plot_epsf(astrometry_session.epsf)
    epsf_stars = plot_epsfstars(astrometry_session.epsfstars)

    if save_files:
        outdir = astrometry_session.config.output_folder

        image_name = astrometry_session.image_name
        if image_name == 'unnamed':
            warnings.warn('creating output for unnamed image. May overwrite previous plots of unnamed')
        outname = p.join(outdir, image_name)

        save(outname+'_source_measured', source_measured)
        save(outname+'_xy_dev', xy_dev)
        save(outname+'_deviation_vs_magnitude', dev_v_mag)
        save(outname+'_deviation_histograms', deviation_histograms)
        save(outname+'_epsf', epsf)
        save(outname+'_epsfstars', epsf_stars)







