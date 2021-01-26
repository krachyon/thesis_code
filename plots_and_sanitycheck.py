import numpy as np
import photutils
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import Optional
from config import Config
from util import match_observation_to_source


def concat_star_images(stars: photutils.psf.EPSFStars) -> np.ndarray:
    assert len(set(star.shape for star in stars)) == 1  # all stars need same shape
    N = int(np.ceil(np.sqrt(len(stars))))
    shape = stars[0].shape
    out = np.zeros(np.array(shape, dtype=int)*N)

    from itertools import product

    for row, col in product(range(N), range(N)):
        if (row+N*col) >= len(stars):
            continue
        xstart = row*shape[0]
        ystart = col*shape[1]

        xend = xstart + shape[0]
        yend = ystart + shape[1]
        i = row+N*col
        out[xstart:xend, ystart:yend] = stars[i].data
    return out


def plot_image_with_source_and_measured(image: np.ndarray, input_table: Table, result_table: Table,
                                        output_path: Optional[str] = None):
    plt.figure()
    plt.imshow(image, norm=LogNorm())


    plt.plot(input_table['x'], input_table['y'], 'o', fillstyle='none',
             markeredgewidth=0.5, markeredgecolor='red', label='reference')
    plt.plot(result_table['x_fit'], result_table['y_fit'], '.', markersize=1,
             markeredgecolor='orange', label='photometry')
    plt.legend()

    if output_path:
        plt.savefig(output_path, dpi=300)


def plot_input_vs_photometry_positions(input_table: Table, result_table: Table,
                                       output_path: Optional[str] = None):

    plt.figure()

    offsets = match_observation_to_source(input_table, result_table)

    plt.title('offset between measured and input')
    plt.xlabel('x offsets')
    plt.ylabel('y offsets')
    plt.plot(offsets['x_fit']-offsets['x_orig'], offsets['y_fit']-offsets['y_orig'], '.'
             , markersize=2, markeredgecolor='orange')

    if output_path:
        plt.savefig(output_path, dpi=300)
