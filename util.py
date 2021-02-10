import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.table import Table
from typing import Tuple


def gauss(x, a, x0, σ):
    """just the formula"""
    return a * np.exp(-(x - x0) ** 2 / (2 * σ ** 2))


def make_gauss_kernel(σ=1.0):
    """create a 5x5 gaussian convolution kernel"""
    x, y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
    d = np.sqrt(x * x + y * y)

    gauss_kernel = gauss(d, 1., 0.0, σ)
    return gauss_kernel/np.sum(gauss_kernel)


@np.vectorize
def flux_to_magnitude(flux):
    return -2.5 * np.log10(flux)


def fit_gaussian_to_psf(psf: np.ndarray, plot=False) -> np.ndarray:
    """Take a (centered) psf image and attempt to fit a gaussian through
    :return: tuple of a, x0, σ
    """
    # TODO does it really make sense to fit only a 1-d slice, not the whole image?

    assert (psf.ndim == 2)
    assert (psf.shape[0] == psf.shape[1])

    size = psf.shape[0]
    # get horizontal rows of pixels through center
    psf_slice_h = psf[size // 2, :]
    # psf_slice_w = psf[:, size//2]

    x_vals = np.arange(-size // 2, size // 2, 1)

    popt, pcov = curve_fit(gauss, x_vals, psf_slice_h)

    if plot:
        plt.figure()
        plt.plot(x_vals, psf_slice_h, 'o')
        x_dense = np.linspace(-size // 2, size // 2, size * 10)
        plt.plot(x_dense, gauss(x_dense, *popt))

    return popt


def write_ds9_regionfile(x_y_data: np.ndarray, filename: str) -> None:
    """
    Create a DS9 region file from a list of coordinates
    :param x_y_data: set of x-y coordinate pairs
    :param filename: where to write to
    :return:
    """
    assert (x_y_data.ndim == 2)
    assert (x_y_data.shape[1] == 2)

    with open(filename, 'w') as f:
        f.write("# Region file format: DS9 version 3.0\n")
        f.write(
            "global color=blue font=\"helvetica 10 normal\" select=1 edit=1 move=1 delete=1 include=1 fixed=0 source\n")
        for row in x_y_data:
            # +1 for ds9 one-based indexing...
            f.write(f"image;circle( {row[0] + 1:f}, {row[1] + 1:f}, 1.5)\n")
        f.close()


def match_observation_to_source(reference_catalog: Table, photometry_result: Table) \
        -> Table:
    """
    Match the closest points in a photometry catalogue to the input catalogue
    :param reference_catalog: Table containing input positions in 'x' and 'y' columns
    :param photometry_result: Table containing measured positions in 'x_fit' and 'y_fit' columns
    :return: photometry_result updated with 'x_orig', 'y_orig' and 'offset' (euclidean distance) columns
    """
    from scipy.spatial import cKDTree

    x_y_pixel = np.array((reference_catalog['x'], reference_catalog['y'])).T
    lookup_tree = cKDTree(x_y_pixel)

    photometry_result = photometry_result.copy()
    photometry_result['x_orig'] = np.nan
    photometry_result['y_orig'] = np.nan
    photometry_result['offset'] = np.nan

    seen_indices = set()
    for row in photometry_result:

        dist, index = lookup_tree.query((row['x_fit'], row['y_fit']))
        # if index in seen_indices:
        #     print('Warning: multiple match for source')  # TODO make this message more useful/use warning module
        seen_indices.add(index)
        row['x_orig'] = x_y_pixel[index, 0]
        row['y_orig'] = x_y_pixel[index, 1]
        row['offset'] = dist

    return photometry_result


def center_cutout(image: np.ndarray, cutout_size: Tuple[int, int]):
    """cutout a cutout_size[0] x cutout_size[1] section from the center of an image"""
    shape = image.shape
    xstart = int(shape[0]/2 - cutout_size[0]/2)
    xend = int(shape[0]/2 + cutout_size[0]/2)

    ystart = int(shape[0]/2 - cutout_size[0]/2)
    yend = int(shape[0]/2 + cutout_size[0]/2)

    return image[xstart:xend, ystart:yend]


def center_cutout_shift_1(image: np.ndarray, cutout_size: Tuple[int, int]):
    """cutout a cutout_size[0] x cutout_size[1] section from the center of an image"""
    shape = image.shape
    xstart = int(shape[0]/2 - cutout_size[0]/2) + 1
    xend = int(shape[0]/2 + cutout_size[0]/2) + 1

    ystart = int(shape[0]/2 - cutout_size[0]/2) + 1
    yend = int(shape[0]/2 + cutout_size[0]/2) + 1

    return image[xstart:xend, ystart:yend]