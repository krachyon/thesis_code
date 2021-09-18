import numpy as np
import matplotlib.pyplot as plt
import pathlib

import photutils
from scipy.optimize import curve_fit
from typing import Tuple, Union
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from photutils.utils.errors import calc_total_error
from scipy.interpolate import RectBivariateSpline
from photutils import CircularAperture
from typing import List, Any
from dataclasses import dataclass
import re
import astropy.io.fits
import pickle
import os
import contextlib



class ClassRepr(type):
    """
    Use this as a metaclass to make a class (and not just an instance of it) print its contents.
    Kinda hacky and doesn't really consider edge-cases
    """

    def __new__(mcs, *args, **kwargs):
        return super().__new__(mcs, *args, **kwargs)

    def __repr__(cls):
        items = [item for item in cls.__dict__.items() if not item[0].startswith('__')]
        item_string = ', '.join([f'{item[0]} = {item[1]}' for item in items])
        return f'{cls.__name__}({item_string})'

    def __str__(cls):
        return repr(cls)


@contextlib.contextmanager
def work_in(path: Union[str, pathlib.Path]):
    """A context manager which changes the working directory to the given
    path, and then changes it back to its previous value on exit.
    LICENSE: MIT
    from: https://code.activestate.com/recipes/576620-changedirectory-context-manager/
    """
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def gauss(x, a, x0, σ):
    """just the formula"""
    return a * np.exp(-(x - x0) ** 2 / (2 * σ ** 2))


def make_gauss_kernel(σ=1.0):
    """create a 5x5 gaussian convolution kernel"""
    x, y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
    d = np.sqrt(x * x + y * y)

    gauss_kernel = gauss(d, 1., 0.0, σ)
    return gauss_kernel/np.sum(gauss_kernel)


def airy_fwhm(r):
    """FWHM= 1.028λ/d ; θ~=r=1.22 λ/d"""
    fwhm = r * 0.8426229508196722
    return fwhm


@np.vectorize
def flux_to_magnitude(flux):
    return -2.5 * np.log10(flux)


@np.vectorize
def magnitude_to_flux(magnitude):
    return 10**(-magnitude/2.5)


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
    # TODO is this really necessary? there seems to be x_0, y_0 in results,
    #  can use this in cases where the guess is known
    """
    Match the closest points in a photometry catalogue to the input catalogue
    :param reference_catalog: Table containing input positions in 'x' and 'y' columns
    :param photometry_result: Table containing measured positions in 'x_fit' and 'y_fit' columns
    :return: photometry_result updated with 'x_orig', 'y_orig' and 'offset' (euclidean distance) columns
    """
    from scipy.spatial import cKDTree

    x_y_pixel = np.array((reference_catalog['x'], reference_catalog['y'], reference_catalog['m'])).T
    lookup_tree = cKDTree(x_y_pixel[:, :2])  # only feed x and y to the lookup tree

    photometry_result = photometry_result.copy()
    photometry_result['x_orig'] = np.nan
    photometry_result['y_orig'] = np.nan
    photometry_result['m_orig'] = np.nan
    photometry_result['offset'] = np.nan

    seen_indices = set()
    for row in photometry_result:

        dist, index = lookup_tree.query((row['x_fit'], row['y_fit']))
        # if index in seen_indices:
        #     print('Warning: multiple match for source')  # TODO make this message more useful/use warning module
        seen_indices.add(index)
        row['x_orig'] = x_y_pixel[index, 0]
        row['y_orig'] = x_y_pixel[index, 1]
        row['m_orig'] = x_y_pixel[index, 2]

    photometry_result['x_offset'] = photometry_result['x_fit'] - photometry_result['x_orig']
    photometry_result['y_offset'] = photometry_result['y_fit'] - photometry_result['y_orig']
    photometry_result['offset'] = np.sqrt(photometry_result['x_offset']**2 + photometry_result['y_offset']**2)

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


def linspace_grid(start: float, stop: float, num: int):
    """
    Construct a 2D meshgrid analog to np.mgrid, but use linspace syntax instead of indexing
    """
    # complex step: use number of steps instead
    return np.mgrid[start:stop:1j*num, start:stop:1j*num]


def estimate_photometric_precision_peak_only(image: np.ndarray, sources: Table, fwhm: float, effective_gain: float = 1):
    """
    Estimate the possible position precision for stars in an image based on the SNR and the PSF FWHM.
    To calculate the SNR of a star with fractional coordinates, bilinear interpolation is used
    see. Lindegren, Lennart. “Photoelectric Astrometry - A Comparison of Methods for Precise Image Location.”

    :param image: input exposure
    :param sources: table with 'x' and 'y' columns in pixel coordinates
    :param fwhm: Full width at half maximum for the PSF of the image
    :param effective_gain: gain/quantum_efficiency
    :return: list of computed σ_pos same order as sources in table
    """
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    bkg_error = std
    error_img = calc_total_error(image-median, bkg_error, effective_gain)
    snr = (image-median)/error_img

    row_idxs, col_idxs = np.ogrid[0:image.shape[0], 0:image.shape[1]]
    # this should do linear interpolation
    snr_interpolated = RectBivariateSpline(row_idxs, col_idxs, snr, kx=1, ky=1)
    assert np.allclose(snr, snr_interpolated(row_idxs, col_idxs))

    # snr_interpolated is indexed the same way as images/arrays: outer dimension (== y) first
    sigma_pos = np.abs([float(fwhm / snr_interpolated(row['y'], row['x'])) for row in sources])
    return sigma_pos


def estimate_photometric_precision_full(image: np.ndarray, sources: Table, fwhm: float, effective_gain: float = 1):
    """
    Estimate the possible position precision for stars in an image based on the SNR and the PSF FWHM.
    The signal is summed over a circular aperture with radius = fwhm, the error the geometric mean
    within this aperture.
    see. Lindegren, Lennart. “Photoelectric Astrometry - A Comparison of Methods for Precise Image Location.”

    :param image: input exposure
    :param sources: table with 'x' and 'y' columns in pixel coordinates
    :param fwhm: Full width at half maximum for the PSF of the image
    :param effective_gain: gain/quantum_efficiency
    :return: list of computed σ_pos same order as sources in table
    """
    # Idea: use aperture photometry to sum the signal, internally it will perform quadratic mean
    #  of pixel errors, so for each star we have signal and error.
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    bkg_error = std
    error_img = calc_total_error(image, bkg_error, effective_gain)

    xy = np.array((sources['x'], sources['y'])).T
    apertures = CircularAperture(xy, r=fwhm)
    signals, errors = apertures.do_photometry(image-median, error_img)

    # sometimes signal is negative over FWHM, avoid negative sigma
    sigma_pos = np.abs(fwhm/(signals/errors))
    return sigma_pos


class DebugPool:
    """
    Limited fake of multiprocess.Pool that executes sequentially and synchronously to allow debugging
    """

    @dataclass
    class Future:
        results: Any

        def get(self, *args):
            return self.results

        def ready(self):
            return True


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def map(self, function, args):
        return list(map(function, args))

    def imap(self, function, args):
        return map(function, args)

    def imap_unordered(self, function, args):
        return self.imap(function, args)

    def apply_async(self, function, args):
        future = self.Future(function(*args))
        return future

    def starmap_async(self, function, arg_lists):
        future = self.Future([])
        for arg_list in arg_lists:
            future.results.append(function(*arg_list))
        return future


_number_regex = re.compile(r'''[+-]?\d+  # optional sign, mandatory digit(s) 
                              \.?\d*  # possible decimal dot can be followed by digits
                              (?:[eE][+-]?\d+)? # non-capturing group for exponential
                           ''', re.VERBOSE)


def _read_header(daophot_filename: str) -> dict:
    """
    Read the header of a daophot file
    expected format:
    0: name0 name1 name2
    1: num0 num1 num2

    As daophot uses fixed with, numbers are not always separated by spaces
    :param daophot_filename: file containing a daophot table
    :return: dictionary {name0:num0,...}
    """
    # parse header
    with open(daophot_filename, 'r') as f:
        name_line = f.readline().strip()
        number_line = f.readline().strip()

    header_names = re.split(r'\s+', name_line)
    header_values = re.findall(_number_regex, number_line)

    assert len(header_values) == len(header_names)
    return dict(zip(header_names, header_values))


def read_dp_coo(coo_filename: str) -> Table:
    """
    Read the contents of a daophot .coo (FIND) file as an astropy table
    :param coo_filename:
    :return:
    """

    meta = _read_header(coo_filename)
    # read main content
    tab = Table.read(coo_filename, data_start=3, format='ascii',
                     names=['id', 'x', 'y', 'm', 'sharp', 'round', 'dy'])
    tab.meta = meta

    # adapt xy to zero-indexing
    tab['x'] -= 1
    tab['y'] -= 1
    return tab


def read_dp_ap(ap_filename: str) -> Table:
    """
    read contents of a daophot .ap (PHOTOMETRY) file as an astropy table
    :param ap_filename:
    :return:
    """
    meta = _read_header(ap_filename)

    with open(ap_filename, 'r') as f:
        content = f.readlines()[2:]

    datalines = ''.join(content).strip().split('\n\n')
    data = np.array([re.findall(_number_regex, i) for i in datalines], dtype=float)

    # variable column width depending on number of apertures, 6 fixed columns
    n_apertures = (data.shape[1] - 6)/2
    assert (n_apertures - int(n_apertures)) == 0
    n_apertures = int(n_apertures)
    names = ['id', 'x', 'y'] + \
            [f'mag_{i}' for i in range(n_apertures)] +\
            ['sky', 'sky_err', 'sky_skew'] + \
            [f'mag_err_{i}' for i in range(n_apertures)]

    tab = Table(data, names=names)
    tab.meta = meta
    # adapt xy to zero-indexing
    tab['x'] -= 1
    tab['y'] -= 1
    return tab


def getdata_safer(filename, *args, **kwargs):
    """
    Wrapper for astropy.io.fits.getdata to coerce data into float64 with native byteorder.
    FITS are default big-endian which can cause weird stuff to happen as numpy stores it as-is in memory
    :param filename: what to read
    :param args: further args to fits.getdata
    :param kwargs: further keyword-args to fits.getdata
    :return: the read data
    """
    data = astropy.io.fits.getdata(filename, *args, **kwargs)\
        .astype(np.float64, order='C', copy=False)

    assert data.dtype.byteorder == '='
    assert data.flags.c_contiguous

    return data


def save_plot(outdir, name, dpi=250):
    plt.savefig(os.path.join(outdir, name+'.pdf'), dpi=dpi)
    with open(os.path.join(outdir, name+'.mplf'), 'wb') as f:
        pickle.dump(plt.gcf(), f)


def center_of_image(img: np.ndarray) -> tuple[float, float]:
    """in pixel coordinates, pixel center convention

    (snippet to verify the numpy convention)
    img=np.random.randint(1,10,(10,10))
    y,x = np.indices(img.shape)
    imshow(img)
    plot(x.flatten(),y.flatten(),'ro')
    """
    assert len(img.shape) == 2
    ycenter, xcenter = (np.array(img.shape)-1)/2

    return xcenter, ycenter


def estimate_fwhm(psf: photutils.psf.EPSFModel) -> float:
    """
    Use a 2D symmetric gaussian fit to estimate the FWHM of an empirical psf
    :param psf: psfmodel to estimate
    :return: FWHM in pixel coordinates, takes into account oversampling parameter of EPSF
    """
    from astropy.modeling import fitting
    from astropy.modeling.functional_models import Gaussian2D

    # Not sure if this would work for non-quadratic images
    assert (psf.data.shape[0] == psf.data.shape[1])
    assert (psf.oversampling[0] == psf.oversampling[1])

    y, x = np.indices(psf.data.shape)
    xcenter, ycenter = center_of_image(psf.data)

    gauss_in = Gaussian2D(x_mean=xcenter, y_mean=ycenter, x_stddev=5, y_stddev=5)
    # force a symmetric gaussian
    gauss_in.y_stddev.tied = lambda model: model.x_stddev

    gauss_out = fitting.LevMarLSQFitter()(gauss_in, x, y, psf.data)

    # have to divide by oversampling to get back to original scale
    return gauss_out.x_fwhm / psf.oversampling[0]


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