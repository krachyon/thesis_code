from pylab import *
import photutils as phot
from astropy.io import fits
import numpy as np
from astropy.table import vstack, Table
import itertools
from itertools import repeat
from astropy.stats import sigma_clipped_stats
import multiprocessing as mp
from astropy.nddata import NDData
from dataclasses import dataclass
from typing import List, Any
from pathlib import Path
import re

# Configuration
known_psf_oversampling = 4  # for precomputed psf
known_psf_size = 201  # dito

# photometry
photometry_iters = 8
fitshape = 15

# epsf_fit
cutout_size = 51
smoothing_kernel = 'quadratic'
oversampling = 4
epsf_iters = 5

# grouper
group_radius = 1.5
# starfinder

threshold_factor = 0.01  # img_median*this
fwhm_factor = 1.  # fwhm*this
n_brightest = 50  # use only n stars
minsep_fwhm = 0.3  # only find stars at least this*fwhm apart
peakmax = 10_000  #only find stars below this pixel value


class DebugPool:
    """
    Limited fake of multiprocess.Pool that executes sequentially and synchronously to allow debugging
    """

    @dataclass
    class Future:
        results: List[Any]

        def get(self,*args):
            return self.results


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

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

def estimate_fwhm(psf: phot.psf.EPSFModel) -> float:
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
    dim = psf.data.shape[0]
    center = int(dim / 2)
    gauss_in = Gaussian2D(x_mean=center, y_mean=center, x_stddev=5, y_stddev=5)

    # force a symmetric gaussian
    gauss_in.y_stddev.tied = lambda model: model.x_stddev

    x, y = np.mgrid[:dim, :dim]
    gauss_out = fitting.LevMarLSQFitter()(gauss_in, x, y, psf.data)

    # have to divide by oversampling to get back to original scale
    return gauss_out.x_fwhm / psf.oversampling[0]


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


psf_offsets = [[0, 0], [200, 0], [400, 0],
               [0, 200], [200, 200], [400, 200],
               [0, 400], [200, 400], [400, 400]]


def get_psf_subframe(data: np.ndarray):
    subframes = [data[:201, :201],
                 data[200:401, :201],
                 data[400:, :201],

                 data[:201, 200:401],
                 data[200:401, 200:401],
                 data[400:, 200:401],

                 data[:201, 400:],
                 data[200:401, 400:],
                 data[400:, 400:]]

    return subframes


def get_img_subframes(data: np.ndarray, segments=3):
    ys = np.linspace(0, data.shape[0]-1, segments+1)
    xs = np.linspace(0, data.shape[1]-1, segments+1)
    subframes = []
    offsets = []
    for (x_start, x_end), (y_start, y_end) in itertools.product(pairwise(xs), pairwise(ys)):

        subframes.append(
            data[int(np.floor(y_start)):int(np.ceil(y_end)),
                 int(np.floor(x_start)):int(np.ceil(x_end))]
        )
        offsets.append([int(np.floor(x_start)), int(np.floor(y_start))])
    return subframes, offsets


def psf_from_image(image: np.ndarray):
    offset = int(known_psf_size / 2)
    origin = [offset / known_psf_oversampling, offset / known_psf_oversampling]
    model = phot.psf.EPSFModel(image, flux=1, oversampling=known_psf_oversampling, origin=origin)
    return phot.prepare_psf_model(model, renormalize_psf=False)


def naco_astrometry(image, input_psf, offset, reference_table, use_reference=False, use_psf=False):

    fwhm = estimate_fwhm(input_psf.psfmodel)
    mean, median, std = sigma_clipped_stats(image)
    # todo make configurable
    #  values here are handfudged to get maximum amount of candidates
    #  ommitted peakmax = 10_000

    finder = phot.IRAFStarFinder(threshold=median*threshold_factor,
                                 fwhm=fwhm*fwhm_factor,
                                 brightest=n_brightest,
                                 minsep_fwhm=minsep_fwhm,
                                 peakmax=peakmax)

    if np.all(np.isnan(image)):
        return Table()

    if use_reference:
        stars_tbl = reference_table.copy()
        stars_tbl.rename_columns(['XRAW', 'YRAW'], ['x', 'y'])
        stars_tbl['x'] -= offset[0]
        stars_tbl['y'] -= offset[1]
        cut_x = (stars_tbl['x'] >= 0) & (stars_tbl['x'] <= image.shape[1])
        cut_y = (stars_tbl['y'] >= 0) & (stars_tbl['y'] <= image.shape[0])
        stars_tbl = stars_tbl[cut_x & cut_y]
    else:
        stars_tbl = finder(image)
        stars_tbl.rename_columns(['xcentroid', 'ycentroid'], ['x', 'y'])

    if use_psf:
        psf = input_psf
    else:
        image_no_background = image - median
        stars = phot.extract_stars(NDData(image_no_background), stars_tbl, size=cutout_size)
        epsf, fitted_stars = phot.EPSFBuilder(oversampling=oversampling,
                                         maxiters=epsf_iters,
                                         progress_bar=True,
                                         smoothing_kernel=smoothing_kernel).build_epsf(stars)
        psf = phot.prepare_psf_model(epsf, renormalize_psf=False)


    grouper = phot.DAOGroup(group_radius*fwhm)

    if use_reference:
        photometry = phot.BasicPSFPhotometry(
            group_maker=grouper,
            finder=finder,
            bkg_estimator=phot.MMMBackground(),
            aperture_radius=fwhm,
            fitshape=fitshape,
            psf_model=psf)
        stars_tbl.rename_columns(['x', 'y'], ['x_0', 'y_0'])
        size = len(stars_tbl)
        stars_tbl['x_0'] += np.random.uniform(0.1,  0.2, size) * np.random.choice([-1, 1], size)
        stars_tbl['y_0'] += np.random.uniform(0.1,  0.2, size) * np.random.choice([-1, 1], size)
        result = photometry(image, init_guesses=stars_tbl)
    else:
        photometry = phot.IterativelySubtractedPSFPhotometry(
            group_maker=grouper,
            finder=finder,
            bkg_estimator=phot.MMMBackground(),
            aperture_radius=fwhm,
            fitshape=fitshape,
            psf_model=psf,
            niters=photometry_iters
        )

        result = photometry(image)

    result['x_fit'] += offset[0]
    result['y_fit'] += offset[1]

    return result


def astrometry_wrapper(image_name: str, psf_name: str, reference_table_name: str):

    image_data = fits.getdata(image_name).astype(np.float64)
    image_data[image_data < 0] = np.nan
    mask = np.zeros(image_data.shape, dtype=bool)
    mask[0:512, 0:512] = 1
    image_data[mask] = np.nan

    image_subframes, offsets = get_img_subframes(image_data)
    psf_data   = fits.getdata(psf_name).astype(np.float64)
    psf_subframes = get_psf_subframe(psf_data)
    psf_models = [psf_from_image(p) for p in psf_subframes]

    ref = Table.read('../test_images_naco/NACO.2018-08-12T00:10:49.488_NGC6441_P13_flt.subtr15.clean.xym',
                           format='ascii')
    # make 0-indexed
    ref['XRAW'] -= 1
    ref['YRAW'] -= 1

    # with DebugPool() as p:
    with mp.Pool() as p:
        ref_psf = p.starmap_async(
            naco_astrometry, zip(image_subframes, psf_models, offsets, repeat(ref), repeat(True), repeat(True)))
        ref_nopsf = p.starmap_async(
            naco_astrometry, zip(image_subframes, psf_models, offsets, repeat(ref), repeat(True), repeat(False)))
        noref_psf = p.starmap_async(
            naco_astrometry, zip(image_subframes, psf_models, offsets, repeat(ref), repeat(False), repeat(True)))
        noref_nopsf = p.starmap_async(
            naco_astrometry, zip(image_subframes, psf_models, offsets, repeat(ref), repeat(False), repeat(False)))

        results = {'ref_psf': vstack(ref_psf.get()),
                   'ref_nopsf': vstack(ref_nopsf.get()),
                   'noref_psf': vstack(noref_psf.get()),
                   'noref_nopsf': vstack(noref_nopsf.get())
                   }
        return image_data, psf_data, ref, results


if __name__ == '__main__':
    img_folder = '../test_images_naco'
    basenames = ['NACO.2018-08-12T00:10:49.488_NGC6441_P13_flt.subtr15',
                 'NACO.2018-08-12T00:10:49.488_NGC6441_P13_flt.subtr15']

    for basename in basenames:
        psf_name = Path(img_folder)/('PSF.'+basename+'.clean.fits')
        img_name = Path(img_folder)/(basename+'.fits')
        ref_max_name = Path(img_folder)/(basename+'.clean.xym')
        ref_dav_name = Path(img_folder)/(basename+'_davide.coo')

        img, psf, ref_max, results = astrometry_wrapper(img_name, psf_name, ref_max_name)
        ref_dav = read_dp_coo(ref_dav_name)


        for name, res in results.items():
            figure()
            imshow(img, cmap='hot')
            plot(ref_max['XRAW'], ref_max['YRAW'], 'go', markersize=3, alpha=1, label='max analysis')
            plot(ref_dav['x'], ref_dav['y'], 'rx', markersize=2, label='davide analysis')
            plot(res['x_fit'], res['y_fit'], 'b+', markersize=2, alpha=1, label='photutils')
            title(name)
            legend()
        show()


