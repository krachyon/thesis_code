from typing import Tuple

import numpy as np

import photutils
import astropy

from photutils.detection import IRAFStarFinder, find_peaks, DAOStarFinder
from photutils.background import MMMBackground, MADStdBackgroundRMS
from photutils.psf import BasicPSFPhotometry, extract_stars, DAOGroup, IntegratedGaussianPRF
from photutils import EPSFBuilder

from astropy.stats import SigmaClip, sigma_clipped_stats, gaussian_sigma_to_fwhm
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata import NDData
from astropy.table import Table

from scipy.spatial import cKDTree



def do_photometry_basic(image: np.ndarray, σ_psf: float) -> Tuple[Table, np.ndarray]:
    """
    Find stars in an image

    :param image: The image data you want to find stars in
    :param σ_psf: expected deviation of PSF
    :return: tuple result table, residual image
    """
    bkgrms = MADStdBackgroundRMS()

    std = bkgrms(image)

    iraffind = IRAFStarFinder(threshold=3 * std, sigma_radius=σ_psf,
                              fwhm=σ_psf * gaussian_sigma_to_fwhm,
                              minsep_fwhm=2, roundhi=5.0, roundlo=-5.0,
                              sharplo=0.0, sharphi=2.0)
    daogroup = DAOGroup(0.1 * σ_psf * gaussian_sigma_to_fwhm)

    mmm_bkg = MMMBackground()

    # my_psf = AiryDisk2D(x_0=0., y_0=0.,radius=airy_minimum)
    # psf_model = prepare_psf_model(my_psf, xname='x_0', yname='y_0', fluxname='amplitude',renormalize_psf=False)
    psf_model = IntegratedGaussianPRF(sigma=σ_psf)
    # psf_model = AiryDisk2D(radius = airy_minimum)#prepare_psf_model(AiryDisk2D,xname ="x_0",yname="y_0")
    # psf_model = Moffat2D([amplitude, x_0, y_0, gamma, alpha])

    # photometry = IterativelySubtractedPSFPhotometry(finder=iraffind, group_maker=daogroup,
    #                                                bkg_estimator=mmm_bkg, psf_model=psf_model,
    #                                                fitter=LevMarLSQFitter(),
    #                                                niters=2, fitshape=(11,11))
    photometry = BasicPSFPhotometry(finder=iraffind, group_maker=daogroup,
                                    bkg_estimator=mmm_bkg, psf_model=psf_model,
                                    fitter=LevMarLSQFitter(), aperture_radius=11.0,
                                    fitshape=(11, 11))

    result_table = photometry.do_photometry(image)
    return result_table, photometry.get_residual_image()


# TODO how to
#  - find the best star candidates that are isolated for the PSF estimation?
#  - Guess the FWHM for the starfinder that is used in the Photometry pipeline? Do we need that if we have a custom PSF?

def cut_edges(peak_table: Table, box_size: int, image_size: int) -> Table:
    half = box_size/2
    x = peak_table['x']
    y = peak_table['y']
    mask = ((x > half) & (x < (image_size - half)) & (y > half) & (y < (image_size - half)))
    return peak_table[mask]


def cut_close_stars(peak_table: Table, cutoff_dist: float) -> Table:
    peak_table['nearest'] = 0.
    x_y = np.array((peak_table['x'], peak_table['y'])).T
    lookup_tree = cKDTree(x_y)
    for row in peak_table:
        # find the second nearest neighbour, first one will be the star itself...
        dist, _ = lookup_tree.query((row['x'], row['y']), k=[2])
        row['nearest'] = dist[0]

    peak_table = peak_table[peak_table['nearest'] > cutoff_dist]
    return peak_table



def FWHM_estimate(psf: photutils.psf.EPSFModel) -> float:
    """
    Use a 2D symmetric gaussian fit to estimate the FWHM of a empirical psf
    :param model: EPSFModel instance that was derived
    :return: FWHM in pixel coordinates, takes into account oversampling parameter of EPSF
    """
    from astropy.modeling import models, fitting
    from astropy.modeling.functional_models import Gaussian2D

    # Not sure if this would work for non-quadratic images
    assert(psf.data.shape[0] == psf.data.shape[1])
    assert(psf.oversampling[0] == psf.oversampling[1])
    dim = psf.data.shape[0]
    center = int(dim/2)
    gauss_in = Gaussian2D(x_mean=center, y_mean=center, x_stddev=5, y_stddev=5)

    # force a symmetric gaussian
    gauss_in.y_stddev.tied = lambda model: model.x_stddev

    x, y = np.mgrid[:dim, :dim]
    gauss_out = fitting.LevMarLSQFitter()(gauss_in, x, y, psf.data)

    # have to divide by oversampling to get back to original scale
    return gauss_out.x_fwhm/psf.oversampling[0]


def epsf_just_combine_candidates(stars):
    # quick and dirty:
    import functools
    combined = functools.reduce(lambda x, y: x + y, (star.data for star in stars))

    # with offset/

    from image_registration.fft_tools import upsample_image
    avg_center = functools.reduce(lambda x, y: x+y, [np.array(st.cutout_center) for st in stars])/len(stars)

    combined_better = functools.reduce(lambda x,y: x+y,
    (upsample_image(star.data, upsample_factor=4,
                    xshift=star.cutout_center[0]-avg_center[0],
                    yshift=star.cutout_center[1]-avg_center[1]
                    )
     for star in stars))

    return combined_better


def make_epsf(image: np.ndarray) -> photutils.psf.EPSFModel:

    ###
    # magic parameters
    clip_sigma = 3.0
    threshold_factor = 3.
    box_size = 10
    cutout_size = 50  # TODO PSF is pretty huge, right?
    oversampling = 4
    epsfbuilder_iters = 5
    fwhm_guess = 2.5
    ###
    # background_rms = MADStdBackgroundRMS(sigma_clip=SigmaClip(3))(image)
    mean, median, std = sigma_clipped_stats(image, sigma=clip_sigma)
    threshold = median + (threshold_factor * std)

    # The idea here is to run a "greedy" starfinder that finds a lot more candidates than we need and then
    # to filter out the bright and isolated stars
    peaks_tbl = DAOStarFinder(threshold, fwhm_guess)(image)
    peaks_tbl.rename_columns(['xcentroid', 'ycentroid'], ['x', 'y'])

    peaks_tbl = cut_edges(peaks_tbl, cutout_size, image.shape[0])
    #stars_tbl = cut_close_stars(peaks_tbl, cutoff_dist=3)
    stars_tbl = peaks_tbl

    image_no_background = image - median
    stars = extract_stars(NDData(image_no_background), stars_tbl, size=cutout_size)

    epsf, fitted_stars = EPSFBuilder(oversampling=oversampling, maxiters=epsfbuilder_iters, progress_bar=True)(stars)
    return epsf, stars


def do_photometry_epsf(image: np.ndarray):
    epsf = make_epsf(image)

    bkgrms = MADStdBackgroundRMS()
    std = bkgrms(image)
    # iraffind = IRAFStarFinder(threshold=3.5 * std,
    # fwhm = sigma_psf * gaussian_sigma_to_fwhm,
    # minsep_fwhm = 0.01, roundhi = 5.0, roundlo = -5.0, sharplo = 0.0, sharphi = 2.0)


if __name__ == '__main__':
    from astropy.io import fits
    img = fits.open('output_files/observed_00.fits')[0].data
    epsf, stars = make_epsf(img)


