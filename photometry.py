from typing import Tuple, Union

import numpy as np
import photutils
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata import NDData
from astropy.stats import sigma_clipped_stats, gaussian_sigma_to_fwhm
from astropy.table import Table
from image_registration.fft_tools import upsample_image
from photutils import EPSFBuilder
from photutils.background import MMMBackground, MADStdBackgroundRMS
from photutils.detection import IRAFStarFinder, DAOStarFinder
from photutils.psf import BasicPSFPhotometry, extract_stars, DAOGroup, IntegratedGaussianPRF,\
    IterativelySubtractedPSFPhotometry

import astropy

from config import Config

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


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
    half = box_size / 2
    x = peak_table['x']
    y = peak_table['y']
    mask = ((x > half) & (x < (image_size - half)) & (y > half) & (y < (image_size - half)))
    return peak_table[mask]


def FWHM_estimate(psf: photutils.psf.EPSFModel) -> float:
    """
    Use a 2D symmetric gaussian fit to estimate the FWHM of a empirical psf
    :param model: EPSFModel instance that was derived
    :return: FWHM in pixel coordinates, takes into account oversampling parameter of EPSF
    """
    from astropy.modeling import fitting
    from astropy.modeling.functional_models import Gaussian2D

    # Not sure if this would work for non-quadratic images
    assert (psf.psfmodel.data.shape[0] == psf.psfmodel.data.shape[1])
    assert (psf.psfmodel.oversampling[0] == psf.psfmodel.oversampling[1])
    dim = psf.psfmodel.data.shape[0]
    center = int(dim / 2)
    gauss_in = Gaussian2D(x_mean=center, y_mean=center, x_stddev=5, y_stddev=5)

    # force a symmetric gaussian
    gauss_in.y_stddev.tied = lambda model: model.x_stddev

    x, y = np.mgrid[:dim, :dim]
    gauss_out = fitting.LevMarLSQFitter()(gauss_in, x, y, psf.psfmodel.data)

    # have to divide by oversampling to get back to original scale
    return gauss_out.x_fwhm / psf.psfmodel.oversampling[0]


def make_stars_guess(image: np.ndarray,
                     threshold_factor: float = Config.threshold_factor,
                     clip_sigma: float = Config.clip_sigma,
                     fwhm_guess: float = Config.fwhm_guess,
                     cutout_size: int = Config.cutout_size) -> photutils.psf.EPSFStars:

    mean, median, std = sigma_clipped_stats(image, sigma=clip_sigma)
    threshold = median + (threshold_factor * std)

    # The idea here is to run a "greedy" starfinder that finds a lot more candidates than we need and then
    # to filter out the bright and isolated stars
    peaks_tbl = DAOStarFinder(threshold, fwhm_guess)(image)
    peaks_tbl.rename_columns(['xcentroid', 'ycentroid'], ['x', 'y'])

    peaks_tbl = cut_edges(peaks_tbl, cutout_size, image.shape[0])
    # TODO this gets medianed away with the image combine approach, so more star more good?
    # stars_tbl = cut_close_stars(peaks_tbl, cutoff_dist=3)
    stars_tbl = peaks_tbl

    image_no_background = image - median
    stars = extract_stars(NDData(image_no_background), stars_tbl, size=cutout_size)
    return stars


def make_epsf_combine(stars: photutils.psf.EPSFStars, oversampling: int = Config.oversampling) -> photutils.psf.EPSFModel:
    # TODO to make this more useful
    #  - maybe normalize image before combination? Now median just picks typical
    #    value so we're restricted to most common stars
    #  - add iterations where the star positions are re-determined with the epsf and

    avg_center = np.sum([np.array(st.cutout_center) for st in stars], axis=0) / len(stars)

    # upsample_image should scale and shift/resample an image with a FFT, aligning the cutouts more precisely
    combined = np.median([upsample_image(star.data, upsample_factor=oversampling,
                                                xshift=star.cutout_center[0] - avg_center[0],
                                                yshift=star.cutout_center[1] - avg_center[1]
                                                ).real
                                 for star in stars], axis=0)

    origin = np.array(combined.shape)/2/oversampling
    # TODO What we return here needs to actually use the image in it's __call__ operator to work as a model
    # type: ignore
    return photutils.psf.EPSFModel(combined, flux=None,
                                   origin=origin,
                                   oversampling=oversampling,
                                   normalize=False)



def make_epsf_fit(stars: photutils.psf.EPSFStars,
                  iters: int = Config.epsfbuilder_iters,
                  oversampling: int = Config.oversampling,
                  smoothing_kernel: Union[str, np.ndarray] = 'quartic') -> photutils.psf.EPSFModel:
    # TODO
    #  how does the psf class interpolate/smooth the data internaly? Jay+Anderson2016 says to use a x^4 polynomial
    #  Also evaluating epsf(x,y) shows the high order noise, but if you only evaluate at the sample points it
    #  does look halfway decent so maybe the fit just creates garbage where it's not constrained by smoothing

    epsf, fitted_stars = EPSFBuilder(oversampling=oversampling,
                                     maxiters=iters,
                                     progress_bar=True,
                                     smoothing_kernel=smoothing_kernel)(stars)
    return epsf


def do_photometry_epsf(image: np.ndarray,
                       epsf: photutils.psf.EPSFModel,
                       threshold_factor: float = Config.threshold_factor,
                       separation_factor: float = Config.separation_factor,
                       clip_sigma: float = Config.clip_sigma,
                       photometry_iterations: int = Config.photometry_iterations) -> Table:

    epsf = photutils.psf.prepare_psf_model(epsf, renormalize_psf=False)  # renormalize is super slow...
    # TODO
    #  Okay, somehow this seems to be the issue: CompoundModel._map_parameters somehow gets screwed up by the way
    #  prepare_psf_model combines models into a tree and you get wrong parameter names (offset_0_1 -> offset_4)
    #  For some reason the call to _map_parameters really messes up the debugger when you try to step in.
    #  Figure out if we can maybe add the missing Parameters ourselves somehow. But working with these models seems
    #  unpleasant as far as just adding parameters
    #  This issue is only triggered if you get multiple stars per group as then the compound of two star models is
    #  constructed

    background_rms = MADStdBackgroundRMS()

    _, img_median, img_stddev = sigma_clipped_stats(image, sigma=clip_sigma)
    threshold = img_median + (threshold_factor * img_stddev)
    fwhm_guess = FWHM_estimate(epsf)
    star_finder = DAOStarFinder(threshold, fwhm_guess)

    grouper = DAOGroup(separation_factor*fwhm_guess)

    shape = (epsf.psfmodel.shape/epsf.psfmodel.oversampling).astype(np.int64)

    epsf.fwhm = astropy.modeling.Parameter('fwhm', 'this is not the way to add this I think')
    epsf.fwhm.value = fwhm_guess
    photometry = IterativelySubtractedPSFPhotometry(
        finder=star_finder,
        group_maker=grouper,
        bkg_estimator=background_rms,
        psf_model=epsf,
        fitter=LevMarLSQFitter(),
        niters=3,
        fitshape=shape
    )

    return photometry(image)

