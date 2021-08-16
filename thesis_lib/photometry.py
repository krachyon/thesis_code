from typing import Tuple, Union, Optional
from collections import namedtuple
import numpy as np

import astropy
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata import NDData
from astropy.stats import sigma_clipped_stats, gaussian_sigma_to_fwhm
from astropy.table import Table
from image_registration.fft_tools import upsample_image

import photutils
from photutils import EPSFBuilder
from photutils.background import MMMBackground, MADStdBackgroundRMS
from photutils.detection import IRAFStarFinder, DAOStarFinder
from photutils.psf import BasicPSFPhotometry, extract_stars, DAOGroup, IntegratedGaussianPRF, \
    IterativelySubtractedPSFPhotometry
from photutils.psf import EPSFModel

from .config import Config
from .util import flux_to_magnitude
config = Config.instance()


PhotometryResult = namedtuple('PhotometryResult',
                              ('image', 'input_table', 'result_table', 'epsf', 'star_guesses', 'config', 'image_name'))


def do_photometry_basic(image: np.ndarray, σ_psf: float) -> Tuple[Table, np.ndarray]:
    """
    Find stars in an image with IRAFStarFinder

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

def cut_edges(peak_table: Table, cutout_size: int, image_size: int) -> Table:
    """
    Exclude sources from peak_table which are too close to the borders of the image to fully
    produce a cutout
    :param peak_table: Table containing sources as 'x' and 'y' columns
    :param cutout_size: how big of a cutout we want around every star
    :param image_size:  how big the image is
    :return:
    """
    half = cutout_size / 2
    x = peak_table['x']
    y = peak_table['y']
    mask = ((x > half) & (x < (image_size - half)) & (y > half) & (y < (image_size - half)))
    return peak_table[mask]


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
    dim = psf.data.shape[0]
    center = int(dim / 2)
    gauss_in = Gaussian2D(x_mean=center, y_mean=center, x_stddev=5, y_stddev=5)

    # force a symmetric gaussian
    gauss_in.y_stddev.tied = lambda model: model.x_stddev

    x, y = np.mgrid[:dim, :dim]
    gauss_out = fitting.LevMarLSQFitter()(gauss_in, x, y, psf.data)

    # have to divide by oversampling to get back to original scale
    return gauss_out.x_fwhm / psf.oversampling[0]


def make_stars_guess(image: np.ndarray,
                     star_finder: photutils.StarFinderBase,
                     cutout_size: int = config.cutout_size) -> photutils.psf.EPSFStars:
    """
    Given an image, extract stars as EPSFStars for psf fitting
    :param image: yes
    :param cutout_size: how big should the regions around each star used for fitting be?
    :param star_finder: which starfinder to use?
    :return: instance of exctracted EPSFStars
    """

    # The idea here is to run a "greedy" starfinder that finds a lot more candidates than we need and then
    # to filter out the bright and isolated stars
    peaks_tbl = star_finder(image)
    peaks_tbl.rename_columns(['xcentroid', 'ycentroid'], ['x', 'y'])

    peaks_tbl = cut_edges(peaks_tbl, cutout_size, image.shape[0])
    # TODO this gets medianed away with the image combine approach, so more star more good?
    # stars_tbl = cut_close_stars(peaks_tbl, cutoff_dist=3)
    stars_tbl = peaks_tbl

    image_no_background = image - np.median(image)
    stars = extract_stars(NDData(image_no_background), stars_tbl, size=cutout_size)
    return stars


def get_finder(image:np.ndarray, config: Config)-> photutils.StarFinderBase:
    """construct a StarFinder from a given configuration and image(needed for threshold)"""

    mean, median, std = sigma_clipped_stats(image, sigma=config.clip_sigma)
    threshold = median + config.threshold_factor * std

    finder = DAOStarFinder(threshold=threshold,
                           fwhm=config.fwhm_guess,
                           sigma_radius=config.sigma_radius,
                           sharplo=config.sharplo,
                           sharphi=config.sharphi,
                           roundlo=config.roundlo,
                           roundhi=config.roundhi,
                           exclude_border=config.exclude_border)
    return finder


def make_epsf_combine(stars: photutils.psf.EPSFStars,
                      oversampling: int = config.oversampling) -> photutils.psf.EPSFModel:
    """
    Alternative way of deriving an EPSF. Use median after resampling/scaling to just overlay images
    :param stars: candidate stars as EPSFStars
    :param oversampling: How much to scale
    :return: epsf model
    """
    # TODO to make this more useful
    #  - maybe normalize image before combination? Now median just picks typical
    #    value so we're restricted to most common stars
    #  - add iterations where the star positions are re-determined with the epsf and overlaying happens again
    #  BUG: This will not always yield a centered array for the EPSF

    avg_center = np.mean([np.array(st.cutout_center) for st in stars], axis=0)
    output_shape = np.array(stars[0].data.shape)*oversampling//2*2+1
    scaling = output_shape/np.array(stars[0].data.shape)


    combined = np.median([upsample_image(star.data/star.data.max(),
                                         upsample_factor=scaling[0],
                                         output_size=output_shape,
                                         xshift=star.cutout_center[0] - avg_center[0],
                                         yshift=star.cutout_center[1] - avg_center[1]
                                         ).real
                          for star in stars], axis=0)

    combined-=np.min(combined)
    combined/=np.max(combined)

    origin = np.array(combined.shape) / 2 / oversampling
    # TODO What we return here needs to actually use the image in it's __call__ operator to work as a model
    # type: ignore
    return photutils.psf.EPSFModel(combined, flux=None,
                                   origin=origin,
                                   oversampling=oversampling,
                                   normalize=False)


def make_epsf_fit(stars: photutils.psf.EPSFStars,
                  iters: int = config.epsfbuilder_iters,
                  oversampling: int = config.oversampling,
                  smoothing_kernel: Union[str, np.ndarray] = 'quartic',
                  epsf_guess: Optional[EPSFModel] = None) -> photutils.psf.EPSFModel:
    """
    wrapper around EPSFBuilder
    """
    try:
        epsf, fitted_stars = EPSFBuilder(oversampling=oversampling,
                                         maxiters=iters,
                                         progress_bar=True,
                                         smoothing_kernel=smoothing_kernel).build_epsf(stars, init_epsf=epsf_guess)
    except ValueError:
        print('Warning: epsf fit diverged. Some data will not be analyzed')
        raise
    return epsf


def do_photometry_epsf(image: np.ndarray,
                       epsf: photutils.psf.EPSFModel,
                       star_finder: Optional[photutils.StarFinderBase],
                       initial_guess: Optional[Table] = None,
                       config: Config = Config()
                       ) -> Table:
    """
    Given an image an a epsf model, perform photometry and return star positions (and more) in table
    :param image: input image
    :param epsf: EPSF model to use in photometry
    :param star_finder: which starfinder to use?
    :param initial_guess: initial estimates for star positions
    :param config:

    :return: Table with results
    """

    separation_factor = config.separation_factor
    clip_sigma = config.clip_sigma
    photometry_iterations = config.photometry_iterations

    epsf = photutils.psf.prepare_psf_model(epsf, renormalize_psf=False)  # renormalize is super slow...

    background_rms = MADStdBackgroundRMS()

    _, img_median, img_stddev = sigma_clipped_stats(image, sigma=clip_sigma)
    fwhm_guess = estimate_fwhm(epsf.psfmodel)

    grouper = DAOGroup(separation_factor * fwhm_guess)

    epsf.fwhm = astropy.modeling.Parameter('fwhm', 'this is not the way to add this I think')
    epsf.fwhm.value = fwhm_guess

    photometry = IterativelySubtractedPSFPhotometry(
        finder=star_finder,
        group_maker=grouper,
        bkg_estimator=background_rms,
        psf_model=epsf,
        fitter=LevMarLSQFitter(),
        niters=photometry_iterations,
        fitshape=config.fitshape
    )

    return photometry.do_photometry(image, init_guesses=initial_guess)


def cheating_astrometry(image, input_table, psf: np.ndarray, filename: str = '?', config: Config = Config.instance()):
    """
    Evaluate the maximum achievable precision of the EPSF fitting approach by using a hand-defined psf
    :param input_table:
    :param image:
    :param filename:
    :param psf:
    :param config:
    :return:
    """
    try:
        print(f'starting job on image {filename} with {config}')
        origin = np.array(psf.shape) / 2
        # type: ignore
        epsf = photutils.psf.EPSFModel(psf, flux=1, origin=origin, oversampling=1, normalize=False)
        epsf = photutils.psf.prepare_psf_model(epsf, renormalize_psf=False)

        finder = get_finder(image, config)

        #fwhm = estimate_fwhm(epsf.psfmodel)
        fwhm = config.fwhm_guess
        grouper = DAOGroup(config.separation_factor * fwhm)

        epsf.fwhm = astropy.modeling.Parameter('fwhm', 'this is not the way to add this I think')
        epsf.fwhm.value = fwhm
        bkgrms = MADStdBackgroundRMS()

        photometry = BasicPSFPhotometry(
            finder=finder,
            group_maker=grouper,
            bkg_estimator=bkgrms,
            psf_model=epsf,
            fitter=LevMarLSQFitter(),
            fitshape=config.fitshape
        )

        guess_table = input_table.copy()
        guess_table = cut_edges(guess_table, 101, image.shape[0])
        guess_table.rename_columns(['x', 'y'], ['x_0', 'y_0'])

        guess_table['x_0'] += np.random.uniform(-0.1, +0.1, size=len(guess_table['x_0']))
        guess_table['y_0'] += np.random.uniform(-0.1, +0.1, size=len(guess_table['y_0']))

        result_table = photometry(image, guess_table)

        return PhotometryResult(image, input_table, result_table, epsf, None, config, filename)
    except Exception as ex:
        import traceback
        print(f'error in cheating_astrometry({filename}, {psf}, {config})')
        error = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        print(error)
        return error


def run_photometry(image: np.ndarray,
                   input_table: Table,
                   filename: str = '?',
                   config=Config.instance()) -> Union[PhotometryResult, str]:
    """
    apply EPSF fitting photometry to a testimage

    :param input_table:
    :param image:
    :param filename:
    :param config: instance of Config containing all processing parameters
    :return: PhotometryResult, (image, input_table, result_table, epsf, star_guesses)
    """

    print(f'starting job on image {filename} with {config}')

    finder = get_finder(image, config)

    # TODO should this also be done using the catalogue positions?
    # TODO can we somehow sort the stars according to usefulness?
    star_guesses = make_stars_guess(image, finder, cutout_size=config.cutout_size)[:config.stars_to_keep]

    if len(star_guesses) < config.stars_to_keep:
        print('Warning: found less stars than config.stars_to_keep')

    epsf = make_epsf_fit(star_guesses,
                         iters=config.epsfbuilder_iters,
                         oversampling=config.oversampling,
                         smoothing_kernel=config.smoothing,
                         epsf_guess=config.epsf_guess)

    if config.use_catalogue_positions:
        guess_table = input_table.copy()
        guess_table = cut_edges(guess_table, config.cutout_size, image.shape[0])
        guess_table.rename_columns(['x', 'y'], ['x_0', 'y_0'])
        guess_table['x_0'] += np.random.uniform(-0.2, +0.2, size=len(guess_table['x_0']))
        guess_table['y_0'] += np.random.uniform(-0.2, +0.2, size=len(guess_table['y_0']))
    else:
        guess_table = None

    result_table = do_photometry_epsf(image, epsf, finder, initial_guess=guess_table, config=config)
    result_table['m'] = flux_to_magnitude(result_table['flux_fit'])

    return PhotometryResult(image, input_table, result_table, epsf, star_guesses, config, filename)