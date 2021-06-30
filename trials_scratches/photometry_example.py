from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import photutils
from astropy.nddata import NDData
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from matplotlib.colors import LogNorm


def getdata_safer(filename, *args, **kwargs):
    """
    Wrapper for astropy.io.fits.getdata to coerce data into float64 with native byteorder.
    FITS are default big-endian which can cause weird stuff to happen as numpy stores it as-is in memory
    :param filename: what to read
    :param args: further args to fits.getdata
    :param kwargs: further keyword-args to fits.getdata
    :return: the read data
    """
    import astropy.io.fits
    data = astropy.io.fits.getdata(filename, *args, **kwargs) \
        .astype(np.float64, order='C', copy=False)

    assert data.dtype.byteorder == '='
    assert data.flags.c_contiguous

    return data


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


def astrometry(image: np.ndarray,
               reference_table: Optional[Table] = None,
               known_psf: Optional[photutils.EPSFModel] = None):
    """
    All the steps necessary to do basic PSF astrometry with photutils
    :param image:
    :param reference_table:
    :param known_psf:
    :return:
    """
    if known_psf:
        fwhm = estimate_fwhm(known_psf)
    else:
        fwhm = fwhm_guess

    # get image stats and build finder
    mean, median, std = sigma_clipped_stats(image)
    finder = photutils.DAOStarFinder(threshold=median * threshold_factor,
                                     fwhm=fwhm * fwhm_factor,
                                     sigma_radius=sigma_radius,
                                     brightest=n_brightest,
                                     peakmax=peakmax)

    if reference_table:
        stars_tbl = reference_table.copy()
    else:
        stars_tbl = finder(image)
        stars_tbl.rename_columns(['xcentroid', 'ycentroid'], ['x', 'y'])

    # extract star cutouts and fit EPSF from them
    image_no_background = image - median
    stars = photutils.extract_stars(NDData(image_no_background), stars_tbl, size=cutout_size)
    epsf, fitted_stars = photutils.EPSFBuilder(oversampling=oversampling,
                                               maxiters=epsf_iters,
                                               progress_bar=True,
                                               smoothing_kernel=smoothing_kernel).build_epsf(stars)
    # renormalization is probably important if fluxes are interesting, for positions it does not seem to matter
    psf = photutils.prepare_psf_model(epsf, renormalize_psf=False)

    grouper = photutils.DAOGroup(group_radius * fwhm)

    if reference_table:
        photometry = photutils.BasicPSFPhotometry(
            group_maker=grouper,
            finder=None,
            bkg_estimator=photutils.MMMBackground(),  # Don't really know what kind of background estimator is preferred
            aperture_radius=fwhm,
            fitshape=fitshape,
            psf_model=psf)
        stars_tbl.rename_columns(['x', 'y'], ['x_0', 'y_0'])
        size = len(stars_tbl)
        # randomly perturb guesses to make fit do something
        stars_tbl['x_0'] += np.random.uniform(0.1, 0.2, size) * np.random.choice([-1, 1], size)
        stars_tbl['y_0'] += np.random.uniform(0.1, 0.2, size) * np.random.choice([-1, 1], size)

        result = photometry(image, init_guesses=stars_tbl)
    else:
        # it might be a good idea to build another finder/grouper here based on the derived EPSF
        photometry = photutils.IterativelySubtractedPSFPhotometry(
            group_maker=grouper,
            finder=finder,
            bkg_estimator=photutils.MMMBackground(),
            aperture_radius=fwhm,
            fitshape=fitshape,
            psf_model=psf,
            niters=photometry_iters
        )

        result = photometry(image)

    return result


# Parameters go here
known_psf_oversampling = 4  # for precomputed psf
known_psf_size = 201  # dito

# photometry
photometry_iters = 3
fitshape = 11

# epsf_fit
cutout_size = 21
smoothing_kernel = 'quadratic'  # can be quadratic, quartic or custom array
oversampling = 2
epsf_iters = 5

# grouper
group_radius = 1.5  # in fhwm

# starfinder
fwhm_guess = 3.
threshold_factor = 1  # img_median*this
fwhm_factor = 1.  # fwhm*this
# honestly I don't really know what this does, seems there's some truncation of a enhancement kernel...
# has a decent effect on the outcome though...
sigma_radius = 1.5
# use only n stars for epsf determination/each photometry iteration. usefull in iteratively subtracted photometry
n_brightest = 2000
minsep_fwhm = 0.3  # only find stars at least this*fwhm apart
peakmax = 100_000  # only find stars below this pixel value to avoid saturated stars in photometry

image_name = 'test.fits'  # path to image you want to analyze
# End parameters

if __name__ == '__main__':
    image_data = getdata_safer(image_name)
    reference_table = None  # read reference here if desired
    known_psf = None  # provide a-priori epsf if known

    result_table = astrometry(image_data, reference_table, known_psf)

    plt.imshow(image_data, cmap='inferno', norm=LogNorm())
    plt.plot(result_table['x_fit'], result_table['y_fit'], 'ko', markersize=1.5)
    plt.show()
