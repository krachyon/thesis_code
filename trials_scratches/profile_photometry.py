from astropy.io import fits
import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import IRAFStarFinder
from photutils.psf import (DAOGroup, IterativelySubtractedPSFPhotometry,
                           extract_stars)
from astropy.nddata import NDData
from photutils import EPSFBuilder, prepare_psf_model, Background2D
from astropy.stats import SigmaClip
from photutils.background import MADStdBackgroundRMS, MMMBackground
import pickle
import os

image = fits.getdata('./cutout_BS90_V.fits').astype(np.float64)

epsf_file = 'epsf.pkl'

if not os.path.exists(epsf_file):
    mean, median, std = sigma_clipped_stats(image)
    threshold = median + 10 * std

    finder = IRAFStarFinder(threshold=threshold, fwhm=4, minsep_fwhm=5, peakmax=image.max() / 0.8)

    star_table = finder(image)
    star_table.rename_columns(('xcentroid', 'ycentroid'),('x','y'))

    sigma_clip = SigmaClip(sigma=5.0)
    bkg_estimator = MMMBackground()
    bkg = Background2D(image, 5, filter_size=(3, 3),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

    nddata = NDData(image-bkg.background)
    stars = extract_stars(nddata, star_table, size=51)

    epsf, fitted_stars = EPSFBuilder(oversampling=4, maxiters=3, progress_bar=True, smoothing_kernel='quadratic')(stars)
    epsf_model = prepare_psf_model(epsf, renormalize_psf=False)

    with open(epsf_file,'wb') as f:
        pickle.dump([epsf_model, finder], f)
else:
    with open(epsf_file, 'rb') as f:
        epsf_model, finder = pickle.load(f)

phot = IterativelySubtractedPSFPhotometry(group_maker=DAOGroup(5),
                                          bkg_estimator=MMMBackground(),
                                          psf_model=epsf_model,
                                          fitshape=[31,31],
                                          finder=finder,
                                          aperture_radius=5,
                                          niters=2)
phot(image)
