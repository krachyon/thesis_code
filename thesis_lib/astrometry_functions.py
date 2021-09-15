from .config import Config
from .astrometry_types import ImageStats

import numpy as np
import warnings

from astropy.stats import sigma_clipped_stats
from photutils import extract_stars, EPSFStars
from astropy.nddata import NDData
from astropy.table import Table


def calc_image_stats(img: np.ndarray, config: Config) -> ImageStats:
    mean, median, std = sigma_clipped_stats(img, sigma=config.clip_sigma)
    threshold = median + config.threshold_factor * std
    return ImageStats(mean, median, std, threshold)


def extract_epsf_stars(image: np.ndarray, image_stats: ImageStats, stars_tbl: Table, config: Config) -> EPSFStars:
    image_no_background = image - image_stats.median
    # TODO to enable this, need to convert input table magnitude to flux also
    #stars_tbl_filtered = stars_tbl[stars_tbl['flux'] < config.detector_saturation]
    stars = extract_stars(NDData(image_no_background), stars_tbl, size=config.cutout_size)
    if len(stars) == 0:
        warnings.warn('No stars extracted')
    return stars[:config.max_epsf_stars]


def perturb_guess_table(input_table: Table, seed=0):
    rng = np.random.default_rng(seed=seed)
    input_table['x'] += rng.uniform(-0.2, +0.2, size=len(input_table['x']))
    input_table['y'] += rng.uniform(-0.2, +0.2, size=len(input_table['y']))