from .config import Config
from .astrometry_types import ImageStats, INPUT_TABLE_NAMES, STARFINDER_TABLE_NAMES, REFERENCE_NAMES, InputTable,\
    X,Y,FLUX,MAGNITUDE

import numpy as np
import warnings
from scipy.spatial import cKDTree

from astropy.stats import sigma_clipped_stats
from photutils import extract_stars, EPSFStars
from astropy.nddata import NDData
from astropy.table import Table


def calc_image_stats(img: np.ndarray, config: Config) -> ImageStats:
    mean, median, std = sigma_clipped_stats(img, sigma=config.clip_sigma)
    threshold = median + config.threshold_factor * std
    return ImageStats(mean, median, std, threshold)


def extract_epsf_stars(image: np.ndarray, image_stats: ImageStats, stars_tbl: InputTable, config: Config) -> EPSFStars:
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


def match_finder_to_reference(finder_table: Table, reference_table: Table):
    """Needed to relate output of starfinder to reference table."""
    assert set(INPUT_TABLE_NAMES.values()).issubset(reference_table.colnames)

    xym_pixel = np.array((reference_table[INPUT_TABLE_NAMES[X]],
                          reference_table[INPUT_TABLE_NAMES[Y]])).T
    lookup_tree = cKDTree(xym_pixel)

    finder_table[REFERENCE_NAMES[X]] = np.nan
    finder_table[REFERENCE_NAMES[Y]] = np.nan
    finder_table[REFERENCE_NAMES[FLUX]] = np.nan
    finder_table[REFERENCE_NAMES[MAGNITUDE]] = np.nan
    finder_table['reference_index'] = np.nan

    for row in finder_table:
        dist, index = lookup_tree.query(
                (row[STARFINDER_TABLE_NAMES[X]], row[STARFINDER_TABLE_NAMES[Y]]))

        row[REFERENCE_NAMES[X]] = reference_table[INPUT_TABLE_NAMES[X]][index]
        row[REFERENCE_NAMES[Y]] = reference_table[INPUT_TABLE_NAMES[Y]][index]
        row[REFERENCE_NAMES[MAGNITUDE]] = reference_table[INPUT_TABLE_NAMES[MAGNITUDE]][index]
        row[REFERENCE_NAMES[FLUX]] = reference_table[INPUT_TABLE_NAMES[FLUX]][index]
        row['reference_index'] = index

    return finder_table


def calc_extra_result_columns(result_table):
    result_table['x_offset'] = result_table['x_fit'] - result_table['x_orig']
    result_table['y_offset'] = result_table['y_fit'] - result_table['y_orig']
    result_table['offset'] = np.sqrt(result_table['x_offset'] ** 2 + result_table['y_offset'] ** 2)