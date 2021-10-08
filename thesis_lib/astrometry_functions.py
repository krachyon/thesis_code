from .config import Config
from .astrometry_types import ImageStats, INPUT_TABLE_NAMES, STARFINDER_TABLE_NAMES, REFERENCE_NAMES, GUESS_TABLE_NAMES,\
    RESULT_TABLE_NAMES,\
    X,Y,FLUX,MAGNITUDE, OFFSET, XOFFSET, YOFFSET, OUTLIER,\
    InputTable, ResultTable, GuessTable

import numpy as np
import warnings
from scipy.spatial import cKDTree
from typing import Optional

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

    stars_tbl_filtered = stars_tbl[stars_tbl[INPUT_TABLE_NAMES[FLUX]] < config.detector_saturation]
    stars_tbl_filtered.sort(INPUT_TABLE_NAMES[FLUX], reverse=True)
    stars = extract_stars(NDData(image_no_background), stars_tbl_filtered, size=config.cutout_size)
    if len(stars) == 0:
        warnings.warn('No stars extracted')
    return stars[:config.max_epsf_stars]


def perturb_guess_table(input_table: GuessTable, perturb_catalogue_guess: Optional[float] = 0.25, seed=0) -> Table:
    if perturb_catalogue_guess:
        rng = np.random.default_rng(seed=seed)
        pm = perturb_catalogue_guess/2
        input_table[GUESS_TABLE_NAMES[X]] += rng.uniform(-pm, +pm, size=len(input_table['x']))
        input_table[GUESS_TABLE_NAMES[Y]] += rng.uniform(-pm, +pm, size=len(input_table['y']))
    return input_table


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

    finder_table['reference_index'] = finder_table['reference_index'].astype(int)
    return finder_table


# TODO may be a bit overkill but would be nice if the additional names had their own type
def calc_extra_result_columns(result_table: ResultTable) -> ResultTable:
    result_table[XOFFSET] = result_table[RESULT_TABLE_NAMES[X]] - result_table[REFERENCE_NAMES[X]]
    result_table[YOFFSET] = result_table[RESULT_TABLE_NAMES[Y]] - result_table[REFERENCE_NAMES[Y]]
    result_table[OFFSET] = np.sqrt(result_table[XOFFSET] ** 2 + result_table[YOFFSET] ** 2)
    return result_table


def mark_outliers(result_table: ResultTable) -> ResultTable:
    result_table[OUTLIER] = result_table[OFFSET] >= 1
    return result_table
