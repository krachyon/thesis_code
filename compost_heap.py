from scipy.spatial import cKDTree
from astropy.table import Table, Row
import numpy as np
from typing import List
import pyckles

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


def get_spectral_types() -> List[Row]:
    pickles_lib = pyckles.SpectralLibrary('pickles', return_style='synphot')
    return list(pickles_lib.table['name'])