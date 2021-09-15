from collections import namedtuple
from typing import Tuple, Optional, Union
from astropy.table import Table

ImageStats = namedtuple('ImageStats', ['mean', 'median', 'std', 'threshold'])


XNAME = 'xname'
YNAME = 'yname'
FLUXNAME = 'fluxname'
MAGNAME = 'magname'

KEYSET = {XNAME, YNAME, FLUXNAME, MAGNAME}

INPUT_TABLE_NAMES = {XNAME: 'x', YNAME: 'y', MAGNAME: 'm'}  # TODO maybe just calculate this in reference tables
STARFINDER_TABLE_NAMES = {XNAME: 'xcentroid', YNAME: 'ycentroid', FLUXNAME: 'peak', MAGNAME: 'mag'}
GUESS_TABLE_NAMES = {XNAME: 'x_0', YNAME: 'y_0', FLUXNAME: 'flux_0'}
PHOTOMETRY_TABLE_NAMES = {XNAME: 'x_fit', YNAME: 'y_fit', FLUXNAME: 'flux_fit'}

ALL_TABLE_NAMES = (INPUT_TABLE_NAMES, STARFINDER_TABLE_NAMES, GUESS_TABLE_NAMES, PHOTOMETRY_TABLE_NAMES)


def adapt_table_names(input_table: Table, to_names: dict[str, str] = INPUT_TABLE_NAMES):
    for name_dict in ALL_TABLE_NAMES:
        if set(name_dict.values()).issubset(input_table.colnames):
            from_names = name_dict
            break
    else:
        raise ValueError(f'could not match column names of {input_table}')

    output_table = input_table.copy()
    for key in set(from_names.keys()) & set(to_names.keys()):
        output_table.rename_column(from_names[key], to_names[key])
    return output_table

