from collections import namedtuple
from typing import Tuple, Optional
from astropy.table import Table

ImageStats = namedtuple('ImageStats', ['mean', 'median', 'std', 'threshold'])



INPUT_TABLE_NAMES = ('x', 'y', None, 'm')  # TODO maybe just calculate this in reference tables
STARFINDER_TABLE_NAMES = ('xcentroid', 'ycentroid', 'peak', 'mag')
GUESS_TABLE_NAMES = ('x_0', 'y_0', 'flux_0', None)
PHOTOMETRY_TABLE_NAMES = ('x_fit', 'y_fit', 'flux_fit', 'm')
ALL_TABLE_NAMES = (INPUT_TABLE_NAMES, STARFINDER_TABLE_NAMES, GUESS_TABLE_NAMES, PHOTOMETRY_TABLE_NAMES)



def adapt_table_names(input_table: Table, to_names: Tuple[Optional[str]] = INPUT_TABLE_NAMES):
    from_names = None
    for name_set in ALL_TABLE_NAMES:
        found = True
        for test_name in name_set:
            if test_name:
                if test_name not in input_table.colnames:
                    found = False
                    break
        if found:
            from_names = name_set
            break
        else:
            raise ValueError(f'Table {input_table} does not contain recognizable columns')

    output_table = input_table.copy()
    output_table.rename_columns(from_names, to_names)
    return output_table

