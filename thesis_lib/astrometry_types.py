from __future__ import annotations  # makes the "-> __class__" annotation work...

from collections import namedtuple, OrderedDict
from typing import Tuple, Optional, Union
from astropy.table import Table

ImageStats = namedtuple('ImageStats', ['mean', 'median', 'std', 'threshold'])

X = 'X'
Y = 'Y'
FLUX = 'FLUX'
MAGNITUDE = 'MAGNITUDE'


INPUT_TABLE_NAMES = {X: 'x', Y: 'y', MAGNITUDE: 'm', FLUX: 'f'}
REFERENCE_NAMES = {X: 'x_orig', Y: 'y_orig', MAGNITUDE: 'm_orig', FLUX: 'flux_orig'}
STARFINDER_TABLE_NAMES = {X: 'xcentroid', Y: 'ycentroid', FLUX: 'peak', MAGNITUDE: 'mag'}
GUESS_TABLE_NAMES = {X: 'x_0', Y: 'y_0', FLUX: 'flux_0'}
PHOTOMETRY_TABLE_NAMES = {X: 'x_fit', Y: 'y_fit', FLUX: 'flux_fit'}


class TypeCheckedTable(Table):
    required_columns: Optional[dict] = None

    def __init__(self, other_table: Table, *args, **kwargs):
        if not self.required_columns:
            raise ValueError('Please instanciate a child class')
        for name in self.required_columns.values():
            if name not in other_table.colnames:
                raise ValueError(f'required column {name} not present')
        super().__init__(other_table, *args, **kwargs)

    def typecast(self, to_table_kind: __class__) -> __class__:
        return_table = self.copy()
        for key in to_table_kind.required_columns.keys():
            if key not in self.required_columns:
                raise ValueError(f"Can't convert from {type(self)} to {to_table_kind}. Column type {key} not present")
            return_table[to_table_kind.required_columns[key]] = self[self.required_columns[key]]

        return return_table

class StarfinderTable(TypeCheckedTable):
    required_columns = STARFINDER_TABLE_NAMES
class InputTable(TypeCheckedTable):
    required_columns = INPUT_TABLE_NAMES
class GuessTable(TypeCheckedTable):
    required_columns = GUESS_TABLE_NAMES
class ResultTable(TypeCheckedTable):
    required_columns = PHOTOMETRY_TABLE_NAMES

ALL_TABLE_NAMES = (INPUT_TABLE_NAMES, STARFINDER_TABLE_NAMES, GUESS_TABLE_NAMES, PHOTOMETRY_TABLE_NAMES)