from __future__ import annotations  # makes the "-> __class__" annotation work...

import typing
from typing import Tuple, Optional, Union
from collections import namedtuple, OrderedDict

from astropy.table import Table

ImageStats = namedtuple('ImageStats', ['mean', 'median', 'std', 'threshold'])

# Define constants for looking up names of Table columns
ColumnType = typing.NewType('ColumnType', str)

X = ColumnType('X')
Y = ColumnType('Y')
FLUX = ColumnType('FLUX')
MAGNITUDE = ColumnType('MAGNITUDE')


INPUT_TABLE_NAMES: dict[ColumnType, str] = {X: 'x', Y: 'y', MAGNITUDE: 'm', FLUX: 'f'}
REFERENCE_NAMES: dict[ColumnType, str] = {X: 'x_orig', Y: 'y_orig', MAGNITUDE: 'm_orig', FLUX: 'flux_orig'}
STARFINDER_TABLE_NAMES: dict[ColumnType, str] = {X: 'xcentroid', Y: 'ycentroid', FLUX: 'peak', MAGNITUDE: 'mag'}
GUESS_TABLE_NAMES: dict[ColumnType, str] = {X: 'x_0', Y: 'y_0', FLUX: 'flux_0'}
PHOTOMETRY_TABLE_NAMES: dict[ColumnType, str] = {X: 'x_fit', Y: 'y_fit', FLUX: 'flux_fit'}


class TypeCheckedTable(Table):
    """This class is a Table that enforces certain columns to be present. Columns are specified as a dictionary
    made from mapping the predefined types of column to their names.
    """
    required_columns: Optional[dict[ColumnType, str]] = None

    # The reason why we can't override __init__ and validate there is that the slicing implementation
    # in astropy.Table insists on instanciating an empty "__class__" object which would fail validation
    # for that reason also making __init__ explicitly a copy-constructor from other tables fails

    def convert(self, to_table_kind: __class__) -> __class__:
        """Returns a table with column names translated to the desired kind of Table"""
        return_table = self.copy()
        for key in to_table_kind.required_columns.keys():
            if key not in self.required_columns:
                raise ValueError(f"Can't convert from {type(self)} to {to_table_kind}. Column type {key} not present")
            return_table[to_table_kind.required_columns[key]] = self[self.required_columns[key]]

        return to_table_kind(return_table).validate()

    def validate(self) -> __class__:
        """Call this after instanciating to make sure the required columns are present
        >>> my_table = SomeTableType(otherTable).validate()
        """
        if type(self) is TypeCheckedTable:
            raise ValueError(f'Please create a child class of {__class__}')
        for name in self.required_columns.values():
            if name not in self.colnames:
                raise ValueError(f'required column {name} not present')
        return self


class StarfinderTable(TypeCheckedTable):
    required_columns = STARFINDER_TABLE_NAMES
class InputTable(TypeCheckedTable):
    required_columns = INPUT_TABLE_NAMES
class GuessTable(TypeCheckedTable):
    required_columns = GUESS_TABLE_NAMES
class ResultTable(TypeCheckedTable):
    required_columns = PHOTOMETRY_TABLE_NAMES


ALL_TABLE_NAMES = (INPUT_TABLE_NAMES, STARFINDER_TABLE_NAMES, GUESS_TABLE_NAMES, PHOTOMETRY_TABLE_NAMES)