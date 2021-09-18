from thesis_lib.astrometry_types import StarfinderTable, InputTable, TypeCheckedTable,\
    STARFINDER_TABLE_NAMES, INPUT_TABLE_NAMES
from astropy.table import Table
import numpy as np

from pytest import raises


def test_typechecked_table_use():
    valid_table = Table(np.ones((3, 4)), names=STARFINDER_TABLE_NAMES.values())
    # initial construction
    sftab = StarfinderTable(valid_table).validate()
    # copy construction
    assert StarfinderTable(sftab).validate()
    assert Table(sftab)
    # slicing
    assert len(sftab[[True, False, True]].validate()) == 2


def test_typechecked_table_validation():
    invalid_table = Table(np.ones((3, 4)), names=['not', 'real', 'column', 'names'])

    with raises(ValueError) as e:
        StarfinderTable(invalid_table).validate()

    with raises(ValueError) as e:
        # can't instanciate base
        TypeCheckedTable().validate()


def test_typechecked_table_cast():
    sftab = StarfinderTable(Table(np.arange(4*10).reshape(10, 4), names=STARFINDER_TABLE_NAMES.values())).validate()
    inputtab = sftab.convert(InputTable)
    assert isinstance(inputtab, InputTable)

