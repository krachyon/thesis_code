import pytest
import numpy as np

from thesis_lib.astrometry.functions import match_finder_to_reference, calc_extra_result_columns
from thesis_lib.astrometry.types import InputTable, INPUT_TABLE_NAMES,\
StarfinderTable, STARFINDER_TABLE_NAMES,\
ReferenceTable, REFERENCE_NAMES,\
ResultTable, X, Y, FLUX, MAGNITUDE


def make_catalogue_table(data):
    return InputTable(data=data, names=(INPUT_TABLE_NAMES[X],
                                         INPUT_TABLE_NAMES[Y],
                                         INPUT_TABLE_NAMES[FLUX],
                                         INPUT_TABLE_NAMES[MAGNITUDE]))

def make_starfinder_table(data):
    return StarfinderTable(data=data, names = (STARFINDER_TABLE_NAMES[X],
                                         STARFINDER_TABLE_NAMES[Y],
                                         STARFINDER_TABLE_NAMES[FLUX],
                                         STARFINDER_TABLE_NAMES[MAGNITUDE]) )

@pytest.fixture
def simple_data():
    # X, Y, Flux, Mag
    return np.array(
        [[10, 10, 8, 3],
         [20, 20, 3, 2],
         [20, 10, 1, 5],
         [10, 20, 9, 9],
         [0, 0, 0, 0]]
    )

@pytest.fixture
def offset_data():
    # X, Y, Flux, Mag
    return np.array(
        [[10.1, 10, 8, 3],
         [20.1, 20, 3, 2],
         [20.1, 10, 1, 5],
         [10.1, 20, 9, 9],
         [0.1, 0, 0, 0]]
    )


@pytest.fixture
def catalogue_table_simple(simple_data):
    return make_catalogue_table(simple_data)


@pytest.fixture
def finder_table_simple_perfect(simple_data):
    return make_starfinder_table(simple_data)

@pytest.fixture
def finder_table_simple_slight_difference(offset_data):
    return make_starfinder_table(offset_data)


def test_matching_type(catalogue_table_simple, finder_table_simple_perfect):
    res = match_finder_to_reference(finder_table_simple_perfect, catalogue_table_simple)
    assert ReferenceTable(res)


def test_matching_perfect(catalogue_table_simple, finder_table_simple_perfect):
    res = match_finder_to_reference(finder_table_simple_perfect, catalogue_table_simple)
    assert np.all(np.isclose(res[REFERENCE_NAMES[X]], res[STARFINDER_TABLE_NAMES[X]]))
    assert np.all(np.isclose(res[REFERENCE_NAMES[Y]], res[STARFINDER_TABLE_NAMES[Y]]))
    assert np.all(np.isclose(res[REFERENCE_NAMES[FLUX]], res[STARFINDER_TABLE_NAMES[FLUX]]))
    assert np.all(np.isclose(res[REFERENCE_NAMES[MAGNITUDE]], res[STARFINDER_TABLE_NAMES[MAGNITUDE]]))


def test_matching_offset(catalogue_table_simple, finder_table_simple_slight_difference):
    res = match_finder_to_reference(finder_table_simple_slight_difference, catalogue_table_simple)

    assert np.all(np.isclose(np.abs(res[REFERENCE_NAMES[X]]-res[STARFINDER_TABLE_NAMES[X]]), 0.1))
    assert np.all(np.isclose(np.abs(res[REFERENCE_NAMES[Y]]-res[STARFINDER_TABLE_NAMES[Y]]), 0))


def test_additional_perfect(catalogue_table_simple):
    guess_table = catalogue_table_simple.convert(ReferenceTable)
    mock_result = guess_table.convert(ResultTable)
    res = calc_extra_result_columns(mock_result)
    assert(np.all(np.isclose(res['offset'], 0)))
    assert(np.all(np.isclose(res['x_offset'], 0)))
    assert(np.all(np.isclose(res['y_offset'], 0)))


def test_additional_offset(catalogue_table_simple, finder_table_simple_slight_difference):
    mock_result = match_finder_to_reference(
        finder_table_simple_slight_difference, catalogue_table_simple).convert(ResultTable)
    mock_result = mock_result.convert(ResultTable)
    res = calc_extra_result_columns(mock_result)
    assert (np.all(np.isclose(res['offset'], 0.1)))
    assert (np.all(np.isclose(res['x_offset'], 0.1)))
    assert (np.all(np.isclose(res['y_offset'], 0)))
