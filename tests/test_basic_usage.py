import numpy as np
import pytest

from thesis_lib.testdata_generators import read_or_generate_image
from thesis_lib.astrometry_types import GUESS_TABLE_NAMES, RESULT_TABLE_NAMES, X, Y

def test_oneline(session_single):
    session_single.do_it_all()

    assert session_single.tables.result_table
    assert len(session_single.tables.result_table) == 1


def test_session(session_single):

    # equivalent way of
    session_single.image = 'testsingle'

    image, input_table = read_or_generate_image('testsingle')
    session_single.image = image
    session_single.input_table = input_table

    session_single.find_stars()
    session_single.select_epsfstars_auto()
    session_single.make_epsf()
    # Here we could e.g. change starfinder and re_run find_stars()
    # TODO
    # session.cull_detections()
    # session.select_epsfstars_qof()
    session_single.make_epsf()
    session_single.do_astrometry()

    assert session_single.tables.result_table
    assert len(session_single.tables.result_table) == 1


def test_builder_session(session_single):
    session_single.find_stars().select_epsfstars_auto().make_epsf().do_astrometry()

    assert session_single.tables.result_table
    assert len(session_single.tables.result_table) == 1


def test_multi_image(session_multi):
    session_multi.find_stars()
    assert len(session_multi.tables.input_table) == len(session_multi.tables.finder_table)
    session_multi.select_epsfstars_auto().make_epsf().do_astrometry()
    assert len(session_multi.tables.result_table) == len(session_multi.tables.input_table)


def test_multi_twice(session_multi):
    session_multi.do_it_all()
    assert session_multi.tables.result_table
    session_multi.do_it_all()


def test_with_catalogue_positions(session_multi):
    session_multi.config.use_catalogue_positions = True
    session_multi.do_it_all()
    result_table = session_multi.tables.result_table
    positions_equal = np.isclose(result_table[RESULT_TABLE_NAMES[X]], result_table[GUESS_TABLE_NAMES[X]]) & \
                      np.isclose(result_table[RESULT_TABLE_NAMES[Y]], result_table[GUESS_TABLE_NAMES[Y]])
    assert np.sum(positions_equal) <= 1  # maybe there are a few fits that hit exactly, maybe not


def test_with_catalogue_positions_grid(session_grid):
    session_grid.config.use_catalogue_positions = True
    session_grid.do_it_all()
    result_table = session_grid.tables.result_table
    positions_equal = np.isclose(result_table[RESULT_TABLE_NAMES[X]], result_table[GUESS_TABLE_NAMES[X]]) & \
                      np.isclose(result_table[RESULT_TABLE_NAMES[Y]], result_table[GUESS_TABLE_NAMES[Y]])
    assert np.sum(positions_equal) <= 2  # maybe there are a few fits that hit exactly, maybe not