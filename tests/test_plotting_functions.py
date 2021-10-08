import os

from thesis_lib.astrometry.plots import make_all_plots
from thesis_lib.util import work_in
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt

import pytest


def test_all_plots(session_grid):
    with TemporaryDirectory() as tmpdirname:
        with work_in(tmpdirname):

            session_grid.do_it_all()
            session_grid.config.output_folder = '.'
            assert session_grid.tables.result_table

            with pytest.warns(UserWarning):
                make_all_plots(session_grid, save_files=True)

            dir_content = os.listdir('.')
            assert len(dir_content) > 8  # at least 4 plots, two files each

            plt.close('all')