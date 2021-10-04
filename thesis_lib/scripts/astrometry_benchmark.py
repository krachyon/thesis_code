import os
import pickle
from typing import Union, Callable, Tuple, Optional

import matplotlib.pyplot as plt
import multiprocess as mp  # not multiprocessing, this can pickle lambdas
from multiprocess.pool import Pool as Pool_t
import numpy as np
from astropy.table import Table
import astropy.table


import thesis_lib.testdata_definitions
from thesis_lib import testdata_generators
from thesis_lib import util
from thesis_lib.config import Config

from thesis_lib.scopesim_helper import download
from thesis_lib.testdata_generators import read_or_generate_image, read_or_generate_helper
from thesis_lib import testdata_definitions
from thesis_lib.astrometry_wrapper import Session
from thesis_lib.astrometry_plots import make_all_plots
from thesis_lib.util import make_gauss_kernel, green, blue, yellow


def photometry_multi(image_recipe_template: Callable[[int], Callable[[], Tuple[np.ndarray, Table]]],
                     image_name_template: str,
                     n_images: int,
                     config=Config.instance(),
                     threads: Union[int, None, bool]=None) -> list[Session]:
    """
    """

    def inner(i):
        image_recipe = image_recipe_template(i)
        image_name = image_name_template+f'_{i}'
        image, input_table = read_or_generate_image(image_name, config, image_recipe)
        session = Session(config, image, input_table)
        session.do_it_all()

        # TODO maybe do this as part of the calc_additional function
        session.tables.result_table['σ_pos_estimated'] = \
            util.estimate_photometric_precision_full(image, input_table, session.fwhm)
        return session

    if threads is False:
        sessions = list(map(inner, range(n_images)))
    else:
        with mp.Pool(threads) as pool:
            sessions = pool.map(inner, range(n_images))

    return sessions


def runner(config, image_name) -> Session:

    print(yellow('running ') + f' {image_name}\n{str(config)}')
    session = Session(config, image_name)
    session.do_it_all()
    make_all_plots(session, save_files=True)
    plt.close('all')
    print(blue('done') + f'with {image_name}\n{str(config)}')
    return session


def main():
    download()
    config = Config()
    config.separation_factor = 1
    config.photometry_iterations = 1

    normal_config = config.copy()
    normal_config.output_folder = 'output_files/normal'

    gauss_config = config.copy()
    gauss_config.smoothing = make_gauss_kernel(0.5)
    gauss_config.output_folder = 'output_files/gauss'

    known_position_config = config.copy()
    known_position_config.use_catalogue_positions = True
    known_position_config.output_folder = 'output_files/known_position'


    configs = [known_position_config, normal_config, gauss_config, ]
    for config in configs:
        config.create_dirs()

    parameters = [(config, image_name)
                  for config in configs
                  for image_name in testdata_definitions.benchmark_images]

    n_threads = None
    #from thesis_lib.util import DebugPool
    #with DebugPool() as pool:
    with mp.Pool(n_threads) as pool:
        run_sessions = pool.starmap(runner, parameters)
    with open('../../all_photometry_results.pickle', 'wb') as f:
        pickle.dump(run_sessions, f)


if __name__ == '__main__':
    main()
