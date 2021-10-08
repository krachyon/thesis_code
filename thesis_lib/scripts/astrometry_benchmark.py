import pickle

import matplotlib.pyplot as plt
import multiprocess as mp  # not multiprocessing, this can pickle lambdas

from thesis_lib.config import Config

from thesis_lib.scopesim_helper import download
from thesis_lib import testdata_definitions
from thesis_lib.astrometry_wrapper import Session
from thesis_lib.astrometry_plots import make_all_plots
from thesis_lib.util import make_gauss_kernel, blue, yellow


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
    gauss_config.detector_saturation = 150000
    gauss_config.smoothing = make_gauss_kernel(0.5)
    gauss_config.output_folder = 'output_files/gauss'

    known_position_config = gauss_config.copy()
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
