import time

from thesis_lib import testdata_generators
from thesis_lib.photometry import run_photometry
from thesis_lib.config import Config
from thesis_lib import util

import numpy as np
from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor
from skopt.space import Real, Integer, Categorical
import skopt.plots
import matplotlib.pyplot as plt
import multiprocess as mp
import dill
import os
from collections import namedtuple

result_filename = 'optimize_result_RF_lpcluster.pkl'
image_name = 'gausscluster_N2000_mag22_lowpass'
image_recipe = testdata_generators.benchmark_images[image_name]


def objective(cutout_size: int, fitshape_half: int, sigma: float, iters: int):
    try:
        config = Config()
        config.use_catalogue_positions = True
        config.photometry_iterations = 1
        config.oversampling = 2

        config.smoothing = util.make_gauss_kernel(sigma)
        config.fitshape = fitshape_half*2+1
        config.cutout_size = cutout_size
        config.epsfbuilder_iters=iters

        image, input_table = testdata_generators.read_or_generate_image(image_recipe, image_name, config.image_folder)
        result = run_photometry(image, input_table, image_name, config)
        result_table = util.match_observation_to_source(input_table, result.result_table)

        loss = np.sqrt(np.sum(result_table['offset']**2))
        return loss
    except Exception as ex:
        import traceback
        print('\033[93m##############\033[0m')
        print(f'error in objective({cutout_size}, {fitshape_half}, {sigma}, {iters})')
        print('\033[93m##############\033[0m')
        error = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        print(error)
        raise ex


#GaussianProcessRegressor(noise=1e-10)

Job = namedtuple('Job', ('result', 'args'))


def not_ready(job: Job):
    return not job.result.ready()


if __name__ == '__main__':
    if os.path.exists(result_filename):
        with open(result_filename, 'rb') as f:
            optimizer = dill.load(f)
    else:
        optimizer = Optimizer(
            dimensions=[Integer(5, 40), Integer(1, 10), Real(0.30, 3.), Categorical([4, 5, 7, 10])],
            n_jobs=12,
            random_state=1,
            base_estimator='RF',
            n_initial_points=14,
            initial_point_generator='random'
        )

    n_procs = 11
    with mp.Pool(n_procs) as p:
        jobs = []
        try:
            for i in range(100):
                args = optimizer.ask()
                optimizer.update_next()
                print('#######')
                print(f'Evaluation #{i}')
                print(args)
                print('#######')
                jobs.append(Job(p.apply_async(objective, args), args))
                for job in jobs:
                    if job.result.ready():
                        optimizer.tell(job.args, job.result.get())
                        jobs.remove(job)
                while sum(map(not_ready, jobs)) >= n_procs:
                    time.sleep(0.5)
            for job in jobs:
                optimizer.tell(job.args, job.result.get())
        except KeyboardInterrupt:
            pass

    # with mp.Pool() as p:
    #     try:
    #         for i in range(30):
    #             args = optimizer.ask(n_points=12)
    #             result = p.starmap(objective, args)
    #             optimizer.tell(args, result)
    #     except KeyboardInterrupt:
    #         pass


    res = optimizer.get_result()

    with open(result_filename, 'wb') as f:
        dill.dump(optimizer, f)

    #skopt.plots.plot_objective(res)
    #plt.show()
