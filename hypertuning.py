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

image_name = 'scopesim_grid_16_perturb2_lowpass_mag18_24'
image_recipe = testdata_generators.benchmark_images[image_name]


def objective(cutout_size: int, fitshape_half: int, oversampling: int):
    config = Config()
    config.use_catalogue_positions = True
    config.smoothing = util.make_gauss_kernel()
    config.photometry_iterations = 1

    config.fitshape = fitshape_half*2+1
    config.oversampling = oversampling
    config.cutout_size = cutout_size

    image, input_table = testdata_generators.read_or_generate_image(image_recipe, image_name, config.image_folder)
    result = run_photometry(image, input_table, image_name, config)
    result_table = util.match_observation_to_source(input_table, result.result_table)

    loss = np.sqrt(np.sum(result_table['offset']**2))
    return loss


if os.path.exists('optimize_result.pkl'):
    with open('optimize_result.pkl', 'rb') as f:
        optimizer = dill.load(f)
else:
    optimizer = Optimizer(
        dimensions=[Integer(10, 40), Integer(14, 75), Integer(1, 5)],
        n_jobs=12,
        random_state=1,
        base_estimator='GP',
        n_initial_points=13,
    )
#GaussianProcessRegressor(noise=1e-10)


def not_ready(job):
    return not job.ready()

def make_callback(args):
    def cb(r):
        print('telling... ', args)
        optimizer.tell(args, r)
    return cb

if __name__ == '__main__':
    # with mp.Pool() as p:
    #     jobs = []
    #     try:
    #         for i in range(160):
    #             args = optimizer.ask()  # x is a list of n_points points
    #             optimizer.update_next()
    #             print(args)
    #             jobs.append(p.apply_async(objective, args, callback=make_callback(args)))
    #             while sum(map(not_ready, jobs)) >= 12:
    #                 time.sleep(0.5)
    #         [j.get() for j in jobs]
    #     except KeyboardInterrupt:
    #         pass

    with mp.Pool() as p:
        try:
            for i in range(10):
                args = optimizer.ask(n_points=12)
                result = p.starmap(objective, args)
                optimizer.tell(args, result)
        except KeyboardInterrupt:
            pass


    res = optimizer.get_result()

    with open('optimize_result.pkl', 'wb') as f:
        dill.dump(optimizer, f)

    #skopt.plots.plot_objective(res)
    #plt.show()