from thesis_lib.parameter_tuning import *
from thesis_lib.testdata_generators import benchmark_images

import skopt
from skopt.space import Real, Categorical, Integer
import dill


if __name__ == '__main__':
    for name, recipe in benchmark_images.items():

        starfinder_obj, starfinder_dims = make_starfinder_objective(recipe, name)

        starfinder_optimizer = skopt.Optimizer(
            dimensions=starfinder_dims,
            n_jobs=12,
            random_state=1,
            base_estimator='RF',
            n_initial_points=12,
            initial_point_generator='random'
        )

        starfinder_result = run_optimizer(starfinder_optimizer, starfinder_obj, n_evaluations=12*2)

        with open(name+'_starfinder_opt.pkl', 'wb') as f:
            dill.dump(starfinder_result, f)

        x = starfinder_result.x
        config = Config()
        config.threshold_factor = x[0]
        config.fwhm_guess = x[1]
        config.sigma_radius = x[2]
        config.roundlo = x[3]
        config.roundhi = x[4]
        config.sharplo = x[5]
        config.sharphi = x[6]
        config.use_catalogue_positions = False
        config.photometry_iterations = 1  # TODO this should also be optimized maybe

        epsf_obj, epsf_dims = make_epsf_objective(config, recipe, name)

        epsf_optimizer = skopt.Optimizer(
            dimensions=epsf_dims,
            n_jobs=12,
            random_state=1,
            base_estimator='RF',
            n_initial_points=12,
            initial_point_generator='random'
        )
        epsf_result = run_optimizer(epsf_optimizer, epsf_obj, n_evaluations=12*2)

        with open(name+'_epsf_opt.pkl', 'wb') as f:
            dill.dump(epsf_result, f)
