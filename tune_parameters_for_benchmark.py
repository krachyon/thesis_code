from thesis_lib.parameter_tuning import *
from thesis_lib.testdata_generators import benchmark_images

import skopt
from skopt.space import Real, Categorical, Integer
import dill
import multiprocess as mp

if __name__ == '__main__':
    for name, recipe in benchmark_images.items():

        starfinder_obj = make_starfinder_objective(recipe, name)
        starfinder_dims = [
            Real(-6., 3, name='threshold'),
            Real(2., 8, name='fwhm'),
            Real(2., 5.5, name='sigma_radius'),
            Real(-15., 0., name='roundlo'),
            Real(0., 15., name='roundhi'),
            Real(-15, 0., name='sharplo'),
            Real(0., 10., name='sharphi')
        ]

        starfinder_optimizer = skopt.Optimizer(
            dimensions=starfinder_dims,
            n_jobs=mp.cpu_count(),
            random_state=1,
            base_estimator='RF',
            n_initial_points=4000,
            initial_point_generator='random'
        )

        starfinder_result = run_optimizer(starfinder_optimizer, starfinder_obj, n_evaluations=5000)

        with open(name+'_starfinder_opt.pkl', 'wb') as f:
            dill.dump(starfinder_result, f)
        with open(name+'starfinder_params.txt', 'w') as f:
            f.write(str(list(zip(starfinder_result.space.dimension_names, starfinder_result.x))))

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
        config.exclude_border = True
        config.photometry_iterations = 1  # TODO this should also be optimized maybe

        epsf_obj = make_epsf_objective(config, recipe, name)
        epsf_dims = [
            Integer(5, 40, name='cutout_size'),
            Integer(1, 10, name='fitshape_half'),
            Real(0.30, 3., name='sigma'),
            Categorical([4, 5, 7, 10], name='iters'),
            Categorical([1, 2, 4], name='oversampling')
        ]

        epsf_optimizer = skopt.Optimizer(
            dimensions=epsf_dims,
            n_jobs=mp.cpu_count(),
            random_state=1,
            base_estimator='RF',
            n_initial_points=1000,
            initial_point_generator='random'
        )
        epsf_result = run_optimizer(epsf_optimizer, epsf_obj, n_evaluations=1300)

        with open(name+'_epsf_opt.pkl', 'wb') as f:
            dill.dump(epsf_result, f)
        with open(name + 'epsf_params.txt', 'w') as f:
            f.write(str(list(zip(epsf_result.space.dimension_names, epsf_result.x))))
