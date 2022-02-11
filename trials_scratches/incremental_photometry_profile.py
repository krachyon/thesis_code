import matplotlib.pyplot as plt
from photutils.psf.incremental_fit_photometry import *
from photutils import MMMBackground
import numpy as np
from thesis_lib.scopesim_helper import make_anisocado_model
from thesis_lib.testdata.generators import read_or_generate_image
from thesis_lib.testdata.recipes import scopesim_groups, scopesim_grid

from photutils import IRAFStarFinder, EPSFBuilder, extract_stars, \
 BasicPSFPhotometry, DAOGroup, MMMBackground, SExtractorBackground, FittableImageModel, StarFinder

anisocado_psf = make_anisocado_model()

rng = np.random.default_rng(seed=12)
def recipe():
    return scopesim_grid(N1d=7, border=100, perturbation=7,
                           magnitude=lambda N: rng.uniform(20, 24, N),
                           custom_subpixel_psf=anisocado_psf, seed=11)
img_grid, tab_grid = read_or_generate_image('grid7_pert7', recipe=recipe, force_generate=False)

guess_table = tab_grid.copy()
guess_table['x_0'] = guess_table['x'].copy()
guess_table['y_0'] = guess_table['y'].copy()
guess_table['x_orig'] = guess_table['x'].copy()
guess_table['y_orig'] = guess_table['y'].copy()
guess_table['flux_0'] = guess_table['f']

guess_table['x_0'] += rng.uniform(-0.02, 0.02, len(guess_table))
guess_table['y_0'] += rng.uniform(-0.02, 0.02, len(guess_table))

fit_stages = [FitStage(10, 1e-10, 1e-11, np.inf, all_individual), # first stage: get flux approximately right
              FitStage(120, 0.5, 0.5, 200, all_individual),
              #FitStage(10, 0.1, 0.1, 1, all_individual)
              #FitStage(10, 0.3, 0.3, 1, all_individual), # optimize position, keep flux constant
              #FitStage(10, 0.2, 0.2, 5000, all_individual),
              #FitStage(30, 0.05, 0.05, 100, all_individual)
             ]


photometry = IncrementalFitPhotometry(SExtractorBackground(),
                                           anisocado_psf,
                                           max_group_size=1,
                                           group_extension_radius=50,
                                           fit_stages=fit_stages,
                                           use_noise=True)

grid_result = photometry.do_photometry(img_grid, guess_table)
print(grid_result['flux_fit'])
pass