# %matplotlib notebook
# %pylab
import matplotlib.pyplot as plt

from photutils.psf.incremental_fit_photometry import *
from photutils import MMMBackground
import numpy as np
from astropy.table import Table
from thesis_lib.scopesim_helper import make_anisocado_model
from thesis_lib.testdata.generators import read_or_generate_image
from thesis_lib.scopesim_helper import download
from thesis_lib.testdata.recipes import scopesim_groups


## use these for interactive, disable for export
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10
plt.rcParams['figure.autolayout'] = True
from IPython.display import HTML
HTML('''
<style>
    .text_cell_render {
    font-size: 13pt;
    line-height: 135%;
    word-wrap: break-word;}
    
    .container { 
    min-width: 1200px;
    width:70% !important; 
    }}
</style>''')

model = make_anisocado_model()

# +
# tight group
def recipe():
    return scopesim_groups(N1d=1, border=500, group_size=8, group_radius=6, jitter=8,
                           magnitude=lambda N: rng.uniform(20.5, 21.5, size=N),
                           custom_subpixel_psf=model, seed=11)

img, table = read_or_generate_image('tight8group', recipe=recipe)
guess_table = table.copy()
guess_table.rename_columns(['x', 'y', 'f'], [xguessname, yguessname, fluxguessname])
guess_table['x_0'] += rng.uniform(-0.5, 0.5, len(guess_table))
guess_table['y_0'] += rng.uniform(-0.5, 0.5, len(guess_table))
guess_table.sort(fluxguessname, reverse=True)

fit_stages_tight = [FitStage(10, 0.001, 0.001, np.inf, all_individual), # first stage: flux is wildly off, get decent guess
              FitStage(10, 2., 2., 50_000, all_individual),
              FitStage(10, 1., 1., 20_000, brightest_simultaneous(3)),
              FitStage(10, 1., 1., 20_000, brightest_simultaneous(5)),
              FitStage(10, 0.5, 0.5, np.inf, all_simultaneous),
              FitStage(50, 0.2, 0.2, 10_000, all_simultaneous)
              ]

photometry = IncrementalFitPhotometry(MMMBackground(),
                                      model,
                                      max_group_size=100,
                                      group_extension_radius=10,
                                      fit_stages=fit_stages_tight)

result_table = photometry.do_photometry(img, guess_table)

# -

plt.figure()
plt.imshow(img)
plt.plot(table['x'], table['y'], 'gx')
plt.plot(result_table['x_fit'], result_table['y_fit'], 'ro')

# +
# isolated groups
rng = np.random.default_rng(seed=11)
def recipe():
    return scopesim_groups(N1d=2, border=200, group_size=8, group_radius=10, jitter=8,
                           magnitude=lambda N: rng.uniform(20.5, 21.5, size=N),
                           custom_subpixel_psf=model, seed=11)

img, table = read_or_generate_image('isolated4x8', recipe=recipe)
guess_table = table.copy()
guess_table.rename_columns(['x', 'y', 'f'], [xguessname, yguessname, fluxguessname])
guess_table['x_0'] += rng.uniform(-0.5, 0.5, len(guess_table))
guess_table['y_0'] += rng.uniform(-0.5, 0.5, len(guess_table))
guess_table.sort(fluxguessname, reverse=True)

fit_stages_tight = [FitStage(10, 0.001, 0.001, np.inf, all_individual), # first stage: flux is wildly off, get decent guess
              FitStage(10, 2., 2., 50_000, all_individual),
              FitStage(10, 1., 1., 20_000, brightest_simultaneous(3)),
              FitStage(10, 1., 1., 20_000, brightest_simultaneous(5)),
              FitStage(10, 0.5, 0.5, np.inf, all_simultaneous),
              FitStage(50, 0.2, 0.2, 10_000, all_simultaneous)
              ]

photometry = IncrementalFitPhotometry(MMMBackground(),
                                      model,
                                      max_group_size=100,
                                      group_extension_radius=20,
                                      fit_stages=fit_stages_tight)

result_table = photometry.do_photometry(img, guess_table)
# -

plt.figure()
plt.imshow(img)
plt.plot(table['x'], table['y'], 'gx')
plt.plot(result_table['x_fit'], result_table['y_fit'], 'ro')


