import thesis_lib.scopesim_helper
from thesis_lib.astrometry.wrapper import Session
from astropy.modeling.fitting import TRFLSQFitter

from thesis_lib.scopesim_helper import make_anisocado_model
from thesis_lib.testdata.generators import read_or_generate_image
from thesis_lib.testdata.recipes import scopesim_groups
from thesis_lib import config
from thesis_lib.astrometry.plots import plot_image_with_source_and_measured, plot_xy_deviation
import numpy as np
import matplotlib.pyplot as plt
from thesis_lib.util import dictoflists_to_listofdicts, dict_to_kwargs
import multiprocess as mp
import pandas as pd
import timeit
from tqdm.auto import tqdm
from pathlib import Path
import os

# -*- coding: utf-8 -*-
# %%
# %matplotlib notebook
# %pylab

## use these for interactive, disable for export
plt.rcParams['figure.figsize'] = (9, 6)
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

# %%
outdir = './04_improving/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

# %%
config = config.Config()
config.use_catalogue_positions = True
config.separation_factor = 20
config.photometry_iterations = 1
config.perturb_catalogue_guess = 0.01

epsf = make_anisocado_model()

def benchmark_parameter_constraints(posbound, fluxbound, groupsize, seed=0):

    def recipe():
        return scopesim_groups(N1d=1, border=300, group_size=groupsize, group_radius=12, jitter=13,
                               magnitude=lambda N: np.random.normal(21, 2, N),
                               custom_subpixel_psf=epsf, seed=seed)

    config.bounds = {'x_0': (posbound, posbound), 'y_0': (posbound, posbound), 'flux_0': (fluxbound, fluxbound)}

    img, tab = read_or_generate_image(f'1x{groupsize}_groups_seed{seed}', config, recipe)
    session = Session(config, image=img, input_table=tab)
    session.fitter = TRFLSQFitter()
    session.epsf = epsf
    session.determine_psf_parameters()
    n_calls, time = timeit.Timer(lambda: session.do_astrometry()).autorange()
    return {'runtime': time/n_calls}

args = dictoflists_to_listofdicts(
        {'posbound': [0.02, 0.1, 0.2, 0.3, 0.5, 1.],
         'fluxbound': [1000, 5000, 10_000,  None],
         'groupsize': [3, 5, 8, 10, 12, 15, 20, 30],
         'seed': list(range(10))}
)

# %%
result_name = Path('cached_results/constraint_performance.pkl')
if result_name.exists():
    performance_data = pd.read_pickle(result_name)
else:
    #thesis_lib.scopesim_helper.download()
    #from thesis_lib.util import DebugPool
    #with DebugPool() as p:
    with mp.Pool() as p:
        recs = p.map(lambda arg: dict_to_kwargs(benchmark_parameter_constraints, arg), tqdm(args), 1)
    df = pd.DataFrame.from_records(recs)

