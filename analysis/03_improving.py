# %%
# %matplotlib notebook
# %pylab

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
outdir = './03_improving/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

# %% [markdown]
# # Runtime

# %%
# Definitions for generating data
cfg = config.Config()
cfg.use_catalogue_positions = True
cfg.separation_factor = 20
cfg.photometry_iterations = 1
cfg.perturb_catalogue_guess = 0.01
cfg.fitshape=51

epsf = make_anisocado_model()


def benchmark_parameter_constraints(posbound, fluxbound, groupsize, seed=0):
    def recipe():
        return scopesim_groups(N1d=1, border=300, group_size=groupsize, group_radius=12, jitter=13,
                               magnitude=lambda N: np.random.normal(21, 2, N),
                               custom_subpixel_psf=epsf, seed=seed)

    cfg.bounds = {'x_0': (posbound, posbound), 'y_0': (posbound, posbound), 'flux_0': (fluxbound, fluxbound)}

    img, tab = read_or_generate_image(f'1x{groupsize}_groups_seed{seed}', cfg, recipe)
    session = Session(cfg, image=img, input_table=tab)
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
# run/read
result_name = Path('cached_results/constraint_performance.pkl')
if result_name.exists():
    performance_data = pd.read_pickle(result_name)
else:
    #thesis_lib.scopesim_helper.download()
    #from thesis_lib.util import DebugPool
    #with DebugPool() as p:
    with mp.Pool() as p:
        recs = p.map(lambda arg: dict_to_kwargs(benchmark_parameter_constraints, arg), tqdm(args), 1)
    performance_data = pd.DataFrame.from_records(recs)
    performance_data.to_pickle(result_name)


# %%
fig=plt.figure()
fake_handles = []
labels = []

large_bound = performance_data[np.isnan(performance_data.fluxbound) & (performance_data.posbound>=0.5)]
small_bound = performance_data[~np.isnan(performance_data.fluxbound) & (performance_data.posbound<=0.1)]

grpd=large_bound.groupby('groupsize', as_index=False)
vp = plt.violinplot([group.runtime for _, group in grpd], [pos for pos, _ in grpd],
               showmedians=True, showextrema=False, widths=1.5)
fake_handles.append(vp['cmedians'])
labels.append('large bounds')

grpd=small_bound.groupby('groupsize', as_index=False)
vp = plt.violinplot([group.runtime for _, group in grpd], [pos for pos, _ in grpd],
               showmedians=True, showextrema=False, widths=1.5)

fake_handles.append(vp['cmedians'])
labels.append('small bounds')

xs = np.linspace(3,30,1000)
p=plt.plot(xs, 0.5*xs**2)
fake_handles.append(p[0])
labels.append(r'$0.5$ groupSize $^2$')

plt.legend(fake_handles, labels, loc='upper left')
plt.xlabel('size of fitted star-group')
plt.ylabel('runtime in seconds')
plt.yscale('log')


# %% [markdown]
# \todo{grouper finds connected components, not just all stars around target within radius}
#
# Considering the science case of astrometry in crowded fields a it is not at all unexpected to find a group of more than 30 stars that form a connected component within 2 FWHMs of the PSF. With the currently available Grouping and Fitting routines in photutils, this whole connected component is fit simultaneously. 
#
# To evaluate if a tight restriction on the fit-parameter can make this process viable \todo{we wanted this because diverging fits}, images containing isolated groups of sources are fit with a modified version of `BasicPSFPhotometry`. With the modifications in \todo{link branch for parameter_restriction} and using a TRF-Fitter one can specify constraints on the parameters.
#
# While initial experiments looked promising wrt. the quality of the results (i.e. no wandering sources) the run-time of the fits was concerning.
#
#
# It is obvious that scaling the simultaneous fit to groups beyond 30 stars is prohibitively expensive. The mean run-time increases with $\mathcal O(N^2)$ with a significant amount of outliers towards much longer run-times. It is suspected that these outliers are pathological cases in which the fit is stuck in a low-gradient local environment.
#
# One could set an aggressive limit on the fit iterations or introduce a timeout on a per-group basis. This would require manual intervention and tuning on a per-image basis and would make an automatic analysis of a single frame infeasible
#
# A strict restriction on the star positions and fluxes within the group can improve performance by a factor of about $5$ in the best cases, but given the quadratic run-time scaling, even a drastic constant improvement factor will not make the simultaneous fit feasible for crowded fields
#
# \todo{image for connected component}

# %% [markdown]
# # Qualitative analysis of restriction effect

# %%
def recipe():
    return scopesim_groups(N1d=8, border=50, group_size=1, group_radius=1, jitter=2,
                           magnitude=lambda N: np.random.normal(21, 2, N),
                           custom_subpixel_psf=epsf, seed=10)

def recipe():
    return scopesim_groups(N1d=13, border=70, group_size=1, group_radius=1, jitter=15,
                           magnitude=lambda N: [20.5]*N,
                           custom_subpixel_psf=epsf, seed=10)

cfg.fithshape=199
cfg.bounds = {'x_0': (0.3, 0.3), 'y_0': (0.3, 0.3), 'flux_0': (10000, 10000)}
cfg.niters = 1

img, tab = read_or_generate_image(f'169x1_group', cfg, recipe)
trialsession = Session(cfg, image=img, input_table=tab)
trialsession.fitter = TRFLSQFitter()
trialsession.epsf = epsf
trialsession.determine_psf_parameters()
trialsession.do_astrometry()

# %%
plot_image_with_source_and_measured(img, tab, trialsession.tables.result_table)
pass

# %%
plot_xy_deviation(trialsession.tables.result_table)
pass

# %% [markdown]
# ## Reinventing the wheel
#
# Severe performance bug when using RectBivariateSpline
