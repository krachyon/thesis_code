# -*- coding: utf-8 -*-
# %%
# %matplotlib notebook
# %pylab
import matplotlib.pyplot as plt
import numpy as np

# %%
from thesis_lib.testdata.generators import read_or_generate_image
from thesis_lib.scopesim_helper import make_anisocado_model
from thesis_lib.astrometry import wrapper
from thesis_lib.config import Config
from thesis_lib.testdata.recipes import scopesim_grid
from thesis_lib.util import make_gauss_kernel, save_plot
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import multiprocess as mp
from tqdm.auto import tqdm
import itertools
import pandas as pd
import pickle
from pathlib import Path
import os
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

# %%
outdir = './02_gridded_models/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

# %%
psf = make_anisocado_model(oversampling=4, degree=5)

# %%
seed = 10

def recipe():
    return scopesim_grid(N1d=8, perturbation=2., seed=10,
                  magnitude=lambda N: np.random.uniform(21.5, 23, N),
                  custom_subpixel_psf=psf)


config = Config()
config.oversampling = 2
config.max_epsf_stars = 1000
config.cutout_size=101
config.epsfbuilder_iters = 5

config.use_catalogue_positions = True
config.smoothing = make_gauss_kernel(σ=0.32)
#config.smoothing = 'quartic'
#config.smoothing = 'quadratic'

img, table = read_or_generate_image(f'grid_N_seed{seed}', config, recipe=recipe, force_generate=False)
#session = wrapper.Session(config, img, table)
#session.select_epsfstars_auto().make_epsf()
#pass


# %%
def make_epsf_with_sigma(config, σ, N, oversampling):
    config = config.copy()
    config.smoothing = make_gauss_kernel(σ=σ, N=N)
    config.oversampling = oversampling
    session = wrapper.Session(config, img, table)
    try: 
        session.select_epsfstars_auto().make_epsf()
        return session.epsf
    except:
        return None

cfgs = [config]
σs=np.linspace(0.25, 5, 100)
Ns=[5,11]
oss=[2,4]
args = list(itertools.product(cfgs, σs, Ns, oss))

cachename = Path('cached_results/epsf_quality_samples.pkl')
if cachename.exists():
    with open(cachename, 'rb') as f:
        epsfs = pickle.load(f)
else:
    #from thesis_lib.util import DebugPool
    #with DebugPool() as p:
    with mp.Pool() as p:
        epsfs = p.starmap(make_epsf_with_sigma, tqdm(args))
    with open(cachename, 'wb') as f:
        pickle.dump(epsfs, f)

# %%
y, x = np.mgrid[-50:50:101j,-50:50:101j]
actual = psf(x,y)
actual /= actual.max()

recs = []
for (_, σ, N, os), epsf in zip(args, epsfs):
    if epsf is not None:
        empiric = epsf(x,y)
        empiric/=empiric.max()

        diff = empiric-actual
        dev = np.sqrt(np.sum(diff**2))
    else:
        dev = np.inf
    recs.append({'dev': dev, 'σ': σ, 'N': N, 'oversampling': os})

df = pd.DataFrame.from_records(recs)                

# %%
fig, axs =plt.subplots(2,1, sharex=True, sharey=True)
axcom = fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
for (N,os), group in df.groupby(['N','oversampling']):
    i = 0 if os==2 else 1
    axs[i].plot(group['σ'], group['dev'], label=f'smoothing kernel size: {N}')
    axs[i].set_yscale('log')

axcom.set_ylabel('RMS residual between EPSF and input PSF on 100x100 pixel grid')
axcom.set_xlabel('standard deviation of gaussian smoothing kernel')

axs[0].set_title('oversampling=2')
axs[0].legend()
axs[1].set_title('oversampling=4')
axs[1].legend()
save_plot(outdir, 'epsf_quality')

# %%
# residual between real PSF and EPSF
y, x = np.mgrid[-50:50:101j,-50:50:101j]
def comparison_plot(empiric, title):
    actual = psf(x,y)
    empiric/=empiric.max()
    actual/=actual.max()

    diff = actual-empiric

    fig, axs = plt.subplots(1,2)

    im = axs[0].imshow(empiric)
    fig.colorbar(im, ax=axs[0])
    axs[0].set_title('derived EPSF')

    im = axs[1].imshow(diff)
    fig.colorbar(im, ax=axs[1])
    axs[1].set_title('EPSF subtracted from PSF')

    fig.set_size_inches(8,3.5)
    fig.suptitle(title)
    return fig

# best 
best_idx = df[(df.N==11)&(df.oversampling==2)].dev.idxmin()
comparison_plot(epsfs[best_idx](x,y), 'best EPSF')
save_plot(outdir, 'epsf_residual_best')
# too much smoothing
best_idx = df[(df.N==11)&(df.oversampling==2)&(df.σ > 4)].dev.idxmin()
comparison_plot(epsfs[best_idx](x,y), 'too much smoothing')
save_plot(outdir, 'epsf_residual_toomuch')
# not enough smoothing
best_idx = df[(df.N==11)&(df.oversampling==2)&(df.σ < 0.7)].dev.idxmin()
comparison_plot(epsfs[best_idx](x,y), 'not enough smoothing')
save_plot(outdir, 'epsf_residual_notenough')
pass

# %%
print("script run success")
