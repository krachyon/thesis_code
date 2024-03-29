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
from thesis_lib.util import make_gauss_kernel, save_plot, cached
import anisocado
import thesis_lib.util as util
from matplotlib.colors import LogNorm, SymLogNorm
import numpy as np
import matplotlib.pyplot as plt
import multiprocess as mp
from tqdm.auto import tqdm
import itertools
import pandas as pd
import pickle
from pathlib import Path
import os

plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12
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
cache_dir = Path('./cached_results/')
if not os.path.exists(outdir):
    os.mkdir(outdir)
#util.RERUN_ALL_CACHED = True

# %%
psf = make_anisocado_model(oversampling=4, degree=5)

# %%
seed = 10

def recipe():
    return scopesim_grid(N1d=8, perturbation=2., seed=seed,
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
    config.smoothing = make_gauss_kernel(σ=σ*oversampling/2, N=N*oversampling/2)
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


with mp.Pool(mp.cpu_count()-1) as p:
    epsfs = cached(lambda: p.starmap(make_epsf_with_sigma, tqdm(args), 1), cache_dir/'epsf_quality_samples',rerun=False)

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
fig, axs =plt.subplots(2,1, sharex=False, sharey=True)
axcom = fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
for (N,os), group in df.groupby(['N','oversampling']):
    i = 0 if os==2 else 1
    axs[i].plot(group['σ']*os/2, group['dev'], label=f'smoothing kernel size: {N*os/2}')
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
    empiric/=empiric.sum()
    actual/=actual.sum()
    
    diff = actual-empiric
    
    power_diff = np.fft.fftshift(np.abs(np.fft.fft2(actual))**2 - np.abs(np.fft.fft2(empiric))**2 )
    xfreq = np.fft.fftshift(np.fft.fftfreq(actual.data.shape[1]))
    yfreq = np.fft.fftshift(np.fft.fftfreq(actual.data.shape[0]))
    extent = (xfreq[0], xfreq[-1], yfreq[0], yfreq[-1])

    fig, axs = plt.subplots(1,3)

    im = axs[0].imshow(empiric)
    fig.colorbar(im, ax=axs[0], shrink=0.7)
    axs[0].set_title('EPSF')

    im = axs[1].imshow(diff)
    fig.colorbar(im, ax=axs[1], shrink=0.7)
    axs[1].set_title('EPSF-PSF')
    
    im = axs[2].imshow(power_diff, extent=extent)
    fig.colorbar(im, ax=axs[2], shrink=0.7)
    axs[2].set_title('Powerspectrum difference')

    fig.set_size_inches(10,3.5)
    fig.tight_layout()
    fig.suptitle(title, fontsize=15)
    return fig

# best 
best_idx = df[(df.N==11)&(df.oversampling==2)].dev.idxmin()
comparison_plot(epsfs[best_idx](x,y), 'Minimum RMS EPSF')
save_plot(outdir, 'epsf_residual_best')
# too much smoothing
best_idx = df[(df.N==11)&(df.oversampling==2)&(df.σ > 4)].dev.idxmin()
comparison_plot(epsfs[best_idx](x,y), 'Too much smoothing')
save_plot(outdir, 'epsf_residual_toomuch')
# not enough smoothing
best_idx = df[(df.N==11)&(df.oversampling==2)&(df.σ < 0.7)].dev.idxmin()
comparison_plot(epsfs[best_idx](x,y), 'Not enough smoothing')
save_plot(outdir, 'epsf_residual_notenough')
pass

# %%
#figure()
#y, x = np.mgrid[-50:50:101j,-50:50:101j]
#imshow(np.fft.fftshift(np.abs(np.fft.fft2(psf(x,y)))**2), norm=LogNorm())

# %%
base_psf = anisocado.AnalyticalScaoPsf(pixelSize=0.004, N=101, seed=10).shift_off_axis(0,14)
actual = psf(x, y)
empiric = epsfs[best_idx](x, y)

empiric /= empiric.sum()
actual /= actual.sum()
base_psf /= base_psf.sum()

vmin = np.min(np.hstack((actual, empiric, base_psf)))
vmax = np.max(np.hstack((actual, empiric, base_psf)))
print(vmin, vmax)
norm = SymLogNorm(linthresh=1e-12, vmin=vmin, vmax=vmax)
cmap = plt.cm.get_cmap('viridis')

xfreq = np.fft.fftshift(np.fft.fftfreq(actual.data.shape[1]))
yfreq = np.fft.fftshift(np.fft.fftfreq(actual.data.shape[0]))
extent = (xfreq[0], xfreq[-1], yfreq[0], yfreq[-1])

actual_mtf = np.fft.fftshift(np.abs(np.fft.fft2(actual)))**2
empiric_mtf = np.fft.fftshift(np.abs(np.fft.fft2(empiric)))**2
base_mtf = np.fft.fftshift(np.abs(np.fft.fft2(base_psf)))**2

fig, axs = plt.subplots(1, 3)
fig.set_size_inches(11, 3.5)

im = axs[0].imshow(actual_mtf, extent=extent, norm=LogNorm())
axs[0].set_title('PSF model')
plt.colorbar(im, ax=axs[0], shrink=0.7)

axs[1].add_artist(Circle((0.,0.),radius=0.333, lw=1.5, ec='red', fill=None, alpha=0.5))
im = axs[1].imshow(empiric_mtf, extent=extent, norm=LogNorm())
axs[1].set_title('ePSF')
plt.colorbar(im, ax=axs[1], shrink=0.7)

                
im = axs[2].imshow(base_mtf, extent=extent, norm=LogNorm())
plt.tight_layout()
axs[2].set_title('Raw AnisoCADO array')
plt.colorbar(im, ax=axs[2], shrink=0.7)

plt.suptitle('PSF powerspectra')
save_plot(outdir, 'epsf_psf_powerspectrums')

fig, axs = plt.subplots(1,3, sharey=True, sharex=True)
fig.set_size_inches(11, 3.5)

axs[0].plot(xfreq,actual_mtf[:,int(actual_mtf.shape[1]/2)])
axs[0].plot(xfreq,actual_mtf[int(actual_mtf.shape[0]/2),:])
axs[0].set_ylabel('fractional value')
axs[1].plot(xfreq,empiric_mtf[:,int(empiric_mtf.shape[1]/2)])
axs[1].plot(xfreq,empiric_mtf[int(empiric_mtf.shape[0]/2),:])
axs[2].plot(xfreq,base_mtf[:,int(base_mtf.shape[1]/2)], label='vertical profile')
axs[2].plot(xfreq,base_mtf[int(base_mtf.shape[0]/2),:], label='horizontal profile')
axs[2].legend()
for ax in axs:
    ax.set_xlabel('spatial frequencies')
plt.suptitle('1D profiles through powerspectra')

save_plot(outdir, 'epsf_psf_powerspectrums_cuts')


# %%
figure()
plt.imshow(base_psf, norm = LogNorm())

# %%
print("script run success")

# %%
