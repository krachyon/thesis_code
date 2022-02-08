# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %matplotlib notebook
# %pylab

from astropy.io.fits import getdata
from pathlib import Path
from matplotlib.colors import LogNorm, SymLogNorm, Normalize
from photutils import IRAFStarFinder, EPSFBuilder, extract_stars, \
 BasicPSFPhotometry, DAOGroup, MMMBackground, SExtractorBackground, FittableImageModel, StarFinder
from astropy.nddata import NDData
from astropy.stats import sigma_clipped_stats
import astropy.table as table
from thesis_lib.util import cached, save_plot
from thesis_lib import util
from thesis_lib.testdata.generators import read_or_generate_image
from thesis_lib.scopesim_helper import make_anisocado_model
from scipy.spatial import ConvexHull
from thesis_lib.testdata.recipes import scopesim_grid
# only on my branch
from photutils.psf.culler import CorrelationCuller
from photutils.psf.incremental_fit_photometry import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
## use these for interactive, disable for export
plt.rcParams['figure.figsize'] = (9, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10
plt.rcParams['figure.autolayout'] = True
from IPython.display import HTML
from thesis_lib.scopesim_helper import download
download()
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
#util.RERUN_ALL_CACHED = True
cache_dir = Path('./cached_results/')
out_dir = Path('./04_comparison')


# %%
def calc_devation(result_table):
    xdiff, ydiff = result_table['x_fit']-result_table['x_orig'], result_table['y_fit']-result_table['y_orig']
    eucdev = np.sqrt(xdiff**2+ydiff**2)
    return xdiff, ydiff, eucdev 

def xy_scatter(result_table):
    plt.figure(figsize=(7.7,6))
    plt.gca().set_aspect('equal')
    result_table.sort('flux_fit', reverse=False)

    xdiff, ydiff, eucdev = calc_devation(result_table)
    xstd, xstd_clipped = np.std(xdiff),sigma_clipped_stats(xdiff, sigma=3)[2]
    ystd, ystd_clipped = np.std(ydiff),sigma_clipped_stats(ydiff, sigma=3)[2]
    
    flux = result_table['flux_fit']
    norm = LogNorm(max(flux.min(),1),flux.max())
    cmap = mpl.cm.get_cmap('plasma').copy()
    cmap.set_under('green')

    plt.scatter(xdiff,ydiff, c=flux, norm=norm, cmap=cmap, s=20, alpha=0.8)
    plt.errorbar([0],[0],xerr=xstd,yerr=ystd, capsize=5, ls=':', color='tab:blue', label='σ')
    plt.errorbar([0],[0],xerr=xstd_clipped,yerr=ystd_clipped, capsize=5, ls=':', color='tab:green', label='σ clipped')
    plt.xlabel(f'x centroid deviation; $σ_x$={xstd:.4f}  $σ_x$ clipped={xstd_clipped:.4f}')
    plt.ylabel(f'y centroid deviation; $σ_y$={ystd:.4f}  $σ_y$ clipped={ystd_clipped:.4f}')

    plt.xlim(-1.3*xstd,1.3*xstd)
    plt.ylim(-1.3*xstd,1.3*xstd)

    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='log(flux)')
    plt.legend()
    

def image_and_sources(image, result_table):
    plt.figure()
    plt.imshow(image, norm=LogNorm())
    plt.plot(result_table['x_0'], result_table['y_0'], 'kx', label='guess positions')
    plt.scatter(result_table['x_fit'], result_table['y_fit'], c=result_table['flux_fit'],
                cmap='Oranges_r', label='fitted positions')
    plt.legend()
    

def visualize_grouper(image, input_table, max_size=20, halo_radius=25):
    table = input_table.copy()
    
    table.rename_columns(['x','y'], ['x_fit', 'y_fit'])
    table['update'] = True
    groups = make_groups(table, max_size=max_size, halo_radius=halo_radius)

    group_ids = np.sort(np.unique([i for group in groups for i in group['group_id']]))
    max_id = np.max(group_ids)

    plt.figure()
    for group in groups:
        core_group = group[group['update']==True]
        group_id = core_group['group_id'][0]

        cmap = plt.get_cmap('prism')

        xy_curr = np.array((group['x_fit'], group['y_fit'])).T

        if len(group)>=3:
            hull = ConvexHull(xy_curr)
            vertices = xy_curr[hull.vertices]
            poly=Polygon(vertices, fill=False, color=cmap(group_id/max_id))
            plt.gca().add_patch(poly)

        plt.scatter(core_group['x_fit'], core_group['y_fit'], color=cmap(group_id/max_id), s=8)

        plt.annotate(group_id, (np.mean(core_group['x_fit']),np.mean(core_group['y_fit'])), 
                     color='white', backgroundcolor=(0,0,0,0.25), alpha=0.7)
    
    plt.imshow(image, norm=LogNorm())
    
def plot_dev_vs_mag(result_table, window_size=60):
    eucdev = np.sqrt((result_table['x_fit']-result_table['x_orig'])**2 + (result_table['y_fit']-result_table['y_orig'])**2)
    flux = result_table['flux_fit']
    
    order = np.argsort(flux)

    dev_slide = np.lib.stride_tricks.sliding_window_view(eucdev[order], window_size)
    flux_slide = np.lib.stride_tricks.sliding_window_view(flux[order], window_size)
    
    plt.figure()
    plt.semilogy(-np.log(flux), eucdev, 'x', alpha=0.8, label='data points')
    plt.semilogy(-np.log(np.mean(flux_slide, axis=1)), np.mean(dev_slide, axis=1), 
                 label=f'running average, window size {window_size}')
    plt.xlabel('-log(flux)')
    plt.ylabel('centroid deviation')
    plt.legend()
    
def plot_residual(photometry,image,result_table):
    plt.figure()
    if photometry._last_image is None:
        photometry._last_image = image - photometry.bkg_estimator(image)
    residual = photometry.residual(result_table)
    plt.imshow(residual, norm=SymLogNorm(linthresh=10))
    plt.plot(result_table['x_fit'], result_table['y_fit'], 'rx', markersize=2, label='fit')
    plt.plot(result_table['x_orig'], result_table['y_orig'], 'k+', markersize=2, label='target')
    plt.colorbar()
    plt.legend()
    
def prepare_table(input_table, perturbation=0.5):
    rng = np.random.default_rng(seed=10)
    guess_table = input_table.copy()
    guess_table['x_orig'] = guess_table['x']
    guess_table['y_orig'] = guess_table['y']
    guess_table.rename_columns(['x', 'y', 'f'], [xguessname, yguessname, fluxguessname])
    guess_table['x_0'] += rng.uniform(-perturbation, perturbation, len(guess_table))
    guess_table['y_0'] += rng.uniform(-perturbation, perturbation, len(guess_table))
    guess_table.sort(fluxguessname, reverse=True)
    return guess_table


# %%
anisocado_psf = make_anisocado_model()


# %% [markdown]
# # Noisefloor

# %%
def do_photometry(seed):
    rng = np.random.default_rng(seed=seed)
    
    def recipe():
        return scopesim_grid(N1d=7, border=100, perturbation=7,
                               magnitude=lambda N: rng.uniform(15, 26, N),
                               custom_subpixel_psf=anisocado_psf, seed=seed)
    
    img_noisefloor, tab_noisefloor = read_or_generate_image(f'grid7_pert7_seed{seed}', recipe=recipe, force_generate=False)

    guess_table = tab_noisefloor.copy()
    guess_table['x_0'] = guess_table['x'].copy()
    guess_table['y_0'] = guess_table['y'].copy()
    guess_table['x_orig'] = guess_table['x'].copy()
    guess_table['y_orig'] = guess_table['y'].copy()
    guess_table['flux_0'] = guess_table['f']

    guess_table['x_0'] += rng.uniform(-0.1, 0.1, len(guess_table))
    guess_table['y_0'] += rng.uniform(-0.1, 0.1, len(guess_table))

    fit_stages = [FitStage(10, 1e-10, 1e-11, np.inf, all_individual), # first stage: get flux approximately right
                  FitStage(60, 0.5, 0.5, 10_000, all_individual),
                  FitStage(130, 0.1, 0.1, 5_000, all_individual),
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

    result_table = photometry.do_photometry(img_noisefloor, guess_table)
    result_table['id'] += seed*10000
    return result_table, photometry.residual(result_table)
    
results = cached(lambda: list(map(do_photometry, range(20))), cache_dir/'synthetic_noisefloor')

noisefloor_result = table.vstack([res[0] for res in results])
noisefloor_result = noisefloor_result[noisefloor_result['flux_fit']>=1]

# %%
xy_scatter(noisefloor_result)

# %%
plot_dev_vs_mag(noisefloor_result)


# %% [markdown]
# # Grid Best effort

# %%
def grid_photometry():
    fit_stages_grid = [FitStage(10, 1e-10, 1e-11, np.inf, all_individual), # first stage: get flux approximately right
                  FitStage(20, 0.5, 0.5, 100_000, all_individual),
                  FitStage(30, 0.1, 0.1, 10_000, all_individual),
                  FitStage(30, 0.1, 0.1, 5_000, all_simultaneous)
                 ]


    photometry_grid = IncrementalFitPhotometry(SExtractorBackground(),
                                               anisocado_psf,
                                               max_group_size=4,
                                               group_extension_radius=100,
                                               fit_stages=fit_stages_grid,
                                               use_noise=True)
    return photometry_grid

# %%
img_grid, tab_grid = read_or_generate_image('scopesim_grid_16_perturb2_mag18_24_subpixel')

guess_table = prepare_table(tab_grid)

photometry = grid_photometry()
grid_result = cached(lambda: photometry.do_photometry(img_grid, guess_table), cache_dir/'synthetic_grid', rerun=False)
[(np.min(np.abs(i)),np.mean(np.abs(i)),np.max(np.abs(i))) for i in calc_devation(grid_result)]

# %%
visualize_grouper(img_grid, tab_grid, 4, 90)

# %%
plot_residual(photometry, img_grid, grid_result)

# %%
xy_scatter(grid_result)

# %%
plot_dev_vs_mag(grid_result)


# %% [markdown]
# # Grid no crowding

# %%
def do_photometry(seed):
    rng = np.random.default_rng(seed=seed)
    def recipe():
        return scopesim_grid(N1d=8, border=50, perturbation=2,
                               magnitude=lambda N: rng.uniform(18, 24, N),
                               custom_subpixel_psf=anisocado_psf, seed=seed)
    img_grid_nocrowd, tab_grid_nowcrowd = read_or_generate_image(f'grid8_pert2_seed{seed}', recipe=recipe, force_generate=False)
    #img_grid, tab_grid = read_or_generate_image('scopesim_grid_16_perturb2_mag18_24_subpixel')

    guess_table = prepare_table(tab_grid_nowcrowd)

    photometry = grid_photometry()
    result_table = photometry.do_photometry(img_grid_nocrowd, guess_table)
    return result_table, photometry

nocrowd_results = cached(lambda: [do_photometry(i) for i in range(4)], cache_dir/'synthetic_grid_nocrowd', rerun=False)
grid_result_nocrowd = table.vstack([table for table, photometry in nocrowd_results])

[(np.min(np.abs(i)),np.mean(np.abs(i)),np.max(np.abs(i))) for i in calc_devation(grid_result_nocrowd)]

# %%
xy_scatter(grid_result_nocrowd)

# %%
plot_dev_vs_mag(grid_result_nocrowd)


# %% [markdown]
# # grid no saturation

# %%
def recipe():
    return scopesim_grid(N1d=16, border=50, perturbation=2,
                           magnitude=lambda N: rng.uniform(19.5, 21, N),
                           custom_subpixel_psf=anisocado_psf, seed=11)
img_grid_nosat, tab_grid_nosat = read_or_generate_image('grid16_pert2_mag195_21', recipe=recipe, force_generate=False)
#img_grid, tab_grid = read_or_generate_image('scopesim_grid_16_perturb2_mag18_24_subpixel')

guess_table = prepare_table(tab_grid_nosat)

photometry = grid_photometry()

grid_result_nosat = cached(lambda: photometry.do_photometry(img_grid_nosat, guess_table), cache_dir/'synthetic_grid_nosat', rerun=False)

[(np.min(np.abs(i)),np.mean(np.abs(i)),np.max(np.abs(i))) for i in calc_devation(grid_result_nosat)]

# %%
xy_scatter(grid_result_nosat)

# %%
plot_dev_vs_mag(grid_result_nosat)

# %% [markdown]
# # Cluster Best effort

# %%
rng = np.random.default_rng(seed=10)

img_gausscluster, tab_gausscluster = read_or_generate_image('gausscluster_N2000_mag22_subpixel')

guess_table = prepare_table(tab_gausscluster, perturbation=1)

fit_stages = [FitStage(1, 1e-11, 1e-11, np.inf, all_individual), # first stage: get flux approximately right              
              FitStage(1, 0.6, 0.6, 200_000, brightest_simultaneous(2)),
              FitStage(1, 0.6, 0.6, 100_000, not_brightest_individual(2)),
              FitStage(1, 0.3, 0.3, 100_000, brightest_simultaneous(4)),
              FitStage(1, 0.3, 0.3, 50_000, not_brightest_individual(4)),
              FitStage(20, 0.2, 0.2, 20_000, all_individual),
              FitStage(20, 0.1, 0.1, 5000, all_simultaneous)
             ]


photometry = IncrementalFitPhotometry(SExtractorBackground(),
                                           anisocado_psf,
                                           max_group_size=12,
                                           group_extension_radius=20,
                                           fit_stages=fit_stages,
                                           use_noise=True)

cluster_result = cached(lambda: photometry.do_photometry(img_gausscluster, guess_table), cache_dir/'synthetic_gausscluster', rerun=False)
[(np.min(np.abs(i)),np.mean(np.abs(i)),np.max(np.abs(i))) for i in calc_devation(cluster_result)]

# %%
[(np.min(np.abs(i)),np.mean(np.abs(i)),np.max(np.abs(i))) for i in calc_devation(cluster_result)]

# %%
if photometry._last_image is None:
    photometry._last_image = image - photometry.bkg_estimator(image)
#residual = photometry.residual(cluster_result)

fig = plt.figure(figsize=(9,12))
gs = GridSpec(5, 2, figure=fig)

norm=SymLogNorm(linthresh=10, vmin=residual.min(), vmax=residual.max())
cmap=plt.cm.get_cmap('viridis')


tl = fig.add_subplot(gs[:2,0])
tl.imshow(residual, norm=norm, cmap=cmap)
tl.set_title('residual')

bl = fig.add_subplot(gs[2:4,0])
im = bl.imshow(img_gausscluster, cmap='viridis', norm=LogNorm())
bl.set_title('original image')

tr = fig.add_subplot(gs[:2,1])
tr.plot(cluster_result['x_fit'], cluster_result['y_fit'], 'rx', markersize=3, label='fit')
tr.plot(cluster_result['x_orig'], cluster_result['y_orig'], 'k+', markersize=3, label='input position')
tr.imshow(residual, norm=norm, cmap=cmap)
tr.set_xlim(650,780)
tr.set_ylim(850,720)
tr.set_title('bright source halo')
tr.legend()

br = fig.add_subplot(gs[2:4,1])
br.plot(cluster_result['x_fit'], cluster_result['y_fit'], 'rx', markersize=3, label='fit')
br.plot(cluster_result['x_orig'], cluster_result['y_orig'], 'k+', markersize=3, label='input position')
br.imshow(residual, norm=norm, cmap=cmap)
br.set_title('fainter close sources')
br.set_xlim(470,540)
br.set_ylim(490,420)
br.legend()

cbarax = fig.add_subplot(gs[4,:])
cbarax.set_axis_off()
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             orientation='horizontal', ax=cbarax,location='top',shrink=1, label='residuals',  pad=0.25)
fig.colorbar(im, orientation='horizontal', ax=cbarax, location='top',shrink=1,label='image', pad=0.35, ticklocation='bottom')

#fig.tight_layout()
save_plot(out_dir, 'synthetic_cluster_overview')

# %%
image_and_sources(img_gausscluster, cluster_result)

# %%
xy_scatter(cluster_result)

# %%
visualize_grouper(img_gausscluster, tab_gausscluster, 12, 20)

# %%
plot_dev_vs_mag(cluster_result[cluster_result['flux_fit']>10000])

# %% [markdown]
# # Cluster without cheating

# %%
