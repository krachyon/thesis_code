# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# use this for interactive plots
# %matplotlib notebook
# %pylab
# use this to export to pdf instead
# #%matplotlib inline
from thesis_lib import *
import thesis_lib
from pprint import pprint
from matplotlib.colors import LogNorm
import multiprocess as mp
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from photutils.psf import EPSFModel
from copy import copy
from IPython.display import clear_output
import thesis_lib.config as config
from thesis_lib.util import save_plot, match_observation_to_source, make_gauss_kernel
from thesis_lib.scopesim_helper import make_anisocado_model
from thesis_lib.testdata_generators import model_add_grid, read_or_generate_image
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
outdir = './progress_subpixel'
if not os.path.exists(outdir):
    os.mkdir(outdir)
scopesim_helper.download()

# %%
lowpass_config = config.Config()
lowpass_config.smoothing = 'quartic'
lowpass_config.output_folder = 'output_files_lowpass'
lowpass_config.use_catalogue_positions = True  # cheat with guesses
lowpass_config.photometry_iterations = 1  # with known positions we know all stars on first iter
lowpass_config.cutout_size = 20  # adapted by hand for PSF at hand
lowpass_config.fitshape = 15
lowpass_config.create_dirs()

# %%
image_name_lpc = 'gausscluster_N2000_mag22_lowpass_subpixel'
image_recipe_lpc = testdata_generators.benchmark_images[image_name_lpc]
image_lpc, input_table_lpc = testdata_generators.read_or_generate_image(image_recipe_lpc,
                                                                        image_name_lpc,
                                                                        lowpass_config.image_folder)

photometry_result_lpc = photometry.run_photometry(
    image_lpc, input_table_lpc, image_name_lpc, lowpass_config)

# %%
image_name_lpsub = 'gausscluster_N2000_mag22_lowpass_subpixel'
image_recipe_lpsub = testdata_generators.benchmark_images[image_name_lpsub]
image_lpsub, input_table_lpsub = testdata_generators.read_or_generate_image(image_recipe_lpsub,
                                                                        image_name_lpsub,
                                                                        lowpass_config.image_folder)

photometry_result_lpsub = photometry.run_photometry(
    image_lpsub, input_table_lpsub, image_name_lpsub, lowpass_config)


# %%
def ref_phot_plot(photometry_result):
    plt.figure()
    plt.imshow(photometry_result.image, norm=LogNorm())
    plt.plot(photometry_result.input_table['x'], photometry_result.input_table['y'], 'o', fillstyle='none', 
             markeredgewidth=1, markeredgecolor='red', label=f'reference N={len(input_table)}')
    plt.plot(photometry_result.result_table['x_fit'], photometry_result.result_table['y_fit'], '.', markersize=5,
             markeredgecolor='orange', label=f'photometry N={len(result_table)}')
    plt.legend()


# %%
ref_phot_plot(photometry_result_lpc)

# %%

ref_phot_plot(photometry_result_lpsub)

# %%
fig,axs = plt.subplots(1,2)
axs[0].imshow(photometry_result_lpc.epsf.data)
axs[1].imshow(photometry_result_lpsub.epsf.data)

# %%
tab_lpsub = match_observation_to_source(photometry_result_lpsub.input_table, photometry_result_lpsub.result_table)
tab_lpc = match_observation_to_source(photometry_result_lpc.input_table, photometry_result_lpc.result_table)

# %%
figure()
plt.scatter(tab_lpsub['x_offset'], tab_lpsub['y_offset'], alpha=0.8, label='lp_sub')
plt.scatter(tab_lpc['x_offset'], tab_lpc['y_offset'], alpha=0.8, label='lp_conv')
plt.legend()

# %%
np.mean(tab_lpsub['offset']), np.mean(tab_lpc['offset']), np.std(tab_lpsub['offset']), np.std(tab_lpc['offset']), 

# %%
multi_config = config.Config()
multi_config.smoothing = 'quartic'  # avoid EPSF artifacts
multi_config.use_catalogue_positions = True  # cheat with guesses
multi_config.photometry_iterations = 1  # with known positions we know all stars on first iter
multi_config.cutout_size = 30  # adapted by hand for PSF at hand
multi_config.fitshape = 15
multi_config.create_dirs()


def recipe_template(seed):
    def inner():
        # These imports are necessary to be able to execute in a forkserver context;
        # it does not copy the full memory space, so
        # we'd have to rely on the target to know the imports
        from thesis_lib.testdata_generators import gaussian_cluster_modeladd
        from thesis_lib.scopesim_helper import make_anisocado_model
        import numpy as np
        psf_model=make_anisocado_model()
        return gaussian_cluster_modeladd(N=1000, seed=seed, magnitude=lambda N: np.random.normal(21, 2, N))
    return inner

result_table_multi = astrometry_benchmark.photometry_multi(recipe_template, 'gausscluster_modeladd',
                                                           n_images=12, config=multi_config, threads=None)
clear_output()

# %%
fig = astrometry_plots.plot_xy_deviation(result_table_multi)
save_plot(outdir, 'multi_astrometry_xy')

# %%
result_table_multi_recenter = result_table_multi.copy()
result_table_multi_recenter['offset']-= np.mean(result_table_multi_recenter['offset'])
fig = astrometry_plots.plot_deviation_vs_magnitude(result_table_multi_recenter)
plt.ylim(-0.07,0.07)
plt.title(plt.gca().get_title()+' (subtracted systematic error)')
save_plot(outdir, 'multi_astrometry_mag')

# %%
figure()

magnitudes = result_table_multi['m']
window_size = 101
x_offset = int(window_size/2)
order = np.argsort(magnitudes)

dists = result_table_multi['offset'][order]
dists_x = result_table_multi['x_orig'][order] - result_table_multi['x_fit'][order]
dists_y = result_table_multi['y_orig'][order] - result_table_multi['y_fit'][order]

# subtract out systematics
dists -= np.mean(dists)
dists_x -= np.mean(dists_x)
dists_y -= np.mean(dists_y)

magnitudes = magnitudes[order]

dists_slide = np.lib.stride_tricks.sliding_window_view(dists, window_size)
dists_std = np.std(dists_slide, axis=1)

dists_slide_x = np.lib.stride_tricks.sliding_window_view(dists_x, window_size)
dists_std_x = np.std(dists_slide_x, axis=1)

dists_slide_y = np.lib.stride_tricks.sliding_window_view(dists_y, window_size)
dists_std_y = np.std(dists_slide_y, axis=1)

plt.title(f'smoothed with window size {window_size}')
plt.plot(magnitudes[x_offset:-x_offset], dists_std, linewidth=1, color='green',
             label=r'measured euclidean deviation')
plt.plot(magnitudes[x_offset:-x_offset], dists_std_x, linewidth=1, color='red',
             label=f'measured x-deviation')
plt.plot(magnitudes[x_offset:-x_offset], dists_std_y, linewidth=1, color='orange',
             label=f'measured y-deviation')
plt.plot(result_table_multi['m'], result_table_multi['σ_pos_estimated'], 'bo',markersize=0.5, label='FWHM/SNR')

plt.xlabel('magnitude')
plt.ylabel('deviation [pixel]')
#plt.ylim(0,dists_std.max()*5)
plt.legend()
save_plot(outdir, 'multi_astrometry_noisefloor')

# %%
model = make_anisocado_model()
img_name_modelgrid = 'model_add_grid_6'
recipe_modelgrid = lambda: model_add_grid(model, N1d=6, noise_σ=10, perturbation=1)

img_modelgrid, tab_modelgrid = read_or_generate_image(recipe_modelgrid, img_name_modelgrid)
img_modelgrid -= np.min(img_modelgrid)+1

# %%
modelgrid_config = config.Config()
modelgrid_config.smoothing = make_gauss_kernel()
modelgrid_config.oversampling = 4
modelgrid_config.use_catalogue_positions = True  # cheat with guesses
modelgrid_config.photometry_iterations = 1  # with known positions we know all stars on first iter
modelgrid_config.cutout_size = 41  # adapted by hand for PSF at hand
modelgrid_config.fitshape = 35
modelgrid_config.epsfbuilder_iters = 7
modelgrid_config.separation_factor = 2
modelgrid_config.create_dirs()

# %%
photometry_result_modelgrid = photometry.run_photometry(
    img_modelgrid, tab_modelgrid, img_name_modelgrid, modelgrid_config)
modelgrid_matched = match_observation_to_source(photometry_result_modelgrid.input_table, photometry_result_modelgrid.result_table)

# %%
fig = astrometry_plots.plot_xy_deviation(modelgrid_matched)

# %%
astrometry_plots.plot_image_with_source_and_measured(img_modelgrid, tab_modelgrid, modelgrid_matched)

# %%
figure()
plt.imshow(photometry_result_modelgrid.epsf.data)

# %%
modelgrid_matched

# %%
