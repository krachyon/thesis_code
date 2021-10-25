# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] hide_input=false
# To use this notebook install `thesis_lib`:
# ```bash
# git clone https://github.com/krachyon/simcado_to_scopesim
# cd simcado_to_scopesim
# pip install -e .
# pip install -r requirements.txt
# ```
# %% hide_input=false
# use this for interactive plots
# %matplotlib notebook
# %pylab
# use this to export to pdf instead
# #%matplotlib inline

import os
from copy import copy

import astropy.table

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from matplotlib.colors import LogNorm

import thesis_lib
from thesis_lib import *
from thesis_lib.astrometry import plots
from thesis_lib.astrometry.types import INPUT_TABLE_NAMES, RESULT_TABLE_NAMES, X, Y, MAGNITUDE
from thesis_lib.astrometry.wrapper import Session
from thesis_lib.util import save_plot

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
outdir = './01_initial_astrometry'
if not os.path.exists(outdir):
    os.mkdir(outdir)
scopesim_helper.download()

# %%
# content of the module and default config
print([i for i in dir(thesis_lib) if not i.startswith('__')])
print('\n')
with open(os.path.join(outdir, 'default_config.txt'), 'w') as f:
    def_config = str(config.Config.instance())
    f.write(def_config)
print(def_config)

# %% [markdown]
# # initial attempt

# %%
# astrometry with default parameters on default scopesim image
default_config = config.Config()
default_config.create_dirs()
image_name_scc = 'scopesim_cluster'
#figure()
#imshow(image_scc, norm=LogNorm())

# %%
scc_session = Session(default_config, 'scopesim_cluster')
scc_session.do_it_all()
scc_session
clear_output()

# %%
# the cluster template generates a lot of very faint sources, only show the ones that could reasonably be detected. 
#
filtered_input_table = scc_session.tables.input_table[scc_session.tables.input_table[INPUT_TABLE_NAMES[MAGNITUDE]] > 1e-12]
fig = plots.plot_image_with_source_and_measured(
        scc_session.image,
        filtered_input_table,
        scc_session.tables.result_table)
plt.xlim(0, 1024)
plt.ylim(0, 1024)
cb = plt.colorbar(shrink=0.7)
cb.set_label('pixel count')
fig.set_size_inches(7, 7)
plt.tight_layout()

save_plot(outdir, 'standard_photutils')

# %% [markdown]
# # Grid

# %%
image_name_sg = 'scopesim_grid_16_perturb2'
sg_session = Session(default_config, image_name_sg)


# %%
#fig = plots_and_sanitycheck.plot_image_with_source_and_measured(
#    photometry_result_sg.image, photometry_result_sg.input_table, photometry_result_sg.result_table)

fig = plt.figure()
plt.imshow(sg_session.image, norm=LogNorm())
plt.title('')
cb = plt.colorbar(shrink=0.7)
cb.set_label('pixel count')
fig.set_size_inches(7,7)
#plt.plot(sg_session.tables.input_table['x'], sg_session.tables.input_table['y'], '.')
save_plot(outdir, 'photutils_grid')


# %% [markdown]
# # EPSF derivation

# %%

gauss_config = config.Config()
gauss_config.smoothing = util.make_gauss_kernel()
sg_gauss_session = Session(gauss_config, image_name_sg)

sg_session.find_stars().select_epsfstars_auto().make_epsf()
sg_gauss_session.find_stars().select_epsfstars_auto().make_epsf()

# %%
fig, axs = plt.subplots(1, 2)
im = axs[0].imshow(sg_session.epsf.data)
axs[0].set_title('quartic smoothing kernel')
plt.colorbar(im, ax=axs[0], shrink=0.5)
im = axs[1].imshow(sg_gauss_session.epsf.data)
axs[1].set_title('gaussian $σ=1$ smoothing kernel')
plt.colorbar(im, ax=axs[1], shrink=0.5)

save_plot(outdir, 'epsf_smoothing_comparison')

# %%
from astropy.modeling.functional_models import AiryDisk2D
screendoor = np.zeros((128, 128))
screendoor[::4, ::4] = 1
screendoor *= AiryDisk2D(radius=0.3)(*np.mgrid[-1:1:128j, -1:1:128j])
fig_a = plt.figure()
#plt.title('schematic representation of how a star image\n is interpolated into the EPSF grid, 4x4 oversampling', wrap=True)
plt.imshow(screendoor+0.02, norm=LogNorm())
save_plot(outdir, 'epsf_flygrid_a')

fig_b = plt.figure()
plt.imshow(2*np.roll(screendoor, (0, 0), axis=(0, 1)) +
          1*np.roll(screendoor, (0, 1), axis=(0, 1)) +
          1*np.roll(screendoor, (0, 2), axis=(0, 1)) +
          1*np.roll(screendoor, (0, 3), axis=(0, 1)) +
          1*np.roll(screendoor, (1, 0), axis=(0, 1)) +
          1*np.roll(screendoor, (1, 1), axis=(0, 1)) +
          3*np.roll(screendoor, (1, 2), axis=(0, 1)) +
          2*np.roll(screendoor, (1, 3), axis=(0, 1)) +
          1*np.roll(screendoor, (2, 0), axis=(0, 1)) +
          1*np.roll(screendoor, (2, 1), axis=(0, 1)) +
          3*np.roll(screendoor, (2, 2), axis=(0, 1)) +
          2*np.roll(screendoor, (2, 3), axis=(0, 1)) +
          1*np.roll(screendoor, (3, 0), axis=(0, 1)) +
          4*np.roll(screendoor, (3, 1), axis=(0, 1)) +
          3*np.roll(screendoor, (3, 2), axis=(0, 1)) +
          2*np.roll(screendoor, (3, 3), axis=(0, 1)) +
          0.01, norm=LogNorm())
#title('schematic EPSF with screendoor effect\n due to unfortunate sampling of stars', wrap=True)

save_plot(outdir, 'epsf_flygrid_b')

# %% [markdown]
# ## Cheating Astrometry

# %%
psf_effect_orig   = scopesim_helper.make_psf()
psf_effect_filter = scopesim_helper.make_psf(transform=testdata.helpers.lowpass())
ε = 1e-10 #prevent zeros in log
fig_a = plt.figure()
plt.imshow(psf_effect_orig.data+ε, norm=LogNorm())
plt.colorbar()
save_plot(outdir, 'psf_orig')

fig_b = plt.figure()
plt.imshow(psf_effect_filter.data+ε, norm=LogNorm())
plt.colorbar()
save_plot(outdir,'psf_lowpass')

# %%
# create a new configuration object with the necessary parameters for analyzing a lowpass filtered PS
lowpass_config = config.Config()
lowpass_config.smoothing = util.make_gauss_kernel()  # avoid EPSF artifacts
lowpass_config.output_folder = 'output_files_lowpass'
lowpass_config.use_catalogue_positions = True  # cheat with guesses
lowpass_config.photometry_iterations = 1  # with known positions we know all stars on first iter
lowpass_config.cutout_size = 20  # adapted by hand for PSF at hand
lowpass_config.fitshape = 15
lowpass_config.separation_factor = 2
lowpass_config.create_dirs()

# %%
# use an image with less sources to not have to filter the input positions
image_name_lpc = 'gausscluster_N2000_mag22'
session_lpc = Session(lowpass_config, image_name_lpc)
session_lpc.do_it_all()

# %%
#DEBUG
result_table = session_lpc.tables.result_table
result_table['x_fit']-result_table['x_0']

# %%
input_table = session_lpc.tables.input_table
result_table = session_lpc.tables.result_table

def ref_phot_plot():
    plt.figure()
    plt.imshow(session_lpc.image, norm=LogNorm())
    plt.plot(input_table[INPUT_TABLE_NAMES[X]], input_table[INPUT_TABLE_NAMES[Y]], 'o', fillstyle='none', 
             markeredgewidth=1, markeredgecolor='red', label=f'reference N={len(input_table)}')
    plt.plot(result_table[RESULT_TABLE_NAMES[X]], result_table[RESULT_TABLE_NAMES[Y]], '.', markersize=5,
             markeredgecolor='orange', label=f'photometry N={len(result_table)}')
    plt.legend()

ref_phot_plot()
plt.xlim((778.8170946421627, 823.9573452589827))
plt.ylim((508.11191403130636, 468.09311665069544))

save_plot(outdir, 'lowpass_astrometry_groupissue')

# %%
ref_phot_plot()
plt.xlim((66.41381370421695, 396.3265565090578))
plt.ylim((995.3013431030644, 644.769053872921))
save_plot(outdir, 'lowpass_astrometry')

# %%
fig = plots.plot_xy_deviation(session_lpc.tables.valid_result_table)

save_plot(outdir,'lowpass_astrometry_xy')

# %%
fig = plots.plot_deviation_vs_magnitude(session_lpc.tables.result_table)
plt.xlim(-16,-12)
save_plot(outdir, 'lowpass_astrometry_magvdev')

# %% [markdown]
# # Multiimage stats

# %%
# if something goes wrong here, check the stdout of the jupyter notebook server for hints
no_overlap_config = copy(lowpass_config)
no_overlap_config.separation_factor = 0.1  # No groups in this case
#no_overlap_config.detector_saturation=15000


def recipe_template(seed):
    def inner():
        # These imports are necessary to be able to execute in a forkserver context; it does not copy the full memory space, so
        # we'd have to rely on the target to know the imports
        from thesis_lib.testdata.recipes import scopesim_grid
        from thesis_lib.testdata.helpers import lowpass
        import numpy as np
        return scopesim_grid(seed=seed, N1d=25, perturbation=2., psf_transform=lowpass(), magnitude=lambda N: np.random.uniform(18, 24, N))
    return inner


sessions_multi = thesis_lib.astrometry.wrapper.photometry_multi(recipe_template, 'mag18-24_grid', n_images=12, config=no_overlap_config, threads=None)
result_table_multi = astropy.table.vstack([session.tables.result_table for session in sessions_multi])
clear_output()

# %%
fig = plots.plot_xy_deviation(result_table_multi)
save_plot(outdir, 'multi_astrometry_xy')

# %%
result_table_multi_recenter = result_table_multi.copy()
result_table_multi_recenter['offset']-= np.mean(result_table_multi_recenter['offset'])
fig = plots.plot_deviation_vs_magnitude(result_table_multi_recenter)
plt.ylim(-0.07,0.07)
plt.title(plt.gca().get_title()+' (subtracted systematic error)')
save_plot(outdir, 'multi_astrometry_mag')

# %%
result_table_multi

# %%
plt.figure()
result_table_multi.sort('m')

window_size = 101

dists = result_table_multi['offset']
dists_x = np.abs(result_table_multi['x_offset'])
dists_y = np.abs(result_table_multi['y_offset'])
magnitudes = result_table_multi['m']


magnitudes_mean= np.mean(np.lib.stride_tricks.sliding_window_view(magnitudes, window_size), axis=1)

dists_slide = np.lib.stride_tricks.sliding_window_view(dists, window_size)
dists_mean = np.mean(dists_slide, axis=1)

dists_slide_x = np.lib.stride_tricks.sliding_window_view(dists_x, window_size)
dists_mean_x = np.mean(dists_slide_x, axis=1)

dists_slide_y = np.lib.stride_tricks.sliding_window_view(dists_y, window_size)
dists_mean_y = np.mean(dists_slide_y, axis=1)

plt.title(f'smoothed with window size {window_size}')
plt.plot(magnitudes_mean, dists_mean, linewidth=1, color='green',
             label=r'measured euclidean deviation')
plt.plot(magnitudes_mean, dists_mean_x, linewidth=1, color='red',
             label=f'measured x-deviation')
plt.plot(magnitudes_mean, dists_mean_y, linewidth=1, color='orange',
             label=f'measured y-deviation')
plt.plot(magnitudes, result_table_multi['σ_pos_estimated'], 'bo',markersize=0.5, label='FWHM/SNR')

plt.xlabel('magnitude')
plt.ylabel('absolute centroid deviation [pixel]')
#plt.ylim(0,dists_std.max()*5)
plt.legend()
save_plot(outdir, 'multi_astrometry_noisefloor')
# %%
print('script run success')
