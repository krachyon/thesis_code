# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
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
from thesis_lib.util import save_plot
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
image_recipe_scc = testdata_generators.misc_images[image_name_scc]
image_scc, input_table_scc = testdata_generators.read_or_generate_image(image_recipe_scc, 
                                                                        image_name_scc, 
                                                                        default_config.image_folder)
#figure()
#imshow(image_scc, norm=LogNorm())

# %%
photometry_result_scc = photometry.run_photometry(image_scc, input_table_scc, image_name_scc, default_config)
clear_output()

# %%
# the cluster template generates a lot of very faint sources, only show the ones that could reasonably be detected
filtered_table_scc = photometry_result_scc.input_table[photometry_result_scc.input_table['weight']>1e-12]
fig = plots_and_sanitycheck.plot_image_with_source_and_measured(
    photometry_result_scc.image, filtered_table_scc, photometry_result_scc.result_table)
plt.xlim(0,1024)
plt.ylim(0,1024)
cb = plt.colorbar(shrink=0.7)
cb.set_label('pixel count')
fig.set_size_inches(7,7)
plt.tight_layout()

save_plot(outdir, 'standard_photutils')

# %% [markdown]
# # Grid

# %%
image_name_sg = 'scopesim_grid_16_perturb2'
image_recipe_sg = testdata_generators.misc_images[image_name_sg]
image_sg, input_table_sg = testdata_generators.read_or_generate_image(image_recipe_sg,
                                                                      image_name_sg,
                                                                      default_config.image_folder)
photometry_result_sg = photometry.run_photometry(
    image_sg, input_table_sg, image_name_sg, default_config)


# %%
#fig = plots_and_sanitycheck.plot_image_with_source_and_measured(
#    photometry_result_sg.image, photometry_result_sg.input_table, photometry_result_sg.result_table)

fig = plt.figure()
plt.imshow(photometry_result_sg.image, norm=LogNorm())
plt.title('')
cb = plt.colorbar(shrink=0.7)
cb.set_label('pixel count')
fig.set_size_inches(7,7)
save_plot(outdir, 'photutils_grid')

# %% [markdown]
# # EPSF derivation

# %%
import photutils
from photutils.detection import DAOStarFinder

gauss_config = config.Config()

mean, median, std = sigma_clipped_stats(image_sg, sigma=gauss_config.clip_sigma)
threshold = median + gauss_config.threshold_factor * std

finder = DAOStarFinder(threshold=threshold, fwhm=gauss_config.fwhm_guess)
star_guesses = photometry.make_stars_guess(image_sg, finder, cutout_size=gauss_config.cutout_size)[:gauss_config.stars_to_keep]

smooth_epsf = photometry.make_epsf_fit(
    star_guesses, iters=5, oversampling=4, smoothing_kernel=util.make_gauss_kernel())

# %%
fig, axs = plt.subplots(1,2)
im=axs[0].imshow(photometry_result_sg.epsf.data)
axs[0].set_title('quartic smoothing kernel')
plt.colorbar(im, ax=axs[0], shrink=0.5)
im=axs[1].imshow(smooth_epsf.data)
axs[1].set_title('gaussian $σ=1$ smoothing kernel')
plt.colorbar(im, ax=axs[1], shrink=0.5)

save_plot(outdir, 'epsf_smoothing_comparison')

# %%
from astropy.modeling.functional_models import AiryDisk2D
screendoor = np.zeros((128, 128))
screendoor[::4, ::4] = 1
screendoor *= AiryDisk2D(radius=0.3)(*mgrid[-1:1:128j, -1:1:128j])
fig_a = figure()
#plt.title('schematic representation of how a star image\n is interpolated into the EPSF grid, 4x4 oversampling', wrap=True)
plt.imshow(screendoor+0.02, norm=LogNorm())
save_plot(outdir, 'epsf_flygrid_a')

fig_b=plt.figure()
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
psf_effect_filter = scopesim_helper.make_psf(transform=testdata_generators.lowpass())
ε=1e-10 #prevent zeros in log 
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
lowpass_config.create_dirs()

# %%
# use an image with less sources to not have to filter the input positions
image_name_lpc = 'gausscluster_N2000_mag22'
image_recipe_lpc = testdata_generators.lowpass_images[image_name_lpc]
image_lpc, input_table_lpc = testdata_generators.read_or_generate_image(image_recipe_lpc,
                                                                        image_name_lpc,
                                                                        lowpass_config.image_folder)

photometry_result_lpc = photometry.run_photometry(
    image_lpc, input_table_lpc, image_name_lpc, lowpass_config)

# %%
input_table = photometry_result_lpc.input_table
result_table=photometry_result_lpc.result_table
def ref_phot_plot():
    plt.figure()
    plt.imshow(photometry_result_lpc.image, norm=LogNorm())
    plt.plot(input_table['x'], input_table['y'], 'o', fillstyle='none', 
             markeredgewidth=1, markeredgecolor='red', label=f'reference N={len(input_table)}')
    plt.plot(result_table['x_fit'], result_table['y_fit'], '.', markersize=5,
             markeredgecolor='orange', label=f'photometry N={len(result_table)}')
    plt.legend()

ref_phot_plot()
plt.xlim((333.0414029362044, 371.33596286811064))
plt.ylim((280.9350859599385, 244.97388631062154))

save_plot(outdir, 'lowpass_astrometry_groupissue')

# %%
ref_phot_plot()
xlim((66.41381370421695, 396.3265565090578))
ylim((995.3013431030644, 644.769053872921))
save_plot(outdir, 'lowpass_astrometry')

# %%
matched_result_lpc = util.match_observation_to_source(photometry_result_lpc.input_table, photometry_result_lpc.result_table)
fig = plots_and_sanitycheck.plot_input_vs_photometry_positions(matched_result_lpc)

save_plot(outdir,'lowpass_astrometry_xy')

# %%
fig = plots_and_sanitycheck.plot_deviation_vs_magnitude(matched_result_lpc)
plt.xlim(-16,-12)
save_plot(outdir, 'lowpass_astrometry_magvdev')

# %% [markdown]
# # Multiimage stats

# %%
# if something goes wrong here, check the stdout of the jupyter notebook server for hints
no_overlap_config = copy(lowpass_config)
no_overlap_config.separation_factor = 0.1  # No groups in this case
def recipe_template(seed):
    def inner():
        # These imports are necessary to be able to execute in a forkserver context; it does not copy the full memory space, so
        # we'd have to rely on the target to know the imports
        from thesis_lib.testdata_generators import scopesim_grid, lowpass
        import numpy as np
        return scopesim_grid(seed=seed, N1d=25, perturbation=2., psf_transform=lowpass(), magnitude=lambda N: np.random.uniform(18, 24, N))
    return inner

result_table_multi = astrometry_benchmark.photometry_multi(recipe_template, 'mag18-24_grid', n_images=12, config=no_overlap_config, threads=12)
clear_output()

# %%
fig = plots_and_sanitycheck.plot_input_vs_photometry_positions(result_table_multi)
save_plot(outdir, 'multi_astrometry_xy')

# %%
result_table_multi_recenter = result_table_multi.copy()
result_table_multi_recenter['offset']-= np.mean(result_table_multi_recenter['offset'])
fig = plots_and_sanitycheck.plot_deviation_vs_magnitude(result_table_multi_recenter)
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
