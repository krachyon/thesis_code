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
import os
import os.path as p

# %%
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
outdir = './out_initial_astrometry'
if not os.path.exists(outdir):
    os.mkdir(outdir)
scopesim_helper.download()

# %% [markdown]
# # Explanation of software
# `thesis_lib` is a module to facilitate quick comparison of the astrometric performance of photutils on generated test images, depending on processing parameters and algorithm selection. Processing parameters are centralized in the config dataclass.
#
# [`photutils`](#photutils) Is a subproject of astropy. Among its functionality it implements a scheme described in [Anderson](#anderson_king) to derive an EPSF from an image. It is currently not able to take a field-varying PSF into account.
# The version used here has been amended in two important ways: It fixes an error that caused the photometric fit of grouped stars to fail and an error that caused the EPSF based photometry to not modify the initial guess due to an inverted parameter.
#
#
# [`scopesim`](#scopesim) and [`anisocado`](#anisocado) are packages to simulate telescope exposures and the expected PSF of the MICADO instrument respectively.
# assumed parameters for PSF: shift=(0, 14), wavelength=2.15 μm
#
# ## Limitations
#
# - Currently all images have a field constant PSF
# - Only the inner detector segment of MICADO is considered
# - As the atmospheric distortion/correction effects for MICADO are currently not working in scopesim, they are not taken into account.

# %%
# content of the module and default config
print([i for i in dir(thesis_lib) if not i.startswith('__')])
print('\n')
with open(os.path.join(outdir, 'default_config.txt'), 'w') as f:
    def_config = str(config.Config.instance())
    f.write(def_config)
print(def_config)

# %% [markdown]
# # Applying the standard photutils workflow
# ## Naive approach
#
# To test the roundtrip performance of the Anderson&King recipe in photutils, an image of a star cluster is generated with scopesim. It uses the default values of the cluster generator in scopesim_templates. The analysis closely follows https://photutils.readthedocs.io/en/stable/epsf.html#build-epsf. 
# Some values where changed to account for obvious differences between HST data and simulated MICADO images:
# - The `cutout_size` which determines the area around candidate stars has been set to 50 pixels
# - `stars_to_keep` controls how many candidate stars are used for PSF building. One can usually get a decent EPSF on HST data with less than 50 stars, adding more also does not seem to improve performance here
# - `fwhm_guess`, was eyeballed to around 7 pixels, a bit more than needed. This should allow the starfinder to select appropriate candidate stars in the first pass and reject some smaller scale noise. In later iterations, the FWHM of the EPSF is estimated by fitting a Gaussian. 
# - `threshold_factor` second starfinder parameter, set to $3σ$ by default
# - `clip_sigma=3`, for background estimation
# - `separation_factor` describes how many FWHMs two stars need to be apart from each other to be considered independent. Set to 2 by default
#

# %%
# astrometry with default parameters on default scopesim image
default_config = config.Config()
default_config.create_dirs()
image_name_scc = 'scopesim_cluster'
image_recipe_scc = testdata_generators.misc_images[image_name_scc]
image_scc, input_table_scc = testdata_generators.read_or_generate_image(image_recipe_scc, 
                                                                        image_name_scc, 
                                                                        default_config.image_folder)
figure()
imshow(image_scc, norm=LogNorm())

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
plt.tight_layout()
cb = plt.colorbar(shrink=0.7)
cb.set_label('pixel count')
fig.set_size_inches(7,7)
plt.savefig(p.join(outdir,'standard_photutils.pdf'), dpi=250)
pass

# %% [markdown]
# The result of applying the standard workflow to the standard scopesim cluster is pretty poor. The fit both failed to detect faint sources while at the same time producing many spurious detections especially around the brightest sources. There are multiple causes of this, most prominently the star detection routine. 
#
# Photutils implements two choices of starfinder which work by searching for local maxima in the image and then use heuristics to filter out non-star sources. 
# The one used here is DAOFind, which is a re-implementation of the FIND routine from daophot. Trying to tune it's parameters `(threshold_factor, fwhm_guess)` to a sensible value for an image like this either discards too many real sources or creates spurious detections from the PSF artifacts.
#
# This is described in [this paper](psf_bump) one can employ some heuristics to exclude "psf bumps" from images. These are not currently implemented in photutils and probably cannot be applied to MICADO images without adaptations.
#
# A second issue is a divergence in the EPSF fitting routine; With the default of quartic smoothing and 4x oversampling a grid pattern is imprinted on the EPSF that causes the EPSF to diverge.
# To demonstrate this issue, an image with "stars" on a regular grid is used. The grid is perturbed by a random offset of 2 pixels to avoid aliasing effects

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
plt.savefig(p.join(outdir,'photutils_grid.pdf'), dpi=250)
#clear_output()

# %% [markdown]
# Even though all source where detected correctly, even in this example with optimal non-overlapping and unsaturated candidate stars it seems the smoothing with a kernel derived from a quartic polynomial fails to smooth out the high-frequency variations reliably. In less optimal conditions with overlapping sources and saturation, these artifacts quickly amplify and the fit fails to converge.

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

plt.savefig(p.join(outdir,'epsf_smoothing_comparison.pdf'), dpi=250)

pass

# %% [markdown]
# A possible solution is to slightly alter the fit procedure by swapping out the smoothing kernel with a 5x5 Gaussian. The resulting EPSF looks to be mostly free of artifacts. This comes at the cost of also potentially loosing some details. The oversampling of the EPSF was introduced to deal with an undersampled PSF, an issue that the MICADO PSF does not have, so it's an open question to what degree oversampling is needed.
#
#
# In the case of 4x oversampling, the Anderson&King method does not not interpolate smoothly into the finer EPSF grid, but a star image pixel's value is added fully to the closest pixel in the EPSF grid. This has the effect of possibly creating a "screen door" effect: There are essentially 16 choices of how to "snap" a star image to the EPSF grid, essentially the operation is equivalent to shrinking each star image pixel by a factor of 4, filling the resulting gaps with zeros and summing these for each star. So a low number of candidate stars can lead to artifacts if the stars happen to be sampled in a way that fills up the positions unevenly.
# The resulting artifacts then seem to be amplified by the fitting process. (TODO: describe plausible scenario?)
#
# Another possible noise reduction strategy besides choosing a smoother kernel, would be to modify the way that the individual star cutouts are combined to the oversampled grid, by interpolating them over all candidate $\mathrm{oversampling}^2$ pixels. This is not currently implemented

# %%
from astropy.modeling.functional_models import AiryDisk2D
screendoor = np.zeros((128, 128))
screendoor[::4, ::4] = 1
screendoor *= AiryDisk2D(radius=0.3)(*mgrid[-1:1:128j, -1:1:128j])
fig_a = figure()
#plt.title('schematic representation of how a star image\n is interpolated into the EPSF grid, 4x4 oversampling', wrap=True)
plt.imshow(screendoor+0.02, norm=LogNorm())
plt.savefig(p.join(outdir,'epsf_flygrid_a.pdf'), dpi=250)


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
plt.savefig(p.join(outdir,'epsf_flygrid_b.pdf'), dpi=250)
pass

# %% [markdown]
# ## Sidestepping issues
# As there's no obvious solution to the problem of detecting PSF artifacts as sources, the astrometric performance of photutils is evaluated for a best-case scenario.
# First we assume that we that we have a-priori knowledge of all relevant sources in the frame, by passing them as a initial guess to the photometry routine we effectively have a perfect star "detection". The guess is perturbed by a random $x$ and $y$ offset between $-0.1$ and $0.1$ pixel to simulate an imperfect but close initial guess.
#
# One caveat to this is that photometry with iterative subtraction does not make sense then, as we "detect" all sources in the first iteration. Overlap between sources should still be accounted for by fitting a compound model to overlapping sources.
#
# Second for the following images the PSF used to generate them has been been dramatically cut by multiplying it with a $σ=5$ Gaussian. This lowpass filter solves the PSF bump problem by cutting of the spiky structures. 
# As there's no full knowledge yet of how the PSF of MICADO will look like in practice and PSF variations during an exposure will probably smear the PSF out this should not totally invalidate the results.  TODO cite

# %%
psf_effect_orig   = scopesim_helper.make_psf()
psf_effect_filter = scopesim_helper.make_psf(transform=testdata_generators.lowpass())
ε=1e-10 #prevent zeros in log 
fig_a = plt.figure()
plt.imshow(psf_effect_orig.data+ε, norm=LogNorm())
plt.savefig(p.join(outdir,'psf_orig.pdf'), dpi=250)
plt.colorbar()

fig_b = plt.figure()
plt.imshow(psf_effect_filter.data+ε, norm=LogNorm())
plt.savefig(p.join(outdir,'psf_lowpass.pdf'), dpi=250)
plt.colorbar()

pass

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

# %% [markdown]
# TODO The `cutout_size` and `fitshape` really matter for precision, higher or lower than this makes it worse. Figure out if that can be automatically derived/tuned or what that means for fitting non-filtered PSFs

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
fig = plots_and_sanitycheck.plot_image_with_source_and_measured(
    photometry_result_lpc.image, photometry_result_lpc.input_table, photometry_result_lpc.result_table)

plt.savefig(p.join(outdir,'lowpass_astrometry.pdf'), dpi=250)

# %% [markdown]
# The result looks a lot more reasonable than the naive approach, there are individual sources that do not get detected e.g. (x=18,y=193) and some confusion in crowded areas like around (x=666,y=481). With regards to the crowding, there's a limit of how big one can choose the group size to not explode the analysis time too much. In the worst case, the whole image would be considered a single group and run-time explode by an unreasonable amount. 
#
# Here an iterative process of first fitting only the brightest sources and subtracting them out before handling fainter sources could be an advantage 
#
# When plotting the positional deviation for this simulated cluster image, some severe outliers are obvious but a good fraction of sources where detected with an accuracy below $\pm 0.2 \,\mathrm{pixel}$. For outliers of more than about 2 pixels the absolute value is probably no longer a sensible measure of the quality of fit as for a totally diverged fit the correspondence between input and measured is no longer unique, as multiple sources are within the deviation radius.

# %%
matched_result_lpc = util.match_observation_to_source(photometry_result_lpc.input_table, photometry_result_lpc.result_table)
fig = plots_and_sanitycheck.plot_input_vs_photometry_positions(matched_result_lpc)
plt.savefig(p.join(outdir,'lowpass_astrometry_xy.pdf'), dpi=250)

# %%
fig = plots_and_sanitycheck.plot_deviation_histograms(matched_result_lpc)

# %%
fig = plots_and_sanitycheck.plot_deviation_vs_magnitude(matched_result_lpc)
plt.xlim(-16,-12)
plt.savefig(p.join(outdir,'lowpass_astrometry_magvdev.pdf'), dpi=250)

# %% [markdown]
# Looking at the relation between magnitude and deviation, the outliers seem to dominate the statistics. To eliminate some of the outliers and to create a larger sample, multiple frames of a regular star grid are analyzed. The simulated star positions are again perturbed by a small amount to avoid aliasing, their brightness is uniformly chosen from a magnitude range of 18 to 24. Each frame contains $25^2$ stars, so with 12 frames, 7500 stars are sampled.

# %%
grid_img, grid_table = testdata_generators.read_or_generate_image(
    lambda: testdata_generators.scopesim_grid(seed=1444, N1d=25, perturbation=2., psf_transform=testdata_generators.lowpass(),
                                      magnitude=lambda N: np.random.uniform(18, 24, N)),
    'mag18-24_grid')
figure()
imshow(grid_img,norm=LogNorm())

# %% [markdown]
# Using the estimate for the achievable position accuracy from [Lindgren](lindgren):
# $$σ_{pos} = k  \frac{\mathrm{FWHM}(\mathrm{PSF})}{\mathrm{SNR}}$$
#
# Here $k=1$ is assumed, the FWHM of the PSF estimated by a gaussian fit.
# The SNR for each star is calculated by the photutils function `calc_total_error()`, which calculates the total error for each pixel as $$σ = \sqrt{\mathrm{bkg\_err}^2 + \mathrm{image\_value}} $$
# To estimate the signal from the whole star's area, the and error are obtained by essentially performing aperture photometry on the image with $r=\mathrm{FWHM}$. 
#
# The estimated SNR for a star is therefore
#
# $$\frac{\sum_{\mathrm{A}}\mathrm{data}}{\sqrt{\sum_{A}{σ^2}}}$$
#
# For this image we see a fairly broad range of achievable precision from $~0.01$ pixel for the brightest sources to $~0.05$ pixel for the faintest.

# %%
fwhm = photometry.estimate_fwhm(EPSFModel(psf_effect_filter.data))

mean,median,std=sigma_clipped_stats(grid_img,sigma=3)


sigmas_grid = util.estimate_photometric_precision_full(grid_img, grid_table,fwhm)
sigmas_grid_peak = util.estimate_photometric_precision_peak_only(grid_img, grid_table,fwhm)

fig = figure()
ax1 = fig.add_subplot(1,2,1)
ax1.hist(sigmas_grid,bins=50)
plt.xlabel('$σ_{pos}$')
plt.ylabel('density')

ax2 = fig.add_subplot(1,2,2)

ax2.plot(grid_table['m'], sigmas_grid, 'o',markersize=2)
plt.xlabel('magnitude')
plt.ylabel('$σ_{pos}$')
pass

# %% [markdown]
# std^2 != median. Sollte diese Relation für den Hintergrund überhaupt gelten? Imho ist der ja nicht poisson-verteilt

# %%
#figure()
#hist(grid_img[grid_img<median+3*std],bins=np.linspace(2000,5000,256))
fig,ax = plt.subplots();ax.hist(grid_img.flatten(),bins=np.linspace(2000,5000,256))
ax.axvline(median,color='r')
ax.axvline(median-std,color="r")
ax.axvline(median+std,color="r")
pass

# %%
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.utils import calc_total_error
aper = CircularAperture(np.array((grid_table["x"],grid_table["y"])).T, r=fwhm)   #r can be really small, then it only calculates the s/n of the peak, or as large as the whole PSF Fit radius i.e. 7.5 

# max
flux_total,_ = aper.do_photometry(grid_img)
# TODO these two lines are effectively sum(errors), should be 
# # sqrt(sum(errors**2))?
flux_sky,_ = aper.do_photometry(np.ones_like(grid_img))
flux_sky*=median
print(f"mean: {mean}, median: {median}, σ: {std}, variance: {std**2}")
std_sum = aper.do_photometry(np.ones_like(grid_img))[0]*median


signal = flux_total - flux_sky
noise = np.sqrt(signal + std_sum)
sn = signal/noise

# my approach
# TODO how is this line motivated?
variance = np.maximum(grid_img-median, 0)
#variance = grid_img-median
#assert np.sum(calc_total_error(grid_img-median, std, 1) - np.sqrt(variance+std**2)) == 0
# error_img = calc_total_error(grid_img, std, 1)
error_img = np.sqrt(variance+std**2)

# signals, errors = aper.do_photometry(grid_img-median, error_img)
signals, _ = aper.do_photometry(grid_img-median)
errors = np.sqrt(aper.do_photometry(error_img**2)[0])

figure()
plot(grid_table["m"], fwhm/sn,".",label="Max Estimate")
plot(grid_table['m'], fwhm/(signals/errors), 'o',markersize=2, label='σ full')
ylim(0,.25)
legend()


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

# %% [markdown]
# With increased sampling, an anisotropy in the $x-y$ deviation and a constant offset becomes apparent. At the time of writing this, it is assumed to be due to an artifact of the way the lowpass-filtered PSF is interpreted by scopesim.
# Overall there are no major outliers with multiple pixels anymore, so it is assumed that the fit did not diverge for any star. 

# %%
fig = plots_and_sanitycheck.plot_input_vs_photometry_positions(result_table_multi)
plt.savefig(p.join(outdir,'multi_astrometry_xy.pdf'), dpi=250)

# %%
fig = plots_and_sanitycheck.plot_deviation_histograms(result_table_multi)

# %%
result_table_multi_recenter = result_table_multi.copy()
result_table_multi_recenter['offset']-= np.mean(result_table_multi_recenter['offset'])
fig = plots_and_sanitycheck.plot_deviation_vs_magnitude(result_table_multi_recenter)
plt.ylim(-0.07,0.07)
plt.title(plt.gca().get_title()+' (subtracted systematic error)')
plt.savefig(p.join(outdir,'multi_astrometry_mag.pdf'), dpi=250)
pass

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
plt.savefig(p.join(outdir,'multi_astrometry_noisefloor.pdf'), dpi=250)

# %%
figure()

plt.plot(result_table_multi['m'], 1/(result_table_multi['σ_pos_estimated']/fwhm), 'bo',markersize=0.5, label='SNR')

plt.legend()

# %% [markdown]
# The functional shapes of the estimated and achieved precision differ quite a bit, separating the error due to background and poisson noise maybe yields some insight

# %%
from photutils import CircularAperture

img_partial, input_table_partial = testdata_generators.read_or_generate_image(lambda:1/0,'mag18-24_grid_0')
result_partial = photometry.run_photometry(img_partial, input_table_partial, config=no_overlap_config)

matched_partial = util.match_observation_to_source(input_table_partial,result_partial.result_table)
mean, median, std = sigma_clipped_stats(img_partial, sigma=3.)

xy = np.array((input_table_partial['x'], input_table_partial['y'])).T
apertures = CircularAperture(xy, r=fwhm)

signals, bkg_error = apertures.do_photometry(img_partial, np.ones(img_partial.shape)*std)
signals, poisson_error = apertures.do_photometry(img_partial, np.sqrt(img_partial))


# %%
plt.figure()
plt.plot(input_table_partial['m'], bkg_error, 'o', markersize=1,
         label='σ due to background')
plt.plot(input_table_partial['m'], poisson_error, 'o', markersize=1,
         label='σ due to poisson noise')
plt.plot(input_table_partial['m'], np.sqrt(poisson_error**2+bkg_error**2), 'o', markersize=1,
         label='σ combined')
plt.xlabel('magnitude')
plt.ylabel('noise for star in photons')
plt.legend()
print(f'FWHM: {photometry.estimate_fwhm(result_partial.epsf)}')

# %% [markdown]
# TODO: The background noise here seems excessively high...
# ## comparison to DAOPHOT
# The following is only run on a single image, not a stack as a sanity-check. There seems no deviation that would indicate a clearly diverging result

# %%
# For this to work you need a binary of daophotII and point the astwro config file to it
# see https://astwro.readthedocs.io/en/latest/installation.html#configuration
fwhm = photometry.estimate_fwhm(EPSFModel(psf_effect_filter.data))
options = (('FITTING RADIUS', f'{fwhm:.2f}'),)
daophot_result = photometry_daophot.run_daophot_photometry(grid_img, grid_table, daophot_options=options)
matched_daophot = util.match_observation_to_source(daophot_result.input_table, daophot_result.result_table)

# %%
fig = plots_and_sanitycheck.plot_input_vs_photometry_positions(matched_daophot)

# %%
fig = plots_and_sanitycheck.plot_deviation_histograms(matched_daophot)

# %%
fig = plots_and_sanitycheck.plot_deviation_vs_magnitude(matched_daophot)

# %% [markdown]
# ## Effect of groups
# To evaluate how the precision of the EPSF estimation and the subsequent astrometry is affected by overlapping sources, groups of stars with a radius of about 8 pixel are simulated on a grid.
#
# Executing the photomteric fit in with groups takes significantly longer as stars in a group are fitted as compound models.
# TODO: ALLSTAR from daophot seems to handle groups a lot better.
#
# ### This whole thing does not make sense to me yet...
# There seems to be a critical point of star density where the fit does not really work anymore:
# Group radius 10, group size 5, uniform magnitude 20: σ is better than for the grid
#
# for higher densities, outliers start to crop up, where individual stars from the group have a huge offset. The bulk of the measurements appears to be mostly unaffected.
#
# the FWHM separation factor to find groups does not actually seem to do that much, or something weird causes the outliers if it does indeed have an effect
#
# The following settings seem to hit the onset of chaos pretty well. Bright stars are still fine, but faint ones contain huge outliers and are even shifted into adjacent groups somehow (???)

# %%
group_config = copy(lowpass_config)
group_config.separation_factor=3  # even with 0.01 it still seems to find groups...

# %%
# use an image with less sources to not have to filter the input positions
image_name_grp = 'scopesim_groups_n7_mag18-24'
image_recipe_grp = lambda: testdata_generators.scopesim_groups(
    N1d=16, jitter=3., psf_transform=testdata_generators.lowpass(), 
    magnitude=lambda N: np.random.uniform(18, 24, N), group_radius=8, group_size=7)

image_grp, input_table_grp = testdata_generators.read_or_generate_image(image_recipe_grp, 
                                                                        image_name_grp, 
                                                                        group_config.image_folder)

photometry_result_grp = photometry.run_photometry(image_grp,input_table_grp,image_name_grp,group_config)
matched_result_grp = util.match_observation_to_source(photometry_result_grp.input_table, photometry_result_grp.result_table)

# %%
fig = plots_and_sanitycheck.plot_image_with_source_and_measured(
    photometry_result_grp.image, photometry_result_grp.input_table, photometry_result_grp.result_table)

# %%
fig = plots_and_sanitycheck.plot_deviation_histograms(matched_result_grp)

# %%
fig = plots_and_sanitycheck.plot_deviation_vs_magnitude(matched_result_grp)

# %% [markdown]
# # Further discussion
# The reachable precision of the EPSF derivation and EPSF photometry of photutils is on the order of

# %%
(0.025*scopesim_helper.pixel_scale*u.pixel).to(u.microarcsecond)

# %% [markdown]
# if the results here where to apply to the zoomed mode of MICADO with reduced plate scale without change, we could expect an accuracy of 

# %%
(0.025*1.5*u.milliarcsecond).to(u.microarcsecond)

# %% [markdown]
# Which would meet the $50μ \mathrm{as}$ target, but considering these results where derived form idealized situations and significantly improving on the requirement would open quite a few science cases, 

# %% [markdown]
# # Bibliography
# 1. <a id='anisocado'></a> https://github.com/astronomyk/ScopeSim
# 1. <a id='scopesim'></a> https://github.com/astronomyk/ScopeSim
# 1. <a id='photutils'></a> https://github.com/astropy/photutils
# 1. <a id='anderson_king'></a>Anderson, Jay, and Ivan R. King. “Toward High-Precision Astrometry with WFPC2. I. Deriving an Accurate Point-Spread Function.” Publications of the Astronomical Society of the Pacific 112 (October 1, 2000): 1360–82. https://doi.org/10.1086/316632.
# 2.<a id='psf_bump'></a> Anderson, Jay, Ivan R. King, Harvey B. Richer, Gregory G. Fahlman, Brad M. S. Hansen, Jarrod Hurley, Jasonjot S. Kalirai, R. Michael Rich, and Peter B. Stetson. “Deep Advanced Camera for Surveys Imaging in the Globular Cluster NGC 6397: Reduction Methods.” The Astronomical Journal 135 (June 1, 2008): 2114–28. https://doi.org/10.1088/0004-6256/135/6/2114.
# 1.<a id='lindgren'></a> Lindegren, Lennart. “Photoelectric Astrometry - A Comparison of Methods for Precise Image Location.” International Astronomical Union Colloquium 1 (September 1, 1978): 197–217. https://doi.org/10.1017/S0252921100074157.
#
#
