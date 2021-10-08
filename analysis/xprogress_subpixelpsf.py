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

# %%
# use this for interactive plots
# %matplotlib notebook
# %pylab
# use this to export to pdf instead
# #%matplotlib inline
from thesis_lib import *
import thesis_lib
from matplotlib.colors import LogNorm
from IPython.display import clear_output
import thesis_lib.config as config
from thesis_lib.util import save_plot, match_observation_to_source, make_gauss_kernel
from thesis_lib.scopesim_helper import make_anisocado_model
from thesis_lib.testdata.generators import read_or_generate_image
from thesis_lib.testdata.recipes import model_add_grid
import thesis_lib.astrometry.wrapper as astrometry_wrapper
import thesis_lib.astrometry.plots as astrometry_plots
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
anisocado_full = make_anisocado_model()
anisocado_lp = make_anisocado_model(lowpass=5)

# %%
from anisocado import AnalyticalScaoPsf
anisocado_img = AnalyticalScaoPsf(pixelSize=0.004/2, N=400*2, seed=0).shift_off_axis(0,0)


# %%
def image_moment(image, x_order, y_order):
    y, x = np.indices(image.shape)
    return np.sum(x**x_order * y**y_order * image)

def centroid(image):
    m00 = image_moment(image, 0, 0)
    m10 = image_moment(image, 1, 0)
    m01 = image_moment(image, 0, 1)
    return m10/m00, m01/m00


# %%
even = AnalyticalScaoPsf(pixelSize=0.004, N=512, seed=0).shift_off_axis(0,0)
odd = AnalyticalScaoPsf(pixelSize=0.004, N=511, seed=0).shift_off_axis(0,0)
print(even.shape, odd.shape)
print(centroid(even), centroid(odd))


# %%
figure()
imshow(even)

# %%
from photutils import FittableImageModel

# %%
figure()
imshow(anisocado_full.data-anisocado_img)

# %%
centroid(anisocado_full.data)

# %%
y,x=np.mgrid[-400:400:801j,-400:400:801j]
print(centroid(anisocado_img))
model = FittableImageModel(anisocado_img,oversampling=2,degree=3)
print(centroid(model.data))
print(centroid(model(x,y)))

# %%
figure()
imshow(anisocado_img)
axhline((anisocado_img.shape[1]-1)/2)
axvline((anisocado_img.shape[0]-1)/2)

figure()
imshow(anisocado_full.data)
axhline((anisocado_full.data.shape[1]-1)/2)
axvline((anisocado_full.data.shape[0]-1)/2)

# %%
from astropy.modeling.functional_models import Gaussian2D
y, x = np.indices(anisocado_img.shape)
gauss_img = Gaussian2D(x_stddev=5, y_stddev=5)(x-x.mean(), y-y.mean())
centroid(gauss_img)

# %%
lowpass_config = config.Config()
lowpass_config.smoothing = make_gauss_kernel()
lowpass_config.output_folder = 'output_files_lowpass'
lowpass_config.use_catalogue_positions = True  # cheat with guesses
lowpass_config.photometry_iterations = 1  # with known positions we know all stars on first iter
lowpass_config.cutout_size = 20  # adapted by hand for PSF at hand
lowpass_config.fitshape = 15
lowpass_config.photometry_iterations=1
lowpass_config.create_dirs()

# %%
session_conv = astrometry_wrapper.Session(lowpass_config, 'gausscluster_N2000_mag22_lowpass')
session_conv.do_it_all()

# %%
session_subp = astrometry_wrapper.Session(lowpass_config, 'gausscluster_N2000_mag22_lowpass_subpixel')
session_subp.do_it_all()

# %%
astrometry_plots.plot_image_with_source_and_measured(session_conv.image, session_conv.tables.input_table, session_conv.tables.result_table)
pass

# %%
astrometry_plots.plot_image_with_source_and_measured(session_subp.image, session_subp.tables.input_table, session_subp.tables.result_table)
pass

# %%
fig,axs = plt.subplots(1,2)
axs[0].imshow(session_conv.epsf.data, norm=LogNorm())
axs[1].imshow(session_subp.epsf.data, norm=LogNorm())

# %%
tab_conv = session_conv.tables.valid_result_table
tab_subp = session_subp.tables.valid_result_table

# %%
np.mean(tab_subp['x_offset']), np.mean(tab_subp['y_offset'])

# %%
figure()
plt.scatter(tab_subp['x_offset'], tab_subp['y_offset'], alpha=0.8, label='subp')
plt.scatter(tab_conv['x_offset'], tab_conv['y_offset'], alpha=0.8, label='conv')
plt.legend()
np.std(tab_subp['x_offset']**2 +tab_subp['y_offset']**2), np.std(tab_conv['x_offset']**2 +tab_conv['y_offset']**2)

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
        from thesis_lib.testdata.generators import gaussian_cluster_modeladd
        from thesis_lib.scopesim_helper import make_anisocado_model
        import numpy as np
        psf_model=make_anisocado_model()
        return gaussian_cluster_modeladd(N=1000, seed=seed, magnitude=lambda N: np.random.normal(21, 2, N))
    return inner

result_table_multi = thesis_lib.astrometry.wrapper.photometry_multi(recipe_template, 'gausscluster_modeladd',
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
