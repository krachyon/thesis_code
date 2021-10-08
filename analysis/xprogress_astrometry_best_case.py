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

# make it work in linked script
import matplotlib.pyplot as plt

from thesis_lib import *
import thesis_lib
from thesis_lib.astrometry.wrapper import Session
from matplotlib.colors import LogNorm
from copy import copy
from thesis_lib.scopesim_helper import make_anisocado_model
import astropy.table

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
# if something goes wrong here, check the stdout of the jupyter notebook server for hints
lowpass_config = config.Config()
lowpass_config.smoothing = util.make_gauss_kernel()  # avoid EPSF artifacts
lowpass_config.output_folder = 'output_files_lowpass'
lowpass_config.image_folder = '/data/beegfs/astro-storage/groups/matisse/messlinger/testfiles'
lowpass_config.use_catalogue_positions = True  # cheat with guesses
lowpass_config.photometry_iterations = 1  # with known positions we know all stars on first iter
lowpass_config.cutout_size = 20  # adapted by hand for PSF at hand
lowpass_config.fitshape = 15
lowpass_config.separation_factor = 2
lowpass_config.detector_saturation=15000
lowpass_config.create_dirs()

# %%
# if something goes wrong here, check the stdout of the jupyter notebook server for hints
no_overlap_config = copy(lowpass_config)
no_overlap_config.separation_factor = 0.1  # No groups in this case



def recipe_template_conv(seed):
    def inner():
        # These imports are necessary to be able to execute in a forkserver context; it does not copy the full memory space, so
        # we'd have to rely on the target to know the imports
        from thesis_lib.testdata.recipes import scopesim_grid
        from thesis_lib.testdata.helpers import lowpass
        import numpy as np
        return scopesim_grid(seed=seed, N1d=9, perturbation=2., psf_transform=lowpass(), magnitude=lambda N: np.random.uniform(18, 24, N))
    return inner

psf = make_anisocado_model(lowpass=5)
def recipe_template_subpixel(seed):
    def inner():
        # These imports are necessary to be able to execute in a forkserver context; it does not copy the full memory space, so
        # we'd have to rely on the target to know the imports
        from thesis_lib.testdata.recipes import scopesim_grid
        import numpy as np
        return scopesim_grid(seed=seed, N1d=9, perturbation=2., custom_subpixel_psf=psf, magnitude=lambda N: np.random.uniform(18, 24, N))
    return inner


sessions_multi_conv = thesis_lib.astrometry.wrapper.photometry_multi(recipe_template_conv, 'mag18-24_grid_conv', n_images=50, config=no_overlap_config, threads=None)
result_table_multi_conv = astropy.table.vstack([session.tables.valid_result_table for session in sessions_multi_conv])

sessions_multi_subpixel = thesis_lib.astrometry.wrapper.photometry_multi(recipe_template_subpixel, 'mag18-24_grid_subpixel', n_images=50, config=no_overlap_config, threads=None)
result_table_multi_subpixel = astropy.table.vstack([session.tables.valid_result_table for session in sessions_multi_subpixel])

# %%
figa = plots.plot_xy_deviation(result_table_multi_conv)
figb = plots.plot_xy_deviation(result_table_multi_subpixel)

# %%
lowpass_config.oversampling = 2
lowpass_config.cutout_size = 40 
lowpass_config.fitshape = 25

#session_std = Session(lowpass_config, 'scopesim_grid_16_perturb2_mag18_24')
#session_known_psf = Session(lowpass_config, 'scopesim_grid_16_perturb2_mag18_24')

session_std = Session(lowpass_config, 'gausscluster_N2000_mag22_subpixel')
session_known_psf = Session(lowpass_config, 'gausscluster_N2000_mag22_subpixel')

session_std.do_it_all()

session_known_psf.epsf = make_anisocado_model()
session_known_psf.do_astrometry()

# %%
figa = plots.plot_xy_deviation(session_std.tables.valid_result_table)
figb = plots.plot_xy_deviation(session_known_psf.tables.valid_result_table)

# %%
fig, axs = plt.subplots(1,2)
axs[0].imshow(session_known_psf.epsf.data, norm=LogNorm())
axs[1].imshow(session_std.epsf.data, norm=LogNorm())

# %%
