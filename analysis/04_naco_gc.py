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
from matplotlib.colors import LogNorm,SymLogNorm, Normalize
from photutils import IRAFStarFinder, EPSFBuilder, extract_stars, \
 BasicPSFPhotometry, DAOGroup, MMMBackground, SExtractorBackground, FittableImageModel, StarFinder
from astropy.nddata import NDData
from astropy.stats import sigma_clipped_stats
from thesis_lib.util import cached, save_plot
from thesis_lib import util
# only on my branch
from photutils.psf.culler import CorrelationCuller
from photutils.psf.incremental_fit_photometry import IncrementalFitPhotometry, FitStage, all_individual, brightest_simultaneous

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
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
#util.RERUN_ALL_CACHED = True
cache_dir = Path('./cached_results/')
out_dir = Path('./04_comparison')
base_dir = Path('./gc_images').absolute()
image_cut = np.s_[170:, 100:]
fitshape = 19

# %%
img_combined = getdata(base_dir/'ssa_deep.fits').astype(np.float64)[image_cut]

# %%
epsf_finder = IRAFStarFinder(threshold=20, fwhm=4.5, minsep_fwhm=10, peakmax=10_000, exclude_border=True, brightest=300)
epsf_sources = epsf_finder(img_combined)

# %%
plt.figure()
plt.imshow(img_combined, norm=LogNorm())
plt.plot(epsf_sources['xcentroid'], epsf_sources['ycentroid'], 'rx')

# %%
epsf_sources['x'] = epsf_sources['xcentroid']
epsf_sources['y'] = epsf_sources['ycentroid']

epsf_stars = extract_stars(NDData(img_combined), epsf_sources, size=(fitshape+2, fitshape+2))
builder = EPSFBuilder(oversampling=2, smoothing_kernel='quadratic', maxiters=7)
initial_epsf, _ = cached(lambda: builder.build_epsf(epsf_stars), cache_dir/'initial_epsf_naco')

# %%
plt.figure()
plt.imshow(initial_epsf.data, norm=LogNorm())

# %%
y,x = np.mgrid[-fitshape/2:fitshape/2:1j*fitshape, -fitshape/2:fitshape/2:1j*fitshape]
psffind = StarFinder(threshold=10, kernel=initial_epsf(x,y), min_separation=20, exclude_border=True, peakmax=10_000, brightest=400)
epsf_sources_improved = psffind(img_combined)

# %%
plt.figure()
plt.imshow(img_combined, norm=LogNorm())
plt.plot(epsf_sources_improved['xcentroid'], epsf_sources_improved['ycentroid'], 'rx')
save_plot(out_dir, 'naco_epsfstars')

# %%
epsf_sources_improved['x'] = epsf_sources_improved['xcentroid']
epsf_sources_improved['y'] = epsf_sources_improved['ycentroid']

epsf_stars = extract_stars(NDData(img_combined-SExtractorBackground()(img_combined)), epsf_sources_improved, size=(fitshape+2, fitshape+2))
from thesis_lib.util import make_gauss_kernel
builder = EPSFBuilder(oversampling=4, smoothing_kernel=make_gauss_kernel(1.4), maxiters=7)
improved_epsf, _ = cached(lambda: builder.build_epsf(epsf_stars), cache_dir/'improved_epsf_naco')
builder = EPSFBuilder(oversampling=4, smoothing_kernel='quadratic', maxiters=7)
quadratic_epsf, _ = cached(lambda: builder.build_epsf(epsf_stars), cache_dir/'improved_epsf_naco_quadratic')

# %%
xfreq = np.fft.fftshift(np.fft.fftfreq(improved_epsf.data.shape[1]))
yfreq = np.fft.fftshift(np.fft.fftfreq(improved_epsf.data.shape[0]))
extent = (xfreq[0], xfreq[-1], yfreq[0], yfreq[-1])
print(extent)
fig,axs = plt.subplots(2,2)

axs[0,0].imshow(improved_epsf.data, norm=LogNorm())
axs[0,1].imshow(np.fft.fftshift(np.abs(np.fft.fft2(improved_epsf.data))), norm=LogNorm(), extent=extent)
axs[1,0].imshow(quartic_epsf.data, norm=LogNorm())
axs[1,1].imshow(np.fft.fftshift(np.abs(np.fft.fft2(quartic_epsf.data))), norm=LogNorm(), extent=extent)
axs[0,0].set_ylabel('gaussian smoothing')
axs[1,0].set_ylabel('quadratic smoothing')
axs[1,0].set_xlabel('PSF data')
axs[1,1].set_xlabel('Fourier transformed (absolute value)')
plt.tight_layout()
save_plot(out_dir, 'naco_findpsfs')

# %%
from astropy.table import Table
manual_epsf_table = Table(np.array([
[440.9201050546983 , 562.0829050562483           ], 
[526.1653615834007 , 671.176530445491            ],
[650.8574669847673 , 564.0537050610642           ],  
[629.0981219113359 , 387.1530804872457           ],  
[401.1200046657005 , 380.96142614080304          ],  
[332.03392694641326,  395.0369384274606          ],  
[435.9594031452876 , 404.8023428317487           ],  
[410.89328434733125,  243.84401800616766         ],  
[556.1720130882134 , 101.92213914631849          ],  
[400.97038204170934,  58.831588259703054         ],  
[225.94354374993011,  128.16128264019468         ],  
[48.97044913464503 , 147.180936233043            ],  
[50.86614879766106 , 410.0444327250993           ],  
[157.0439679234367 , 517.937521752227            ]]), names=['x','y'] )

#epsf_stars = extract_stars(NDData(img_combined), manual_epsf_table, size=(fitshape+2, fitshape+2))
#builder = EPSFBuilder(oversampling=1, smoothing_kernel='quadratic', maxiters=7)
#improved_epsf, _ = builder.build_epsf(epsf_stars)

# %%
plt.figure()
plt.imshow(img_combined, norm=LogNorm())
plt.plot(manual_epsf_table['x'], manual_epsf_table['y'], 'rx')

# %% [markdown]
# # with custom changes

# %%
photometry_finder = StarFinder(threshold=5, 
                               kernel=improved_epsf(x,y),
                               min_separation=1, 
                               exclude_border=True)
def detect_and_cull():
    photometry_sources = photometry_finder(img_combined)
    culler = CorrelationCuller(cutoff_corr=1, image=img_combined, model=improved_epsf)
    culler.cull_data(photometry_sources)
    return photometry_sources

photometry_sources = cached(detect_and_cull, cache_dir/'photometry_sources_naco')
photometry_sources_culled = photometry_sources[np.isfinite(photometry_sources['model_chisquare'])]

# %%
plt.figure()
plt.imshow(img_combined, norm=LogNorm())
plt.plot(photometry_sources_culled['xcentroid'], photometry_sources_culled['ycentroid'], 'kx', label=f'selected sources N={len(photometry_sources_culled)}')
deselected = photometry_sources[~np.isfinite(photometry_sources['model_chisquare'])]
plt.plot(deselected['xcentroid'], deselected['ycentroid'], 'rx', label=f'culled sources N={len(deselected)}')
plt.legend()
plt.xlim(760,990)
plt.ylim(820,600)
save_plot(out_dir, 'naco_culler')

# %%
# This is a ugly hack to avoid extrapolation of the interpolator breaking and messing up the residual image: 
# Also some genius just re-implemented the interpolator from FittableImageModel 
# in its subclass EPSFModel in a slightly different way that breaks the way I use it (oversampling applied differently)
data = np.zeros(np.array(improved_epsf.data.shape)+8)
data[4:-4, 4:-4] = improved_epsf.data
epsf_adapted = FittableImageModel(data=data, oversampling=improved_epsf.oversampling)

data_quadratic= np.zeros(np.array(quadratic_epsf.data.shape)+8)
data_quadratic[4:-4, 4:-4] = quadratic_epsf.data
quadratic_epsf_adapted = FittableImageModel(data=data_quadratic, oversampling=quadratic_epsf.oversampling)

#figure()
#imshow(epsf_adapted.data)

# %%
guess_table = photometry_sources.copy()
guess_table['x_0'] = guess_table['xcentroid']
guess_table['y_0'] = guess_table['ycentroid']
guess_table['flux_0'] = guess_table['flux']

fit_stages = [FitStage(10, 1e-10, 1e-10, np.inf, all_individual), # first stage: get flux approximately right
              FitStage(10, 0.5, 0.5, 10, all_individual), # optimize position, keep flux constant
              FitStage(10, 1., 1., 4000, brightest_simultaneous(3)), # optimize brightest sources in a group
              FitStage(10, 0.2, 0.2, 500, all_individual), # fit everything again with tight bounds
              FitStage(30, 0.1, 0.1, 200, all_individual), # fit everything again with tight bounds and larger image area
              ]


photometry = IncrementalFitPhotometry(SExtractorBackground(),
                                           epsf_adapted,
                                           max_group_size=15,
                                           group_extension_radius=15,
                                           fit_stages=fit_stages,
                                           use_noise=True)

photometry_quadratic = IncrementalFitPhotometry(SExtractorBackground(),
                                           quadratic_epsf_adapted,
                                           max_group_size=15,
                                           group_extension_radius=15,
                                           fit_stages=fit_stages,
                                           use_noise=True)


result_table_combined_quadratic = cached(lambda: photometry_quadratic.do_photometry(img_combined, guess_table), cache_dir/'photometry_combined_naco_quadratic')
result_table_combined = cached(lambda: photometry.do_photometry(img_combined, guess_table), cache_dir/'photometry_combined_naco_gauss')

# %%
photometry._last_image = img_combined - photometry.bkg_estimator(img_combined)
residual = photometry.residual(result_table_combined)
plt.figure()
#plt.imshow(residual, norm=SymLogNorm(10,vmin=-7000,vmax=5000, clip=True))
cmap = plt.cm.get_cmap('viridis').copy()
cmap.set_over('red')
cmap.set_under('red')
plt.imshow(residual, norm=Normalize(vmin=-700, vmax=400, clip=False), cmap=cmap)
#plt.imshow(np.abs(img_combined/residual), norm=Normalize(vmin=0, vmax=200, clip=False))
plt.colorbar()
save_plot(out_dir, "naco_residual")

# %%
bootstrapped_imgs = [getdata(path).astype(np.float64)[image_cut] for path in sorted(list((base_dir).glob('ssa_deep_*.fits')))]

def run_photometry():
    result_tables = []
    for img in bootstrapped_imgs:
        result_tables.append(photometry.do_photometry(img, guess_table))
    return result_tables

def run_photometry_quadratic():
    result_tables = []
    for img in bootstrapped_imgs:
        result_tables.append(photomtery_quadratic.do_photometry(img, guess_table))
    return result_tables

result_tables = cached(run_photometry, cache_dir/'naco_photometry_bootstrapped_gauss')
result_tables_quadratic = cached(run_photometry, cache_dir/'naco_photometry_bootstrapped_quadratic')

# %%
plt.figure()
#plt.hexbin(xs%1, ys%1,gridsize=200)
plt.plot(result_table_combined_quadratic['x_fit']%1, result_table_combined_quadratic['y_fit']%1,
         '+', alpha=0.7, label='EPSF smoothed with quadratic interpolation')
plt.plot(result_table_combined['x_fit']%1, result_table_combined['y_fit']%1,
         'x', alpha=0.7, label='EPSF smoothed with σ=1.4 gaussian')
plt.xlabel('x-pixel-phase')
plt.ylabel('y-pixel-phase')
plt.gcf().set_size_inches(6,6)
plt.legend()
save_plot(out_dir, 'naco_pixelphase_distribution')

# %%
result_table_combined.sort('id')
for tab in result_tables:
    tab.sort('id')
xs = np.array([tab['x_fit'] for tab in result_tables]).flatten()
ys = np.array([tab['y_fit'] for tab in result_tables]).flatten()
xdev = np.array([tab['x_fit']-result_table_combined['x_fit'] for tab in result_tables]).flatten()
ydev = np.array([tab['y_fit']-result_table_combined['y_fit'] for tab in result_tables]).flatten()
flux = np.array([tab['flux_fit'] for tab in result_tables]).flatten()

xstd = np.std(xdev)
_,_,xstd_clipped = sigma_clipped_stats(xstd, sigma=100)
ystd = np.std(xdev)
_,_,ystd_clipped = sigma_clipped_stats(ystd, sigma=100)

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
#ax.set_rscale('symlog')
norm = LogNorm(flux.min(),flux.max())
cmap = mpl.cm.get_cmap('plasma').copy()
plt.scatter(np.arctan2(xdev,ydev), (xdev**2+ydev**2)**(1/32) , alpha=0.5, s=20, c=flux, norm=norm, cmap=cmap, linewidths=0)
rs = np.array([0,0.2,0.4,0.6,0.8,1.])
rlabels = [f'{r:.2e}' for r in rs**16]
ax.set_rticks(rs, rlabels, backgroundcolor=(1,1,1,0.8))
ax.set_rlabel_position(67)
cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='log(flux)')
save_plot(out_dir, 'naco_xy_density')
xstd, ystd, xstd_clipped, ystd_clipped

# %%
plt.figure()
eucdev = np.sqrt(xdev**2+ydev**2)
plt.hist(eucdev,bins=100, log=True)
plt.axvline(0.01, color='orange')
np.sum(np.sqrt(xdev**2+ydev**2) < 0.01), len(xdev)
plt.xlim(-0.05, 0.3)
plt.text(-0.045, 4, f'$N_{{<0.01}} = {np.sum(eucdev<0.01)}$', bbox={'facecolor': 'white', 'alpha': 0.8})
plt.text(0.03, 4, f'$N_{{>0.01}} = {np.sum(eucdev>0.01)}$', bbox={'facecolor': 'white', 'alpha': 0.8})
plt.ylabel('number of samples')
plt.xlabel('euclidean centroid deviation from summed image')
save_plot(out_dir, 'naco_deviation_histogram')

# %%
plt.figure()
plt.plot(np.log(1/flux), eucdev,'.', alpha=0.4)

# %% [markdown]
# # Just debugging below

# %%
from photutils.centroids import centroid_quadratic
from thesis_lib.util import center_of_index
initial_epsf_adapted = FittableImageModel(initial_epsf.data, oversampling=initial_epsf.oversampling)
print(centroid_quadratic(initial_epsf_adapted.data))
interp = initial_epsf_adapted.interpolator(np.arange(initial_epsf_adapted.nx),np.arange(initial_epsf_adapted.ny))
print(centroid_quadratic(interp))
print(center_of_index((initial_epsf_adapted.data.shape)))
np.max(initial_epsf_adapted.data-interp)

# %%
y, x = [i.flatten() for i in np.mgrid[-10:10:100j, -10:10:100j]]
y, x = np.random.uniform(-2,2,(2,250,250)); y=y.flatten(); x=x.flatten()
plt.figure()
plt.plot(*log_squish(x,y, exp=3), 'x')

# %%
plt.figure()
plt.imshow(img_combined)
jet = plt.cm.get_cmap('jet')
for i, tab in enumerate(result_tables):
    plt.scatter(tab['x_fit'], tab['y_fit'], label=f'{i}', alpha=0.4, color=jet(i/len(result_tables)))
plt.legend()

# %%
plt.figure()
plt.imshow(img_combined, norm=LogNorm())
plt.plot(result_table_combined['x_0'], result_table_combined['y_0'], '+', color='orange', label='initial')
plt.plot(result_table_combined['x_fit'], result_table_combined['y_fit'], 'rx', label='fit')
plt.legend()


# %%
#result_table_combined['x'] = result_table_combined['x_fit']
#result_table_combined['y'] = result_table_combined['y_fit']
#result_table_combined.sort('flux_fit', reverse=True)
#epsf_result_table = result_table_combined[result_table_combined['max_value'] <10_000][:100]

#epsf_stars = extract_stars(NDData(img_combined), epsf_result_table, size=(51, 51))
#builder = EPSFBuilder(oversampling=4, smoothing_kernel='quadratic', maxiters=7)
#subtraction_psf, _ = cached(lambda: builder.build_epsf(epsf_stars), cache_dir/'subtraction_psf_naco', rerun=True)

#data = np.zeros(np.array(subtraction_psf.data.shape)+2)
#data[4:-4, 4:-4] = subtraction_psf.data
#subtraction_psf_adapted = FittableImageModel(data=data, oversampling=subtraction_psf.oversampling)
#plt.figure()
#plt.imshow(subtraction_psf_adapted.data, norm=LogNorm())

# %%
# caution: takes a while
from photutils.psf.incremental_fit_photometry import make_groups
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib.patches import Polygon
def visualize_grouper(image, input_table, max_size=20, halo_radius=25):
    table = input_table.copy()
    
    table.rename_columns(['x_0','y_0'], ['x_fit', 'y_fit'])
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

        #plt.annotate(group_id, (np.mean(core_group['x_fit']),np.mean(core_group['y_fit'])), 
        #             color='white', backgroundcolor=(0,0,0,0.25), alpha=0.7)


    plt.imshow(image, norm=LogNorm())
    
guess_table = photometry_sources_culled.copy()
guess_table['x_0'] = guess_table['xcentroid']
guess_table['y_0'] = guess_table['ycentroid']
guess_table['flux_0'] = guess_table['flux']
#visualize_grouper(img_combined, guess_table, max_size=15, halo_radius=15)

# %%
def log_squish(xs, ys, exp=1):
    rs = np.sqrt(xs**2 + ys**2)
    ϕs = np.arctan2(xs,ys)
    
    lrs = np.log(exp*rs+1)
    
    return lrs * np.cos(ϕs), lrs * np.sin(ϕs)

def sqrt_squish(xs, ys, base=2):
    rs = np.sqrt(xs**2 + ys**2)
    ϕs = np.arctan2(xs,ys)
    
    lrs = rs**(1/base)
    
    return lrs * np.cos(ϕs), lrs * np.sin(ϕs)


# %%
plt.figure()
#plt.hexbin(*sqrt_squish(xdev, ydev, base=16), norm=LogNorm())
plt.hexbin(xdev, ydev, norm=LogNorm(), gridsize=300)
plt.xlabel('deviation in x direction')
plt.ylabel('deviation in y direction')
plt.colorbar(label='number of samples')
save_plot(out_dir, 'naco_xy_density')
