# -*- coding: utf-8 -*-
# %%
# %matplotlib notebook
# %pylab

from itertools import chain
import os
import multiprocess as mp
from pathlib import Path
import pickle
import zstandard
from tqdm.auto import tqdm

from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from scipy.signal import fftconvolve
from scipy.stats import zscore

import astropy.units as u
from astropy.modeling.functional_models import AiryDisk2D
from astropy.convolution import AiryDisk2DKernel, convolve

from thesis_lib.standalone_analysis.sampling_precision import *
from thesis_lib.testdata.recipes import convolved_grid
from thesis_lib.util import save_plot, estimate_fwhm, psf_cramer_rao_bound, psf_to_fisher, dictoflists_to_listofdicts, cached
from thesis_lib.standalone_analysis.fitting_weights1D import fit, Gaussian1D, anderson_gauss, anderson, ones, xs, \
    fillqueue, plot_lambda_vs_precission_relative, fit_dictarg
from thesis_lib.standalone_analysis.psf_radial_average import psf_radial_reduce
from thesis_lib.standalone_analysis import fitting_weights1D

## use these for interactive, disable for export
plt.rcParams['figure.figsize'] = (9, 6)
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
cachedir = Path('cached_results/')
if not os.path.exists(outdir):
    os.mkdir(outdir)
#util.RERUN_ALL_CACHED=True

# %% [markdown]
# # PSF convolution

# %%
img_even, _ = convolved_grid(N1d=2, border=10, size=35,
                             kernel=AiryDisk2DKernel(radius=4, x_size=20, y_size=20),
                             perturbation=1., seed=0)
img_odd, position_table = convolved_grid(N1d=2, border=10, size=35,
                                         kernel=AiryDisk2DKernel(radius=4, x_size=21, y_size=21),
                                         perturbation=1., seed=0)

# %%
fig, axs = plt.subplots(1, 2)
plt.suptitle('applying PSF by convolution')
axs[0].imshow(img_even)
axs[0].plot(position_table['x'], position_table['y'], 'ro', markersize=3, label='reference position')
axs[0].legend()
axs[0].set_title('even kernel-size')

axs[1].imshow(img_odd)
axs[1].plot(position_table['x'], position_table['y'], 'ro', markersize=3, label='reference position')
axs[1].legend()
axs[1].set_title('odd kernel-size')

save_plot(outdir, 'odd_even_kernel')
# fig.tight_layout()

# %%
plt.figure()
plt.imshow(img_even - img_odd)
plt.title('difference even-odd')
save_plot(outdir, 'odd_even_difference')
plt.colorbar()

# %%
N = 10  # number of sources
r = 4  # width of airy

height = 30
width = 150

xs = np.linspace(15, width - 15, N).astype(np.int64) + np.linspace(0, 0.5, N)
ys = np.repeat(height / 2, N)

data_conv_template = np.zeros((height, width))
data_add = data_conv_template.copy()
y_grid, x_grid = np.indices(data_conv_template.shape)

for i in range(N):
    data_conv_template[int(ys[i]), int(xs[i])] = 1 - (xs[i] % 1)
    data_conv_template[int(ys[i]), int(xs[i]) + 1] = xs[i] % 1

    data_add += AiryDisk2D(radius=r, x_0=xs[i], y_0=ys[i])(x_grid, y_grid)

data_conv = convolve(data_conv_template, AiryDisk2DKernel(radius=r, x_size=31, y_size=31))

data_conv /= data_conv.max()
data_add /= data_add.max()

fig, axs = plt.subplots(3, 1, sharex=True)
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)

im0 = axs[0].imshow(data_conv_template)
fig.colorbar(im0, ax=axs[0], shrink=0.35, ticks=MaxNLocator(5))
axs[0].set_ylabel('convolution input')

im1 = axs[1].imshow(data_conv)
fig.colorbar(im1, ax=axs[1], shrink=0.35, ticks=MaxNLocator(5))
axs[1].set_ylabel('convolved')

diff = data_conv - data_add
im2 = axs[2].imshow(diff)
fig.colorbar(im2, ax=axs[2], shrink=0.35, ticks=MaxNLocator(5))
axs[2].set_ylabel('difference to\n model evaluation')

fig.subplots_adjust(left=0, right=1, wspace=-1, hspace=-0.8)
fig.tight_layout()

save_plot(outdir, 'pixelphase_distortion')

# %% [markdown]
# # Numerical limits
# ## identity fit

# %%
# model = Gaussian2D(x_stddev=5., y_stddev=5.)
model = AiryDisk2D(radius=5)
# model = make_anisocado_model(oversampling=2)

size_1D = 25j
xy_pairs = np.mgrid[0:0.5:size_1D, 0:0.5:size_1D].transpose((1, 2, 0)).reshape(-1, 2)

dl = {'input_model': [model],
      'n_sources': [1],
      'img_border': [32],
      'img_size': [64],
      'pixelphase': xy_pairs,
      'fitshape': [(21, 21)],
      'σ': [0],
      'λ': [None],
      'model_mode': ['same'],
      'fit_accuracy': [1.49012e-08],
      'fitter_name': ['LM', 'Simplex'],
      }
def do_it():
    with mp.Pool() as p:
        results = pd.DataFrame.from_records(p.map(fit_models_dictarg, tqdm(create_arg_list(dl)),2))
    results = transform_dataframe(results)
    return results
results = cached(do_it, cachedir/'gridded_identity_fit')

# %%
lm = results[results.fitter_name == 'LM']
simplex = results[results.fitter_name == 'Simplex']

max_deviation_lm = np.max(lm.dev)
max_deviation_simplex = np.max(simplex.dev)

print(f'LM: {max_deviation_lm}')
print(f'Simplex: {max_deviation_simplex}')

# %%
plot_phase_vs_deviation3d(simplex)

# %%
plot_phase_vs_deviation3d(lm)

# %% [markdown]
# ## gridded fit
#

# %%
model = AiryDisk2D(radius=5)

size_1D = 30j
xy_pairs = np.mgrid[0:0.5:size_1D, 0:0.5:size_1D].transpose((1, 2, 0)).reshape(-1, 2)

dl = {'input_model': [model],
      'n_sources': [1],
      'img_border': [32],
      'img_size': [100],
      'pixelphase': xy_pairs,
      'fitshape': [(21, 21)],
      'model_oversampling': [1, 2],  # this gives more or less bumps
      'model_degree': [3],  # this changes things here
      'model_mode': ['grid'],
      'fit_accuracy': [1.49012e-08]
      }

def do_it():
    with mp.Pool() as p:
        results = pd.DataFrame.from_records(p.map(fit_models_dictarg, tqdm(create_arg_list(dl)), 10))
    results = transform_dataframe(results)
    return results
results = cached(do_it, cachedir/'gridded_gridded_fit')

# %%
plot_phase_vs_deviation3d(results[results.model_oversampling == 1])
plt.title('direct fit')
plt.gcf().set_size_inches(7, 7)
plt.tight_layout()
save_plot(outdir, 'eggcarton_a')

# %%
plot_phase_vs_deviation3d(results[results.model_oversampling == 2])
plt.title('2x oversampling')
plt.gcf().set_size_inches(7, 7)
plt.tight_layout()
save_plot(outdir, 'eggcarton_b')

# %%
model = AiryDisk2D(radius=5)

size_1D = 20j
xy_pairs = np.mgrid[0:0.5:size_1D, 0:0.5:size_1D].transpose((1, 2, 0)).reshape(-1, 2)

dl = {'input_model': [AiryDisk2D(radius=5)],
      'n_sources': [1],
      'img_border': [32],
      'img_size': [64],
      'pixelphase': xy_pairs,
      'fitshape': [(21, 21)],
      'σ': [0],
      'λ': [None],
      'model_oversampling': [1, 2, 3, 4, 5, 6],
      'model_degree': [1, 2, 3, 4, 5],
      'model_mode': ['grid'],
      'fit_accuracy': [1.49012e-08]
      }
def do_it():
    with mp.Pool() as p:
        results = pd.DataFrame.from_records(p.map(fit_models_dictarg, tqdm(create_arg_list(dl))))
    results = transform_dataframe(results)
    return results
results = cached(do_it, cachedir/'gridded_model_degree_os')

# %%
degree_grouped = results.groupby(['model_degree'])
plt.figure()

fake_handles = []
labels = []

sigma_cut = 3

for i, (degree, degree_group) in enumerate(degree_grouped):
    # oversampling=group.groupby('model_oversampling').aggregate({'dev': ['min', 'max', 'mean', 'std']})
    # xs = oversampling.index
    # ys = oversampling.dev['mean']
    os_grouped = degree_group.groupby('model_oversampling')
    offset = 0  # (i/len(degree_grouped)-0.5)/3
    # remove all values where deviation is better than 3σ
    vp = plt.violinplot([os_group.dev[zscore(os_group.dev) > -sigma_cut] for (_, os_group) in os_grouped],
                        [oversampling + offset for (oversampling, _) in os_grouped],
                        showmeans=True, showextrema=False, widths=0.7)
    fake_handles.append(vp['cmeans'])
    labels.append(f'degree={degree}')

plt.xticks(results.model_oversampling.unique())
plt.yscale('log')
# plt.ylim(1e-12,1e-2)
plt.title(f'Deviations of gridded fit models clipped to σ={sigma_cut}')
plt.xlabel('oversampling')
plt.ylabel('centroid deviation')
plt.grid(which='both', axis='y')
plt.legend(fake_handles, labels)

save_plot(outdir, 'oversampling_interpolation_comp')

# %% [markdown]
# # Anisocado Model

# %%
anisocado = make_anisocado_model(oversampling=4, degree=5, offaxis=[0, 0])
estimate_fwhm(anisocado) * u.pixel

# %%
fig, axs = plt.subplots(1, 2)
im = axs[0].imshow(anisocado.data, norm=LogNorm())
plt.colorbar(im, ax=axs[0], shrink=0.6)
axs[0].set_title('On axis PSF at 2.15 μm')

im = axs[1].imshow(fftconvolve(anisocado.data, anisocado.data, mode='same'), norm=LogNorm())
axs[1].set_title('PSF auto-correlation')
plt.colorbar(im, ax=axs[1], shrink=0.6)

fig.set_size_inches(10, 5)
save_plot(outdir, 'anisocado_model')

# %%
size_1D = 20j
xy_pairs = np.mgrid[0:0.5:size_1D, 0:0.5:size_1D].transpose((1, 2, 0)).reshape(-1, 2)

dl = {'input_model': [anisocado],
      'n_sources': [1],
      'img_border': [32],
      'img_size': [64],
      'pixelphase': xy_pairs,
      'fitshape': [(21, 21)],
      'σ': [0],
      'λ': [None],
      'model_mode': ['same'],
      'fit_accuracy': [1.49012e-08],
      'fitter_name': ['LM', 'Simplex'],
      'fit_bounds': [None, (1.1, 1.1)]
      }

def do_it():
    with mp.Pool() as p:
        results = pd.DataFrame.from_records(p.map(fit_models_dictarg, tqdm(create_arg_list(dl))))

    results = transform_dataframe(results)
    results_constrained = results[~results.fit_bounds.isnull()]
    results = results[results.fit_bounds.isnull()]
    return results, results_constrained
results, results_constrained = cached(do_it, cachedir/'gridded_lm_v_simplex')

# %%
lm = results[results.fitter_name == 'LM']
simplex = results[results.fitter_name == 'Simplex']
print(f'''
LM diverged: {len(lm[lm.dev > 1])}/{len(lm)}
Simplex diverged: {len(simplex[simplex.dev > 1])}/{len(simplex)}
Constraint diverged: {len(results_constrained[results_constrained.dev > 1])}/{len(results_constrained)}
'''.strip())

# %%
plt.figure()
lm = results[(results.fitter_name == 'LM') & (results.dev < 1)]
simplex = results[(results.fitter_name == 'Simplex') & (results.dev < 1)]
plt.plot(simplex.x_dev, simplex.y_dev, 'o', label='Simplex')
plt.plot(lm.x_dev, lm.y_dev, 'o', label='LM')
plt.xlabel('x deviation')
plt.ylabel('y deviation')
plt.title('Fit deviation of anisocado model for successful fits')
plt.gcf().set_size_inches(7, 7)
plt.legend()
save_plot(outdir, 'anisocado_succ')

# %%
plt.figure()
unconstrained = results[(results.dev > 1)]
plt.plot(unconstrained.x_dev, unconstrained.y_dev, 'o', label=f'unconstrained N={len(unconstrained)}')
# constrained_lm = results_constrained[results_constrained.fitter_name=='LM']
# constrained_simplex = results_constrained[results_constrained.fitter_name=='Simplex']
# plt.plot(constrained_lm.x_dev, constrained_lm.y_dev, 'o', label='constrained LM')
# plt.plot(constrained_simplex.x_dev, constrained_simplex.y_dev, 'o', label='constrained Simplex')
plt.plot(results_constrained.x_dev, results_constrained.y_dev, 'o', label=f'constrained N={len(results_constrained)}')

plt.xlabel('x deviation')
plt.ylabel('y deviation')
plt.title('Diverged fits')
plt.gcf().set_size_inches(7, 7)
plt.legend()
save_plot(outdir, 'anisocado_fail')

# %% [markdown]
# # Noise

# %%
size_1D = 15j
xy_pairs = np.mgrid[0:0.5:size_1D, 0:0.5:size_1D].transpose((1, 2, 0)).reshape(-1, 2)
dl = {'input_model': [anisocado],
      'n_sources': [1],
      'img_border': [32],
      'img_size': [64],
      'pixelphase': xy_pairs,
      'fitshape': [(21, 21)],
      'σ': [0, 25],
      'λ': np.logspace(2, 6.5, 100),
      'model_oversampling': [2],
      'model_degree': [5],
      'model_mode': ['same'],
      'fit_accuracy': [1.49012e-08],
      'use_weights': [True],
      'fitter_name': ['LM']
      }

def do_it():
    with mp.Pool() as p:
        results = pd.DataFrame.from_records(p.map(fit_models_dictarg, tqdm(create_arg_list(dl))))
    results = transform_dataframe(results)
    return results
results = cached(do_it, cachedir/'gridded_noisevdev', rerun=False)
#results = results[results.dev < 10]
# %%
# calculate k*FWHM of anisocado PSF
k_times_fwhm = np.sqrt(np.sum(np.linalg.inv(psf_to_fisher(anisocado.data, oversampling=anisocado.oversampling[0]))))
k_times_fwhm

# %%
mod = make_anisocado_model(oversampling=1)
def plot_deviation_vs_noise(results: pd.DataFrame):

    # issue: pickling destroys type identity, so we need to convert to strings
    # select all distinct models and distinct noise /types/
    grouped = results.groupby([results.input_model.astype(str), results.σ], as_index=False)

    plt.figure()
    for i, ((model_str, σ), group) in enumerate(grouped):
        # for each noisetype, plot magnitude vs deviation
        noisegroup = group.groupby('λ')
        plt.errorbar(noisegroup.first().index, noisegroup.mean().dev, noisegroup.std().dev,
                     errorevery=(i, len(grouped)),
                     fmt='o', label=f'{repr(group.input_model.iloc[0])} σ={σ}', alpha=0.8)

    plt.legend(fontsize='small')
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Total Flux')

    plt.ylabel('Mean Deviation in measured Position')
    
plot_deviation_vs_noise(results)


grpd = results.groupby(['λ','σ'], as_index=False).mean()

deviation0 = np.array([psf_cramer_rao_bound(
        mod.data, n_photons=λ, constant_noise_variance=0, oversampling=mod.oversampling[0])
    for λ in grpd.λ])
deviation25 = np.array([psf_cramer_rao_bound(
        mod.data, n_photons=λ, constant_noise_variance=25**2, oversampling=mod.oversampling[0])
    for λ in grpd.λ])


plt.plot(grpd[grpd.σ==0].λ, k_times_fwhm / grpd[grpd.σ==0].snr, ':', label=f'k FWHM/SNR = {k_times_fwhm:.3f}/SNR σ=0', color='C0')
plt.plot(grpd[grpd.σ==25].λ, k_times_fwhm / grpd[grpd.σ==25].snr, ':', label=f'k FWHM/SNR = {k_times_fwhm:.3f}/SNR σ=25', color='C1')

plt.plot(grpd.λ, deviation0,'--', label='Cramér-Rao bound σ=0', color='C0')
plt.plot(grpd.λ, deviation25,'--', label='Cramér-Rao bound σ=25', color='C1')

m = (grpd.peak_flux.iloc[-1]-grpd.peak_flux.iloc[1])/(grpd.λ.iloc[-1]-grpd.λ.iloc[1])
t = grpd.peak_flux.iloc[-1]-m*grpd.λ.iloc[-1]
def flux_to_peak(flux):
    return m*flux+t
def peak_to_flux(peak):
    return peak/m - t/m

secax = plt.gca().secondary_xaxis(-0.13, functions=(flux_to_peak, peak_to_flux))
secax.set_xlabel('Peak Flux')
plt.legend()

save_plot(outdir, 'deviation_vs_noise_anisocado')

# %%
#grpd = results[results.σ == 0].groupby(['λ'], as_index=False)
#fig, axs = plt.subplots(2, 1)
#axs[0].semilogy(grpd.std().λ, grpd.std().x_dev, label='xdev')
#axs[0].semilogy(grpd.std().λ, grpd.std().y_dev, label='ydev')
#axs[0].semilogy(grpd.std().λ, grpd.std().dev, label='dev')
#axs[0].semilogy(grpd.std().λ, deviation0, label='CRLB')
#axs[0].legend()
#axs[0].set_title('std')

#magg = grpd.agg(lambda group: np.mean(np.abs(group)))
#axs[1].semilogy(magg.λ, magg.x_dev, label='xdev')
#axs[1].semilogy(magg.λ, magg.y_dev, label='ydev')
#axs[1].semilogy(magg.λ, magg.dev, label='dev')
#axs[1].semilogy(magg.λ, deviation0, label='CRLB')
#axs[1].legend()
#axs[1].set_title('mean')
pass


# %% [markdown]
# # Fitshape
#

# %%
def plot_xy_fitshape_scatter(results):
    fig = plt.figure()
    cmap = 'turbo'
    plt.gca().set_aspect('equal')


    shapes = results.fitshape.apply(lambda shape: shape[0])

    plt.scatter(results.x_dev, results.y_dev, c=shapes, cmap=cmap, alpha=0.4, lw=0)
    plt.xlabel('x_dev')
    plt.ylabel('y_dev')

    plt.gca().set_aspect('equal')
    plt.colorbar(ScalarMappable(norm=Normalize(shapes.min(), shapes.max()), cmap=cmap)).set_label('Fitshape')

# %%
anisocado = make_anisocado_model(oversampling=2, degree=3)

fitshapes = [(i, i) for i in range(3, 60)]

dl = {'input_model': [anisocado],
      'n_sources': [1, 4],
      'img_border': [70],
      'img_size': [70 * 2 + 25],  # 40 pixel separation beween source positions
      'pixelphase': ['random'] * 25,
      'fitshape': fitshapes,
      'σ': [1.],
      'λ': [500_000],
      'model_mode': ['grid'],
      'model_degree': [3],
      'fit_accuracy': [1.49012e-08],
      'use_weights': [True, False]
      }

def do_it():
    with mp.Pool() as p:
        results = pd.DataFrame.from_records(p.map(fit_models_dictarg, tqdm(create_arg_list(dl))))
    results = transform_dataframe(results)
    results = results[results.dev < 1]
    return results
results = cached(do_it, cachedir/'gridded_fitshape', rerun=False)
results=results[results.dev>1e-4]


# %%
def calc_expected_deviation(input_model, fitshape, σ, λ):
    os = input_model.oversampling[0]
    y,x = np.mgrid[-fitshape/2:fitshape/2:os*fitshape*1j,
                   -fitshape/2:fitshape/2:os*fitshape*1j]
    psf_img = input_model(x,y)

    return psf_cramer_rao_bound(psf_img, constant_noise_variance=σ**2, n_photons=λ, oversampling=os)

def calc_kfwhm(input_model, fitshape, σ, λ):
    return psf_cramer_rao_bound(input_model.data, constant_noise_variance=0, n_photons=1, oversampling=input_model.oversampling[0])

for (input_model,fitshape,σ,λ), group in results.groupby([results.input_model.astype('str'),'fitshape','σ','λ'],as_index=False):
    input_model = anisocado
    results.loc[group.index,'expected_deviation'] = calc_expected_deviation(input_model,fitshape[0],σ,λ)
    results.loc[group.index,'kfwhm'] = calc_kfwhm(input_model,fitshape[0],σ,λ)
    
results['kfwhm_over_snr'] = results.kfwhm/results.snr
results['ratio'] = results.expected_deviation/results.kfwhm_over_snr


# %%
#plot_xy_fitshape_scatter(results[(results.n_sources == 4) & (results.use_weights == False)])
#save_plot(outdir, 'fitshape_xy_unweighted')

# %%
#plot_xy_fitshape_scatter(results[(results.n_sources == 4) & (results.use_weights == True)])
#save_plot(outdir, 'fitshape_xy_weighted')

# %%
def plot_fitshape(results: pd.DataFrame, ax):
    grouped = results.groupby('n_sources',
                              as_index=False)

    #fig, axes = plt.subplots(len(grouped), sharex='all')
    for i, (n_sources, modelgroup) in enumerate(grouped):
        xs = [i[0] for i in modelgroup.groupby('fitshape').mean().index]
        ys = modelgroup.groupby('fitshape').mean().dev
        errs = modelgroup.groupby('fitshape').std().dev
        ax.plot(xs, ys,
                 label=f'{n_sources} sources', color=f'C{i}')
        ax.fill_between(xs, ys - errs, ys + errs, alpha=0.3)
        
        grpd = modelgroup.groupby(modelgroup.fitshape.apply(lambda s: s[0])).mean()
        ax.plot(grpd.index, grpd.kfwhm / grpd.snr, ':', label='k FWHM/SNR',color=f'C{i}')
    
    grpd = results.groupby('fitshape',as_index=False).min()
    ax.plot([i[0] for i in grpd.fitshape], grpd.expected_deviation, '--', label='Crámer-Rao bound', color='C2')

    ax.set_xlabel('Cutout size of fit around star')    
    ax.axvline(2*25, lw=0.5)
    ax.legend()
    ax.set_yscale('log')

fig, axs = plt.subplots(1, 2, sharey=True)
axs[0].set_ylabel('mean position deviation')
fig.set_size_inches(11,5.6)
    
results_weighted = results[results.use_weights==True]
plot_fitshape(results_weighted, axs[0])
axs[0].set_title('weighted fit')
results_unweighted = results[results.use_weights==False]
plot_fitshape(results_unweighted, axs[1])
axs[1].set_title('unweighted fit')

plt.ylim(5e-4,0.04)
plt.tight_layout()
save_plot(outdir, 'crowding_deviation')

# %%
ref_img = fit_models_dictarg({'input_model': anisocado,
                              'n_sources': 4,
                              'img_border': 70,
                              'img_size': 70 * 2 + 25,  # 40 pixel separation beween source positions
                              'pixelphase': 'random',
                              'fitshape': 23,
                              'σ': 5.,
                              'λ': 500_000,
                              'model_mode': 'same',
                              'fit_accuracy': 1.49012e-08,
                              'use_weights': False,
                              'return_imgs': True})['img']
plt.figure()
plt.imshow(ref_img)
plt.colorbar().set_label('pixel count')
save_plot(outdir, 'crowding_img')

# %% [markdown]
# # Weights
#

# %%
model = fitting_weights1D.Gaussian1D()


class DummySeedQueue:
    def get(_):
        return 0


res_anderson = fitting_weights1D.fit(model, σ=15, λ=1000, weight_calc=fitting_weights1D.anderson, iters=1, seed_queue=DummySeedQueue(), return_arrays=True)[0]
res_anderson_gauss = \
fitting_weights1D.fit(model, σ=15, λ=5000, weight_calc=fitting_weights1D.anderson_gauss, iters=1, seed_queue=DummySeedQueue(), return_arrays=True)[0]


# %%
def plot_foo(res):
    weights = res['weights']
    ys = res['ys']

    plt.figure()
    plt.plot(fitting_weights1D.xs, ys, label='signal')
    plt.ylabel('signal value')
    plt.legend(loc='upper left')
    plt.xlabel('arbitrary units')

    plt.twinx()
    plt.plot([], [])
    effective_weights = (ys * weights) ** 2
    effective_weights /= effective_weights.max()
    plt.plot(fitting_weights1D.xs, weights, label='weights')
    plt.plot(fitting_weights1D.xs, effective_weights, label='effective weights')
    plt.legend(loc='upper right')
    plt.ylabel('weights')


plot_foo(res_anderson)
save_plot(outdir, '1d_anderson')
plot_foo(res_anderson_gauss)
save_plot(outdir, '1d_anderson_gauss')

# %%
manager = mp.Manager()
q = manager.Queue(5000)
queueproc = mp.Process(target=fitting_weights1D.fillqueue, args=(q,))
queueproc.start()

# %%
model = fitting_weights1D.Gaussian1D()

params = {
    'λ': [λ for λ in np.linspace(100, 50_000, 150)],
    'σ': [σ for σ in np.linspace(0, 20, 30)],
    'weight_calc': [fitting_weights1D.ones, fitting_weights1D.anderson, fitting_weights1D.anderson_gauss],
    'model': [model],
    'iters': [200],
    'seed_queue': [q],
    'return_arrays': [False],
}

def do_it():
    with mp.Pool() as p:
        records_pic = chain(*p.map(weights1d.fit_dictarg, tqdm(dictoflists_to_listofdicts(params)), 100))
    results_pic = pd.DataFrame.from_records(records_pic)
    #small_results = results_pic[results_pic.columns.difference(['fitted'])]
    return results_pic
#results_pic = do_it()
results_pic = cached(do_it, cachedir/'gridded_1Dsigmalambdaresults')

# %%
results_pic = results_pic.replace('anderson_gauss', 'inverse variance')

# %%
figs = fitting_weights1D.plot_lambda_vs_precission_relative(results_pic)
plt.title('Poisson + Gaussian noise $0<σ<20$')
plt.text(10000, 1.05, 'weighted worse')
plt.text(10000, 0.95, 'weighted better')
save_plot(outdir, 'weights_methods1d')

# %% [markdown]
# # Radial averages of PSF

# %%
psf = AnalyticalScaoPsf(pixelSize=0.004*0.5, N=1025)
on_axis = psf.psf_on_axis
off_axis = psf.shift_off_axis(10, 10)

xs, on_axis_min = psf_radial_reduce(on_axis, np.min)
on_axis_mean = psf_radial_reduce(on_axis, np.mean)[1]
on_axis_max = psf_radial_reduce(on_axis, np.max)[1]

xs, off_axis_min = psf_radial_reduce(off_axis, np.min)
off_axis_mean = psf_radial_reduce(off_axis, np.mean)[1]
off_axis_max = psf_radial_reduce(off_axis, np.max)[1]

# %%
plt.figure()

plt.semilogy(xs, on_axis_mean, label='on-axis mean')
plt.fill_between(xs, on_axis_max, on_axis_min, alpha=0.65, label='on-axis min, max')

plt.semilogy(xs, off_axis_mean, label='off-axis mean')
plt.fill_between(xs, off_axis_min, off_axis_max, alpha=0.65, label='off-axis min, max')

plt.xlim(-5,150)

plt.xlabel('radius from PSF center [px]')
plt.ylabel('fractional PSF value')
plt.legend()
save_plot(outdir, 'radial_reduce')

# %% [markdown]
# # SNR calculation

# %%
anisocado = make_anisocado_model(oversampling=4,degree=5)

y, x = np.mgrid[-100:100:401j, -100:100:401j]
airy = FittableImageModel(AiryDisk2D(radius=5)(x,y)+0.001, oversampling=2)

dl = {'input_model': [anisocado,airy],
      'n_sources': [1],
      'img_border': [64],
      'img_size': [64 * 2],  # 40 pixel separation beween source positions
      'pixelphase': ['random'] * 10,
      'fitshape': [(21,21)],
      'σ': [0.,25.],
      'λ': [1000, 20_000, 500_000],
      'model_mode': ['same'],
      'fit_accuracy': [1.49012e-08],
      'use_weights': [True]
      }

def do_it():
    with mp.Pool() as p:
        results = pd.DataFrame.from_records(p.map(fit_models_dictarg, tqdm(create_arg_list(dl)),2))
    results = transform_dataframe(results)
    return results

results = cached(do_it, cachedir/'gridded_snr',rerun=False)


# %%
def calc_expected_deviation(input_model, fitshape, σ, λ):
    y,x = np.mgrid[-fitshape/2:fitshape/2:fitshape*1j,
                   -fitshape/2:fitshape/2:fitshape*1j]
    psf_img = input_model(x,y)

    return psf_cramer_rao_bound(psf_img, constant_noise_variance=σ**2, n_photons=λ)

def calc_kfwhm(input_model, fitshape, σ, λ):
    return psf_cramer_rao_bound(input_model.data, constant_noise_variance=0, n_photons=1, oversampling=input_model.oversampling[0])

for (input_model,fitshape,σ,λ), group in results.groupby([results.input_model.astype('str'),'fitshape','σ','λ'],as_index=False):
    input_model = anisocado if 'anisocado' in input_model else airy
    results.loc[group.index,'expected_deviation'] = calc_expected_deviation(input_model,fitshape[0],σ,λ)
    results.loc[group.index,'kfwhm'] = calc_kfwhm(input_model,fitshape[0],σ,λ)
    
results['kfwhm_over_snr'] = results.kfwhm/results.snr
results['ratio'] = results.expected_deviation/results.kfwhm_over_snr

# %%
tab = results.groupby([results.input_model.apply(lambda model: repr(model)[1:15]), 'σ', 'λ']).mean()[['snr','kfwhm', 'kfwhm_over_snr','expected_deviation', 'ratio']]
tab

# %%
print(tab.to_latex(float_format="{:0.2f}".format))

# %%
from thesis_lib.config import Config
import thesis_lib.util as util

with util.work_in(Config.scopesim_working_dir):
    fpa_lin=Path('./inst_pkgs/MICADO/FPA_linearity.dat')
    data = np.genfromtxt(fpa_lin)[1:-1]

incident = data[:,0]
measured = data[:,1]

plt.figure()
plt.plot(incident, measured)
plt.xlabel('incident [photons]')
plt.ylabel('measured [ADU]')
plt.title('ScopeSim MICADO detector linearity curve')
save_plot(outdir, 'scopesim_linearity')

# %%
print('script successful')
plt.close('all')
