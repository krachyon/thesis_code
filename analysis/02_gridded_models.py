# -*- coding: utf-8 -*-
# %%
# %matplotlib notebook
# %pylab
import matplotlib.pyplot as plt
import numpy as np

from itertools import chain
import os
import multiprocess as mp

import astropy.units as u
from astropy.convolution.kernels import AiryDisk2DKernel
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from scipy.signal import fftconvolve
from scipy.stats import zscore
from tqdm.auto import tqdm

from thesis_lib.standalone_analysis.sampling_precision import *
from thesis_lib.testdata.recipes import convolved_grid
from thesis_lib.util import save_plot, estimate_fwhm

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
outdir = './02_gridded_models/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

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
from astropy.modeling.functional_models import AiryDisk2D
from astropy.convolution import AiryDisk2DKernel, convolve

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

with mp.Pool() as p:
    results = pd.DataFrame.from_records(p.map(fit_models_dictarg, tqdm(create_arg_list(dl))))
results = transform_dataframe(results)

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

with mp.Pool() as p:
    results = pd.DataFrame.from_records(p.map(fit_models_dictarg, tqdm(create_arg_list(dl)), 10))
results = transform_dataframe(results)

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

with mp.Pool() as p:
    results = pd.DataFrame.from_records(p.map(fit_models_dictarg, tqdm(create_arg_list(dl))))
results = transform_dataframe(results)

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
axs[1].set_title('PSF cross correlation')
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

with mp.Pool() as p:
    results = pd.DataFrame.from_records(p.map(fit_models_dictarg, tqdm(create_arg_list(dl))))

results = transform_dataframe(results)
results_constrained = results[~results.fit_bounds.isnull()]
results = results[results.fit_bounds.isnull()]

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
size_1D = 10j
xy_pairs = np.mgrid[0:0.5:size_1D, 0:0.5:size_1D].transpose((1, 2, 0)).reshape(-1, 2)
dl = {'input_model': [anisocado],
      'n_sources': [1],
      'img_border': [32],
      'img_size': [64],
      'pixelphase': xy_pairs,
      'fitshape': [(21, 21)],
      'σ': [0, 25, 50],
      'λ': np.logspace(2, 5.5, 100),
      'model_oversampling': [2],
      'model_degree': [5],
      'model_mode': ['grid'],
      'fit_accuracy': [1.49012e-08],
      'use_weights': [True]
      }

with mp.Pool() as p:
    results = pd.DataFrame.from_records(p.map(fit_models_dictarg, tqdm(create_arg_list(dl))))
results = transform_dataframe(results)
results = results[results.dev < 1]

# %%
plot_deviation_vs_noise(results)

grpd = results.groupby(['λ'], as_index=False).mean()
plt.plot(grpd.λ, estimate_fwhm(anisocado) / grpd.snr, ':', label='FWHM/SNR')
plt.legend()

save_plot(outdir, 'deviation_vs_noise_anisocado')


# %% [markdown]
# # Fitshape
#

# %%
def plot_xy_fitshape_scatter(results):
    fig = plt.figure()
    cmap = 'turbo'

    shapes = results.fitshape.apply(lambda shape: shape[0])

    plt.scatter(results.x_dev, results.y_dev, c=shapes, cmap=cmap, alpha=0.4, lw=0)
    plt.xlabel('x_dev')
    plt.ylabel('y_dev')

    plt.colorbar(ScalarMappable(norm=Normalize(shapes.min(), shapes.max()), cmap=cmap)).set_label('Fitshape')


# %%
anisocado = make_anisocado_model(oversampling=2)

fitshapes = [(i, i) for i in range(3, 71)]

dl = {'input_model': [anisocado],
      'n_sources': [1, 4],
      'img_border': [64],
      'img_size': [64 * 2 + 40],  # 40 pixel separation beween source positions
      'pixelphase': ['random'] * 25,
      'fitshape': fitshapes,
      'σ': [5.],
      'λ': [500_000],
      'model_mode': ['same'],
      'fit_accuracy': [1.49012e-08],
      'use_weights': [True, False]
      }

with mp.Pool() as p:
    results = pd.DataFrame.from_records(p.map(fit_models_dictarg, tqdm(create_arg_list(dl))))
results = transform_dataframe(results)
results = results[results.dev < 1]

# %%
plot_xy_fitshape_scatter(results[(results.n_sources == 4) & (results.use_weights == False)])
save_plot(outdir, 'fitshape_xy_unweighted')

# %%
plot_xy_fitshape_scatter(results[(results.n_sources == 4) & (results.use_weights == True)])
save_plot(outdir, 'fitshape_xy_weighted')

# %%
grpd = results.groupby(results.fitshape.apply(lambda s: s[0])).mean()

plot_fitshape(results)
plt.plot(grpd.index, estimate_fwhm(anisocado) / grpd.snr, ':', label='FWHM/SNR')
plt.legend()
plt.yscale('log')
save_plot(outdir, 'crowding_deviation')

# %%
ref_img = fit_models_dictarg({'input_model': anisocado,
                              'n_sources': 4,
                              'img_border': 64,
                              'img_size': 64 * 2 + 40,  # 40 pixel separation beween source positions
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
from thesis_lib.standalone_analysis.fitting_weights1D import fit, Gaussian1D, anderson_gauss, anderson, ones, xs, \
    fillqueue, anderson_gauss_ramp, plot_lambda_vs_precission_relative, fit_dictarg

# %%
model = Gaussian1D()


class DummySeedQueue:
    def get(_):
        return 0


res_anderson = fit(model, σ=15, λ=1000, weight_calc=anderson, iters=1, seed_queue=DummySeedQueue(), return_arrays=True)[
    0]
res_anderson_gauss = \
fit(model, σ=15, λ=5000, weight_calc=anderson_gauss, iters=1, seed_queue=DummySeedQueue(), return_arrays=True)[0]


# %%
def plot_foo(res):
    weights = res['weights']
    ys = res['ys']

    plt.figure()
    plt.plot(xs, ys, label='signal')
    plt.ylabel('signal value')
    plt.legend(loc='upper left')
    plt.xlabel('arbitrary units')

    plt.twinx()
    plt.plot([], [])
    effective_weights = (ys * weights) ** 2
    effective_weights /= effective_weights.max()
    plt.plot(xs, weights, label='weights')
    plt.plot(xs, effective_weights, label='effective weights')
    plt.legend(loc='upper right')
    plt.ylabel('weights')


plot_foo(res_anderson)
save_plot(outdir, '1d_anderson')
plot_foo(res_anderson_gauss)
save_plot(outdir, '1d_anderson_gauss')

# %%
manager = mp.Manager()
q = manager.Queue(5000)
queueproc = mp.Process(target=fillqueue, args=(q,))
queueproc.start()

# %%
model = Gaussian1D()

params = {
    'λ': [λ for λ in np.linspace(100, 50_000, 150)],
    'σ': [σ for σ in np.linspace(0, 50, 80)],
    'weight_calc': [ones, anderson, anderson_gauss, anderson_gauss_ramp],
    'model': [model],
    'iters': [300],
    'seed_queue': [q],
    'return_arrays': [False],
}
resname = 'sigmalambdaresults.pkl.gz'
if os.path.exists(resname):
    results_pic = pd.read_pickle(resname)
else:
    with mp.Pool() as p:
        records_pic = chain(*p.map(fit_dictarg, tqdm(dictoflists_to_listofdicts(params)), 10))
        results_pic = pd.DataFrame.from_records(records_pic)
        small_results = results_pic[results_pic.columns.difference(['fitted'])]
        small_results.to_pickle(resname)

# %%
results_pic = results_pic.replace('anderson_gauss', 'inverse variance')

# %%
figs = plot_lambda_vs_precission_relative(
    results_pic[(results_pic.σ < 20) & (results_pic.weight_calc != 'anderson_gauss_ramp')])
plt.title('Poisson + Gaussian noise $0<σ<20$')
plt.text(10000, 1.05, 'weighted worse')
plt.text(10000, 0.95, 'weighted better')
save_plot(outdir, 'weights_methods1d')

# %%
print('script succesfull')
plt.close('all')
