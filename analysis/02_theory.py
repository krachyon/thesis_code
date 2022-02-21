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
import matplotlib.pyplot as plt
import numpy as np
from thesis_lib.scopesim_helper import AnisocadoModel, make_anisocado_model
import anisocado
import multiprocess as mp
import itertools
from functools import reduce
from operator import mul
from pathlib import Path
import matplotlib as mpl
from matplotlib.colors import LogNorm
from tqdm.auto import tqdm
from thesis_lib.util import save_plot, psf_to_fisher,psf_cramer_rao_bound, cached
import os
from IPython.display import HTML

# %%
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10
plt.rcParams['figure.autolayout'] = True

outdir = './02_gridded_models/'
cachedir = Path('cached_results/')
if not os.path.exists(outdir):
    os.mkdir(outdir)

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
# maybe finer spacing, esp in center?
shape = (60, 60, 2, 2)
shared_array = mp.Array('d', reduce(mul, shape))

def make_array():    
    psf = anisocado.AnalyticalScaoPsf()

    def inv_fisher(psf, shift):
        psf_img = psf.shift_off_axis(*shift)
        out_array = np.frombuffer(shared_array.get_obj()).reshape(shape)
        out_array[shift[0],shift[1],:,:] = np.linalg.inv(psf_to_fisher(psf_img))

    with mp.Pool() as p:
        args = list(zip(itertools.repeat(psf),[(i,j) for i in range(shape[0]) for j in range(shape[1])]))
        p.starmap(inv_fisher, tqdm(args) , 10)

    out_array = np.frombuffer(shared_array.get_obj()).reshape(shape)
    return out_array


out_array = cached(make_array, cachedir/'inv_fisher')

# %%
fig, axs = plt.subplots(2,2, sharex=True, sharey=True)
# diagonal elements, off diagonal, first and second on diagonal
imgs = np.array([np.sqrt(np.sum(out_array**2,axis=(2,3))), out_array[:,:,0,1],
        out_array[:,:,0,0], out_array[:,:,1,1]])
titles = ['diag', 'off diag', 'expected $y$-deviation', 'expected $x$-deviation']

imgs[imgs<=0] = 1e-4
mi, ma = np.min(np.sqrt(imgs)), np.max(np.sqrt(imgs))
for i,(img, title) in enumerate(zip(imgs, titles)):
    ax = axs.flat[i]

    im = ax.imshow(np.sqrt(img))
    ax.set_title(title)
    ax.set_ylabel('y off-axis shift [as]')
    ax.set_xlabel('x off-axis shift [as]')
    fig.colorbar(im, ax=ax, label='$\sqrt{\mathcal{I}^{-1}}$')
fig.suptitle('Inverse Fisher matrices for off-axis shift of Anisocado-PSF')
save_plot(outdir, 'psf_fisher_matrix')
pass

# %%
euc_img = np.sqrt(out_array[:,:,0,0]**2+out_array[:,:,1,1]**2)
cov_img = np.sqrt(2)*out_array[:,:,1,0]
y,x = np.indices(euc_img.shape)
r = np.sqrt(x**2+y**2)

plt.figure()
plt.plot(r.flatten(), np.sqrt(euc_img.flatten()), '.', label='diagonal matrix elements', alpha=0.9)
plt.plot(r.flatten(), np.sqrt(cov_img.flatten()), '.', label='off-diagonal elements', alpha=0.9)
plt.ylabel('magnitude of deviation [px]')
plt.xlabel('Combined PSF off-center shift [as]')
plt.legend()

save_plot(outdir, 'psf_radial_deviation')
pass
#ax2=plt.gca().twinx()
#ax2.plot(r.flatten(), np.sqrt(cov_img.flatten())/np.sqrt(euc_img.flatten()), 'o', color='C2')
#ax2.set_ylabel('relative')


# %%
psf = anisocado.AnalyticalScaoPsf().shift_off_axis(0,0)
def id(N):
    return np.sqrt(N)*psf_cramer_rao_bound(psf,constant_noise_variance=0,n_photons=N)
    
id(1),id(5)    

# %%
