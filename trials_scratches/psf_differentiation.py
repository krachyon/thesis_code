# %%
# %matplotlib notebook
# %pylab
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10
plt.rcParams['figure.autolayout'] = True
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
from thesis_lib.util import save_plot


# %% [markdown]
# # Compost

# %%
def psf_to_sigma(psf_img, N, σ, axis, oversampling=1):
    """Formula from
    Lindegren, Lennart. “High-Accuracy Positioning: Astrometry.” ISSI Scientific Reports Series 9 (January 1, 2010): 279–91.

    to estimate variance of position estimate
    """
    normalized_psf = psf_img / np.trapz(np.trapz(psf_img, dx=1 / oversampling), dx=1 / oversampling)

    prefactor = N / (normalized_psf + σ ** 2 / N)
    grad = np.gradient(normalized_psf, 1 / oversampling, axis=axis) ** 2

    res = 1 / np.sqrt(np.trapz(np.trapz(prefactor * grad, dx=1 / oversampling), dx=1 / oversampling))
    return res


def psf_to_sigma_euc(psf_img, N, σ, oversampling=1):
    return np.sqrt(
        psf_to_sigma(psf_img, N, σ, axis=0, oversampling=oversampling) ** 2 +
        psf_to_sigma(psf_img, N, σ, axis=1, oversampling=oversampling) ** 2)

# %%
def likelyhood(detected_img, psf_model):
    out = np.zeros_like(detected_img)
    ys, xs = np.indices(detected_img.shape)
    for y, x in zip(ys.flatten(), xs.flatten()):
        psf_model.x_0 = x
        psf_model.y_0 = y
        if detected_img[y, x]:
            out += detected_img[y, x] * log(psf_model(xs, ys))[::-1, ::-1]
    return out


# %%
detections = np.zeros((31, 31))
detections[15, 15] = 2
detections[16, 20] = 1
detections[3, 8] = 1

model = make_anisocado_model()
like = likelyhood(detections, model)
y, x = np.indices(detections.shape)
plt.figure()
model.x_0 = 15
plt.imshow(like)

# %%

# %%

# %%
import anisocado

psf = anisocado.AnalyticalScaoPsf()
img = psf.psf_latest

# %%

# %%
Ns = np.logspace(2, 8, 1000)
plt.figure()
plt.loglog(Ns, [psf_to_sigma(img, N, 10, 0) for N in Ns])
plt.loglog(Ns, [psf_to_sigma(img, N, 5, 0) for N in Ns])
plt.loglog(Ns, [psf_to_sigma(img, N, 2, 0) for N in Ns])
plt.loglog(Ns, [psf_to_sigma(img, N, 0, 0) for N in Ns])
plt.loglog(Ns, [psf_to_sigma_d(img, N, 10, 0) for N in Ns])
plt.loglog(Ns, [psf_to_sigma_d(img, N, 5, 0) for N in Ns])
plt.loglog(Ns, [psf_to_sigma_d(img, N, 2, 0) for N in Ns])
plt.loglog(Ns, [psf_to_sigma_d(img, N, 0, 0) for N in Ns])
# plt.loglog(Ns,[psf_to_sigma(img, N, 50, 1) for N in Ns])
# plt.loglog(Ns,[psf_to_sigma(img, N, 10, 0) for N in Ns])
# plt.loglog(Ns,[psf_to_sigma(img, N, 0, 0) for N in Ns])

# %%
from astropy.modeling.functional_models import Gaussian2D, AiryDisk2D

y, x = np.mgrid[-10:10:512j, -10:10:512j]
g3 = Gaussian2D(x_stddev=3, y_stddev=3)(x, y)
g15 = Gaussian2D(x_stddev=1.5, y_stddev=1.5)(x, y)
airy = AiryDisk2D(radius=3)(x, y)
figure()

plt.loglog(Ns, [psf_to_sigma(g3, N, 0, 0) for N in Ns])
plt.loglog(Ns, [psf_to_sigma(g15, N, 0, 0) for N in Ns])
plt.loglog(Ns, [psf_to_sigma(airy, N, 0, 0) for N in Ns])
plt.loglog(Ns, [psf_to_sigma(img, N, 0, 0) for N in Ns])

# %%
from sympy import Function, Sum, symbols, ln, Eq
from sympy.abc import N, i

η, ζ = symbols('ζ,η')

P = Function('P')(ζ, η)
l = Sum(ln(P), (i, 1, N))
l

# %%
l.diff(η, 2).simplify()

# %%
(l.diff(η, 1) ** 2).simplify()

# %%
from astropy.modeling.functional_models import Gaussian1D


# %%
def fisher(pdf_grid, dx):
    normed = pdf_grid / np.trapz(pdf_grid, dx=dx)

    grad = np.gradient(normed, dx)

    return np.trapz(grad ** 2 / normed, dx=dx)


# %%
mi, ma, s = -30, 30, 1001
xs = np.linspace(mi, ma, s)
dx = (ma - mi) / s
pdf = Gaussian1D(stddev=1)(xs)
pdf /= np.trapz(pdf, dx=dx)

print(np.trapz(pdf, dx=dx))
print(np.sum(pdf))

# %%
mi, ma, s = -30, 30, 1001
xs = np.linspace(mi, ma, s)
dx = (ma - mi) / s

pdf = Gaussian1D(stddev=1)(xs)
pdf /= np.trapz(pdf, dx=dx)

plt.figure()
plt.plot(xs, pdf)
plt.plot(xs, np.gradient(pdf, dx))
plt.plot(xs, np.cumsum(np.gradient(pdf, dx) * dx))

# %%
mi, ma, s = -2, 2, 1000
xs = np.linspace(mi, ma, s)
dx = (ma - mi) / s

ys = 2 * xs ** 3 + xs ** 2 - 3 * xs
dys = 6 * xs ** 2 + 2 * xs - 3

figure()
plt.plot(xs, ys, label='f')
plt.plot(xs, np.gradient(ys, dx), label="num f'")
plt.plot(xs, dys, label="ana f'")
plt.plot(xs, np.cumsum(dys * dx) + ys[0], label='num ∫f')
plt.legend()


# %%
def var(x, pdf):
    return E(x ** 2, pdf) - E(x, pdf) ** 2


def E(x, pdf):
    return np.sum(pdf(x) * x) / np.sum(pdf(x))


def E2(x, y, pdf):
    return np.sum(x * pdf(x, y)) / np.sum(pdf(x, y)), np.sum(y * pdf(x, y)) / np.sum(pdf(x, y))


# %%
from scipy.stats import norm

# %%
mi, ma, s = -10, 10, 1001
xs = np.linspace(mi, ma, s)
dx = (ma - mi) / s

np.trapz(norm.pdf(xs), dx=dx)

# %%
