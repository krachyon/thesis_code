import numpy as np

from thesis_lib.sampling_precision import *
from matplotlib.colors import LogNorm
from scipy.signal import fftconvolve

from itertools import tee
from copy import copy

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def tracing_call(self, *args, **kwargs):
    self.history.append(self.parameters.copy())
    return self._call_inner(*args, **kwargs)


class TracingModel(object):
    def __init__(self, wrapped, history=None):
        self.wrapped = wrapped
        self.history = history if history else []

    def __call__(self, *args, **kwargs):
        self.history.append(self.wrapped.parameters.copy())
        return self.wrapped(*args, **kwargs)

    def __len__(self):
        return len(self.wrapped)

    def copy(self):
        return TracingModel(self.wrapped.copy(), copy(self.history))

    def __getattr__(self, item):
        return getattr(self.wrapped, item)

    def __setattr__(self, key, value):
        # sketchy, is there an easier way to whitelist these?
        if key in ['wrapped', 'history']:
            super().__setattr__(key, value)
        else:
            setattr(self.wrapped, key, value)


def set_call_trace(model):
    return TracingModel(model)

# I think this fails due to __call__ getting rerouted in a bad way
# def set_call_trace(model: Fittable2DModel):
#     model.history = []
#     if not hasattr(model.__class__, '_call_inner'):
#         # Note: This is Evil, as it needs to change the class, not just the instance
#         # Magic (__foo__) methods are looked up on the class itself
#         model.__class__._call_inner = model.__call__
#         model.__class__.__call__ = tracing_call
#     return model
#
# class TracingAnisocadoModel(AnisocadoModel):
#
#     def __init__(self, *args, **kwargs):
#         self.history = []
#         super().__init__(*args, **kwargs)
#
#     def __call__(self, *args, **kwargs):
#         self.history.append((self.flux.value, self.x_0.value, self.y_0.value))
#         return super().__call__(*args, **kwargs)


# def make_tracing_anisocado_model(oversampling=2, degree=5, seed=0):
#     img = AnalyticalScaoPsf(pixelSize=0.004/oversampling, N=80*oversampling+1, seed=seed).psf_on_axis
#     return TracingAnisocadoModel(img, oversampling=oversampling, degree=degree)


def doit(input_model: Fittable2DModel,
         pixelphase=0.,
         n_sources1d=2,
         img_size=128,
         img_border=16,
         σ=0,
         λ=100_000,
         fitshape=(20, 20),
         model_oversampling=2,
         model_degree=5,
         model_mode='same',
         fit_accuracy=1e-8,
         use_weights=False,
         fitter_name='Simplex',
         return_imgs=False,
         seed=0):

    rng = np.random.default_rng(seed)
    if fitter_name == 'LM':
        iter_name = 'nfev'
        fitter = fitting.LevMarLSQFitter()
    elif fitter_name == 'Simplex':
        iter_name = 'numiter'
        fitter = fitting.SimplexLSQFitter()
    else:
        raise NotImplementedError

    fitshape = np.array(fitshape)
    img, xy_sources = gen_image(input_model, n_sources1d, img_size, img_border, pixelphase, σ, λ, rng)

    y_grid, x_grid = np.indices(img.shape)

    fit_model = make_fit_model(input_model, model_mode, fitshape, model_degree, model_oversampling)
    fit_model = set_call_trace(fit_model)

    imgfig, imgax = plt.subplots()
    plotfig, plotax = plt.subplots()

    imgax.imshow(img-img.min()+1e-8, norm=LogNorm())

    for i, xy_position in enumerate(xy_sources):
        fitted = fit_single_image(fit_model,
                                  img,
                                  x_grid,
                                  y_grid,
                                  xy_position,
                                  fitter,
                                  fitshape,
                                  σ,
                                  λ,
                                  fit_accuracy,
                                  use_weights,
                                  rng)

        history = np.core.records.fromrecords(fitted.history, names=fitted.param_names)
        xname = [name for name in fitted.param_names if name.startswith('x')][0]
        yname = [name for name in fitted.param_names if name.startswith('y')][0]
        fluxname = [name for name in fitted.param_names if 'flux' in name or 'amplitude' in name][0]

        plotax.text(history[0][xname], history[0][yname], f'start', fontsize=9)
        plotax.text(history[-1][xname], history[-1][yname], f'end', fontsize=9)
        plotax.plot(history[xname], history[yname], ':')
        sc = plotax.scatter(history[xname], history[yname], c=history[fluxname], cmap='viridis')

        imgax.plot(fitted.x.value, fitted.y.value, 'ro')

    imgax.plot(xy_sources[:,0],xy_sources[:,1], 'bx')

    #plotfig.colorbar(sc, ax=plotax)

model = make_anisocado_model(2)
#from astropy.modeling.functional_models import Gaussian2D
#model = Gaussian2D()

doit(model,
     pixelphase=0.,
     n_sources1d=2,
     img_size=128,
     img_border=32,
     σ=10,
     λ=200_000,
     fitshape=(13, 13),
     model_oversampling=2,
     model_degree=5,
     model_mode='same',
     fit_accuracy=1e-8,
     use_weights=False,
     fitter_name='LM',
     return_imgs=False,
     seed=0)
plt.show()