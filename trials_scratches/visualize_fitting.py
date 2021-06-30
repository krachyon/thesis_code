import numpy as np

from thesis_lib.sampling_precision import *
from matplotlib.colors import LogNorm
from scipy.signal import fftconvolve

from itertools import tee


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class TracingAnisocadoModel(AnisocadoModel):

    def __init__(self, *args, **kwargs):
        self.history = []
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        self.history.append((self.flux.value, self.x_0.value, self.y_0.value))
        return super().__call__(*args, **kwargs)


def make_tracing_anisocado_model(oversampling=2, degree=5, seed=0):
    img = AnalyticalScaoPsf(pixelSize=0.004/oversampling, N=80*oversampling+1, seed=seed).psf_on_axis
    return TracingAnisocadoModel(img, oversampling=oversampling, degree=degree)



model = make_anisocado_model(2)

λ=1000

img, xy_sources = gen_image(model, N1d=2, border=64, size=168, pixelphase=[0.5, 0.5], σ=10.0, λ=λ)

y, x = np.indices(img.shape)


fitters = [fitting.LevMarLSQFitter(), fitting.SimplexLSQFitter()]

to_fit = make_tracing_anisocado_model(2)
to_fit.x_0 = xy_sources[3][0] + np.random.uniform(-0.1, 0.1)
to_fit.y_0 = xy_sources[3][1] + np.random.uniform(-0.1, 0.1)
to_fit.flux = λ

imgfig, imgax = plt.subplots()
plotfig, plotax = plt.subplots()

imgax.imshow(img, norm=LogNorm())

for fitter in fitters:
    # TODO fitshape
    fitted = fitter(to_fit.copy(), x, y, img)
    history = np.array(fitted.history)

    plotax.text(history[0,1],history[0,2], f'start{fitter.__class__}')
    plotax.text(history[-1,1], history[-1,2], f'end{fitter.__class__}')
    plotax.plot(history[:,1], history[:,2], ':')
    sc = plotax.scatter(history[:, 1], history[:, 2], c=history[:, 0], cmap='viridis')

    imgax.plot(fitted.x_0, fitted.y_0, 'o')

plotfig.colorbar(sc, ax=plotax)
plt.show()