# TODO Reachable Precision vs
#  - Pixelphase
#      * Do we get a "perfect" model from a uniform pixelphase?
#  - Used Model, FWHM
#  - Noise
#  - Model type:
#       - Building model with EPSF builder
#       - using sampled model
#       - analytical model as sanity check: How good can optimizer be?
#  - Interpolation of gridded model
# TODO?
#  - Fitting with weights; Is that done in photutils-photometry?


# Funny things to note:
    # Weird grid progression without any noise and gaussian (σ~1px). 4 Humps with slightly different height
    # Wandering paths when error is same for all generated images
    # Airy2D radius 8: Break in deviation at 0.5 phases

import numbers
import os
from itertools import product
from typing import Union, Optional

import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
import pandas as pd
from anisocado import AnalyticalScaoPsf
from astropy.modeling import Fittable2DModel
from astropy.modeling import fitting
from astropy.modeling.functional_models import Gaussian2D, AiryDisk2D, Const2D
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize

from photutils import FittableImageModel


class Repr:
    def __repr__(self):
        items = [item for item in self.__dict__.items() if not item[0].startswith('__')]
        item_string = ', '.join([f'{item[0]} = {item[1]}' for item in items])
        return f'{type(self)}({item_string})'


class Noise(Repr):
    def __call__(self, data):
        raise NotImplementedError


class CombinedNoise(Noise):
    name = 'CombinedNoise'

    def __init__(self, center, std, max_count):
        self.gauss_center = center
        self.gauss_std = std
        self.max_count = max_count
        if np.isinf(max_count):
            self.poisson_std = 0
        else:
            self.poisson_std = np.sqrt(max_count)/max_count  # scale back to one

        self.std = np.sqrt(self.gauss_std**2 + self.poisson_std**2)

    def __call__(self, data):
        # switch to disable poisson noise
        if np.isinf(self.max_count):
            poissoned = data
        else:
            poissoned = np.random.poisson(self.max_count * data) / self.max_count

        gaussed = poissoned + (np.random.normal(self.gauss_center, self.gauss_std, data.shape)
                               if (self.gauss_std != 0 or self.gauss_center != 0) else 0.)
        return gaussed


class GaussNoise(CombinedNoise):
    name = 'GaussNoise'

    def __init__(self, center, std):
        super().__init__(center, std, np.inf)


class PoissonNoise(CombinedNoise):
    name = 'PoissonNoise'

    def __init__(self, λ):
        super().__init__(0, 0, λ)


def gen_image(model, N1d, size, border, pixelphase, noise: Optional[Noise] = None):
    # TODO could be improved by model.render
    yx_sources = np.mgrid[0+border:size-border:N1d*1j, 0+border:size-border:N1d*1j].transpose((1, 2, 0)).reshape(-1, 2)
    # swap x and y and round to nearest integer
    xy_sources = np.round(np.roll(yx_sources, 1, axis=0))

    # constant pixelphase for both x and y
    if isinstance(pixelphase, numbers.Number):
        xy_sources += pixelphase
    # either constant separate x, y pixelphase or per-entry phase
    elif isinstance(pixelphase, np.ndarray):
        assert pixelphase.shape == (2,) or pixelphase.shape == xy_sources.shape
        xy_sources += pixelphase
    elif pixelphase == 'random':
        xy_sources += np.random.uniform(0, 1, xy_sources.shape)

    y, x = np.indices((size, size))
    data = np.zeros((size, size), dtype=np.float64)

    for xshift, yshift in xy_sources:
        data += model(x-xshift, y-yshift)

    # normalization: Each model should contribute flux=1 (maybe bad?)
    # data /= np.sum(data)*(N1d**2)
    data /= np.max(data)

    if noise:
        data = noise(data)
    return data, xy_sources


def get_cutout_slices(position: np.ndarray, cutout_size: Union[int, np.ndarray]):

    # clip values less than 0 to 0, greater does not matter, that's ignored
    low = np.clip(np.round(position - cutout_size/2+0.5).astype(int), 0, None)
    high = np.clip(np.round(position + cutout_size/2+0.5).astype(int), 0, None)

    # y will be outer dimension of imag
    return slice(low[1], high[1]), slice(low[0], high[0])


class DummyAdd2D(Fittable2DModel):
    @staticmethod
    def evaluate(x, y):
        return np.zeros_like(x)


def fit_models(input_model: Fittable2DModel,
               pixelphase=0.,
               n_sources1d=4,
               img_size=128,
               img_border=16,
               noise=None,
               fitshape=(7, 7),
               model_oversampling=2,
               model_degree=5,
               model_mode='grid',
               fit_accuracy=1e-8,
               use_weights=False,
               return_img=False) -> dict:

    # it's pretty hard to get a different random state in each process...
    np.random.seed(int.from_bytes(os.urandom(4), 'little'))

    fitshape = np.array(fitshape)
    img, xy_sources = gen_image(input_model, n_sources1d, img_size, img_border, pixelphase, noise)
    if noise is None:
        noise = CombinedNoise(0, 0, np.inf)

    if model_mode.startswith('grid'):
        y_init, x_init = np.mgrid[-fitshape[0]-2:fitshape[0]+2.001:1/model_oversampling,
                                  -fitshape[1]-2:fitshape[1]+2.001:1/model_oversampling]
        assert np.sum(y_init) == 0.  # arrays should be centered on zero
        assert np.sum(x_init) == 0.


        gridded_model = FittableImageModel(input_model(x_init, y_init),
                                           oversampling=model_oversampling,
                                           degree=model_degree)
        if '+const' in model_mode:
            fit_model = gridded_model + Const2D(0)
        else:
            fit_model = gridded_model + DummyAdd2D()

    if model_mode == 'same':
        fit_model = input_model.copy() + DummyAdd2D()

    if model_mode == 'EPSF':
        raise NotImplementedError

    # todo So this is a lot more annoying than originally though as it will change the parameter names
    #  if we pass in a Gaussian2D we'd have to change x_0 to x_mean and flux to amplitude and
    #  mess around with getattr(model, parname)/setattr. Already the .left is bugging me
    xname = [name for name in ['x_0', 'x_mean'] if hasattr(fit_model.left, name)][0]  # should always be one, yolo
    yname = [name for name in ['y_0', 'y_mean'] if hasattr(fit_model.left, name)][0]
    fluxname = [name for name in ['flux', 'amplitude'] if hasattr(fit_model.left, name)][0]
    # nail down everything extra that could be changed by the fit, we only want to optimize position and flux
    for name in fit_model.left.param_names:
        if name not in {xname, yname, fluxname}:
            fit_model.left.fixed[name] = True



    fitter = fitting.LevMarLSQFitter()
    y, x = np.indices(img.shape)

    img_pos = img - np.min(img)
    noise_img = np.sqrt(img_pos * noise.poisson_std**2 + noise.gauss_std ** 2)
    if use_weights:
        weights = noise_img
        # scaling doesn't matter much, but probably should be around one on average for each pixel
        weights /= np.sum(weights)/weights.size
    else:
        weights = np.ones_like(x)

    res = np.full((len(xy_sources), 6), np.nan)
    for i, xy_position in enumerate(xy_sources):
        cutout_slices = get_cutout_slices(xy_position, fitshape)

        # initialize model parameters to sensible values +jitter and add +- 1.1 pixel bounds
        # TODO bound on the parameters blows up the fit.
        setattr(fit_model.left, xname, xy_position[0] + np.random.uniform(-0.1, 0.1))
        #getattr(fit_model.left, xname).bounds = (xy_position[0]-1.1, xy_position[0]+1.1)
        setattr(fit_model.left, yname, xy_position[1] + np.random.uniform(-0.1, 0.1))
        #getattr(fit_model.left, yname).bounds = (xy_position[1]-1.1, xy_position[1]+1.1)

        snr = np.sum(img[cutout_slices])/np.sqrt(np.sum(noise_img[cutout_slices]**2))

        fitted = fitter(fit_model, x[cutout_slices], y[cutout_slices], img[cutout_slices],
                        acc=fit_accuracy, maxiter=100_000, weights=weights[cutout_slices])
        res[i] = (xy_position[0], xy_position[1],
                  getattr(fitted.left, xname).value, getattr(fitted.left, yname).value,
                  getattr(fitted.left, fluxname).value,
                  snr)
        residual = fitted(x[cutout_slices], y[cutout_slices]) - img[cutout_slices]

    x_dev = res[:, 0] - res[:, 2]
    y_dev = res[:, 1] - res[:, 3]
    dev = np.sqrt(x_dev**2 + y_dev**2)
    ret = {'x':    res[:, 0], 'y': res[:, 1],
           'dev':  dev, 'x_dev': x_dev, 'y_dev': y_dev,
           'flux': res[:, 4], 'snr': res[:,5], 'residual': residual}
    if return_img:
        ret |= {'img': img}
    return ret


def fit_models_dictarg(kwargs_dict):
    return kwargs_dict | fit_models(**kwargs_dict)


def dictoflists_to_listofdicts(dict_of_list):
    return (dict(zip(dict_of_list.keys(), vals)) for vals in product(*dict_of_list.values()))


def transform_dataframe(results: pd.DataFrame):
    # we get a row for each call to fit_model. Each row contains possibly the results for multiple sources
    # we want a row for each individual measurement, duplicating the parameters
    multicolumns = pd.Index({'x', 'y', 'dev', 'x_dev', 'y_dev', 'flux', 'snr'})
    singlecolumns = results.columns.difference(multicolumns)
    # turn each entry that is not supposed to be an array/list already into single element lists
    results[singlecolumns] = results[singlecolumns].apply(lambda col: [[i] for i in col])
    # magic to turn list in row -> multiple rows for each entry. single element entries are broadcast
    results = results.apply(pd.Series.explode)
    results['uuid'] = results.index  # tag all rows belonging to a single function call
    results[multicolumns] = results[multicolumns].astype(np.float64)
    results = results.apply(lambda col: pd.to_numeric(col, errors='ignore'))  # restore lost numeric types if possible
    return results


class AnisocadoModel(FittableImageModel):
    def __repr__(self):
        return super().__repr__() + f'oversampling: {str(self.oversampling)}'

    def __str__(self):
        return super().__str__() + f'oversampling: {str(self.oversampling)}'


# TODO oversampling=1 does not work...
# TODO also with oversampling=8 the fit diverges...
def make_anisocado_model(oversampling=8):
    img = AnalyticalScaoPsf(pixelSize=0.004/oversampling, N=80*oversampling+1).psf_on_axis
    return AnisocadoModel(img, oversampling=oversampling, degree=3)


def plot_xy_deviation(results: pd.DataFrame):
    cmap = get_cmap('turbo')
    plt.figure()
    plt.title(''.join(results['input_model'].apply(repr).unique()))

    phases = np.sqrt(np.sum(np.stack(results.pixelphase)**2, axis=1))

    plt.scatter(results.x_dev, results.y_dev, marker='o', c=(phases/np.sqrt(2)), alpha=0.5, edgecolors='None', cmap=cmap)
    plt.xlabel('x deviation')
    plt.ylabel('y deviation')
    plt.gca().axis('equal')
    plt.colorbar(ScalarMappable(norm=Normalize(0, np.sqrt(2)), cmap=cmap)).set_label('euclidean pixelphase')


def plot_phase_vs_deviation(results: pd.DataFrame):
    plt.figure()
    plt.title(''.join(results['input_model'].apply(repr).unique()))

    grouped = results.groupby('uuid')

    phases = np.stack(grouped.pixelphase.first())
    phases = np.sum(phases**2, axis=1)
    sigmas = grouped.dev.mean()
    plt.xlabel('euclidean pixelphase')
    plt.ylabel('euclidean xy-deviation')

    sigmas, phases = np.array(sigmas), np.array(phases)
    order = np.argsort(phases)
    plt.plot(phases[order], sigmas[order])


def plot_phase_vs_deviation3d(results: pd.DataFrame):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title(''.join(results['input_model'].apply(repr).unique()))

    grouped = results.groupby('uuid')

    phases_3d = np.stack(grouped.pixelphase.first())
    sigmas = grouped.dev.mean()

    fig.tight_layout()
    ax.plot_trisurf(phases_3d[:, 0], phases_3d[:, 1], sigmas, cmap='gist_earth')
    ax.set_xlabel('x phase')
    ax.set_ylabel('y phase')
    ax.set_zlabel('euclidean  deviation')


def plot_fitshape(results: pd.DataFrame):
    grouped = results.groupby(results.input_model.astype('str'), as_index=False)

    #fig, axes = plt.subplots(len(grouped), sharex='all')
    plt.figure()
    for model, modelgroup in grouped:
        mean = modelgroup.groupby('fitshape').mean()
        std = modelgroup.groupby('fitshape').std()
        plt.errorbar([i[0] for i in mean.index], mean.dev, std.dev,
                     fmt='o', label=repr(modelgroup.input_model.iloc[0]))
    plt.legend()


def plot_deviation_vs_noise(results: pd.DataFrame):

    # issue: pickling destroys type identity, so we need to convert to strings
    # select all distinct models and distinct noise /types/
    grouped = results.groupby([results.input_model.astype(str),
                               results.noise.apply(lambda noise: noise.name)],
                              as_index=False)

    plt.figure()
    for (model_str, noise_name), group in grouped:
        # for each noisetype, plot magnitude vs deviation
        noisegroup = group.groupby(group.noise.apply(lambda noise: noise.std))
        plt.errorbar(noisegroup.first().index, noisegroup.mean().dev, noisegroup.std().dev,
                     fmt='o', label=repr(group.input_model.iloc[0])+' '+noise_name)

    plt.legend(fontsize='small')
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Noise σ')

    plt.ylabel('Deviation in measured Position')
    plt.title("everything's a line on a loglog-plot lol")


def plot_noise_vs_weights(results: pd.DataFrame):
    grouped = results.groupby([results.noise.apply(lambda n: n.name),
                               results.input_model.apply(lambda model: repr(model)),
                               results.use_weights],
                              as_index=False)

    fig = plt.figure()

    for i, ((degree, model, weights), group) in enumerate(grouped):
        # for each noisetype, plot magnitude vs deviation
        noisegroup = group.groupby(group.noise.apply(lambda noise: noise.std))
        model = noisegroup.input_model.first().iloc[0].__class__.__name__
        # plt.errorbar(noisegroup.first().index, noisegroup.mean().dev, noisegroup.std().dev,
        #             alpha=0.8, errorevery=(i,len(grouped)), fmt='o', label=f'input_model: {model}, modeldegree: {degree}, weights: {weights}')
        xs = noisegroup.first().index
        ys = noisegroup.dev.mean()
        errs = noisegroup.dev.std()/xs
        plt.plot(xs, ys/xs, '-', alpha=0.9, label=f'input_model: {model}, modeldegree: {degree}, weights: {weights}')
        plt.fill_between(xs, ys/xs - errs, ys/xs + errs, alpha=0.3) # TODO changeme back

    # noise_σ = results.noise.apply(lambda noise: noise.std).sort_values()
    ## precision = α*fwhm/snr
    # fwhm = 2*sqrt(2*log(2)) * 5
    # peak_snr = 1./noise_σ
    # precission = fwhm/peak_snr
    # plt.plot(noise_σ, precission, ':', label='α=1 Peak SNR Prediction')
    # plt.plot(noise_σ, fwhm/results.snr, ':', label='SNR sum prediction')

    plt.legend(fontsize='small')
    #plt.xscale('log')
    #plt.yscale('log')

    plt.xlabel('Noise σ')
    plt.ylabel('Deviation in measured Position/σ')
    plt.title("everything's a line on a loglog-plot lol")
    plt.tight_layout()


if __name__ == '__main__':
    from tqdm.cli import tqdm

    gauss = Gaussian2D(x_stddev=5., y_stddev=5.)

    size_1D = 3j
    xy_pairs = np.mgrid[0:0.5:size_1D, 0:0.5:size_1D].transpose((1, 2, 0)).reshape(-1, 2)

    noise = [PoissonNoise(λ) for λ in 1/np.linspace(1/np.sqrt(50), 1/np.sqrt(10000), 50)**2] + \
            [GaussNoise(0, σ) for σ in np.linspace(5e-3, 1.5e-1, 50)]

    dl = {'input_model': [gauss],
          'n_sources1d': [4],
          'img_border': [30],
          'img_size': [30*6],
          'pixelphase': xy_pairs,
          'fitshape': [(31, 31)],
          'noise': noise,
          'model_oversampling': [2],
          'model_degree': [3],
          'model_mode': ['same'],
          'fit_accuracy': [1.49012e-08],
          'use_weights': [False, True],
          'return_img': [True]
          }

    #from thesis_lib.util import DebugPool
    #with DebugPool() as p:
    with mp.Pool() as p:
        results = pd.DataFrame.from_records(
            p.imap_unordered(fit_models_dictarg, tqdm(list(dictoflists_to_listofdicts(dl)))))
    results = transform_dataframe(results)

    #results = results[['noise', 'residual', 'use_weights', 'pixelphase']]
    #results.noise = results.noise.transform(lambda n: (n.gauss_std, n.poisson_std))

    #plot_fitshape(results[results.n_sources1d == 1])
    #plot_fitshape(results)
    #plot_xy_deviation(results)
    #plot_phase_vs_deviation(results)
    #plot_phase_vs_deviation3d(results)
    plot_noise_vs_weights(results)
    plt.show()
