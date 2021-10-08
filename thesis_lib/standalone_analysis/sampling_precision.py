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
from itertools import product, count
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anisocado import AnalyticalScaoPsf
from astropy.modeling import Fittable2DModel
from astropy.modeling import fitting
from astropy.modeling.functional_models import Const2D
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize

from photutils import FittableImageModel


def gen_image(model, N, size, border, pixelphase, σ=0, λ=100_000, rng=np.random.default_rng()):

    # TODO Performance could be improved by model.render, but that needs bounding box
    N1d = int(np.ceil(np.sqrt(N)))
    yx_sources = np.mgrid[0+border:size-border:N1d*1j, 0+border:size-border:N1d*1j].transpose((1, 2, 0)).reshape(-1, 2)
    # swap x and y and round to nearest integer; only take N entries
    xy_sources = np.round(np.roll(yx_sources, 1, axis=0))[:N]

    # constant pixelphase for both x and y
    if isinstance(pixelphase, numbers.Number):
        xy_sources += pixelphase
    # either constant separate x, y pixelphase or per-entry phase
    elif isinstance(pixelphase, np.ndarray):
        assert pixelphase.shape == (2,) or pixelphase.shape == xy_sources.shape
        xy_sources += pixelphase
    elif pixelphase == 'random':
        xy_sources += rng.uniform(0, 1, xy_sources.shape)

    y, x = np.indices((size, size))
    data = np.zeros((size, size), dtype=np.float64)

    for xshift, yshift in xy_sources:
        data += model(x-xshift, y-yshift)

    # normalization: Each model should contribute flux=1 (maybe bad?)
    # data /= np.sum(data)*(N1d**2)
    data /= np.max(data)

    # apply noise
    if λ:
        data = rng.poisson(λ*data).astype(np.float64)
    data += rng.normal(0, σ, size=data.shape)

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


def make_fit_model(input_model, model_mode, fitshape, model_degree, model_oversampling):
    if model_mode.startswith('grid'):
        y_init, x_init = np.mgrid[-fitshape[0]-2:fitshape[0]+2.001:1/model_oversampling,
                                  -fitshape[1]-2:fitshape[1]+2.001:1/model_oversampling]
        # assert np.sum(y_init) == 0.  # arrays should be centered on zero
        # assert np.sum(x_init) == 0.

        gridded_model = FittableImageModel(input_model(x_init, y_init),
                                           oversampling=model_oversampling,
                                           degree=model_degree)
        if '+const' in model_mode:
            fit_model = gridded_model + Const2D(0)
        else:
            fit_model = gridded_model + DummyAdd2D()
    elif model_mode == 'same':
        fit_model = input_model.copy() + DummyAdd2D()

    elif model_mode == 'EPSF':
        # TODO it makes more sense to externalize the EPSF creation and just allow using a different fit and input model
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

    # unify names to alias so we can refer to parameters directly
    fit_model.x = getattr(fit_model.left, xname)
    fit_model.y = getattr(fit_model.left, yname)
    fit_model.flux = getattr(fit_model.left, fluxname)

    return fit_model


def fit_single_image(fit_model: Fittable2DModel,
                     img: np.ndarray,
                     x_grid: np.ndarray,
                     y_grid: np.ndarray,
                     xy_position: np.ndarray,
                     fitter,
                     fitshape=(7, 7),
                     σ=0,
                     λ=None,
                     fit_accuracy=1e-8,
                     use_weights=False,
                     fit_bounds=None,
                     rng=np.random.default_rng()):

    cutout_slices = get_cutout_slices(xy_position, fitshape)

    img_variance = (img - img.min()) + 1 + σ ** 2
    if use_weights:
        weights = 1 / np.sqrt(img_variance)
    else:
        weights = np.ones_like(x_grid)

    # initialize model parameters to sensible values +jitter and add +- 1.1 pixel bounds

    fit_model.x.value = xy_position[0] + rng.uniform(-0.2, 0.2)
    fit_model.y.value = xy_position[1] + rng.uniform(-0.2, 0.2)

    # TODO bound on the parameters blows up the fit.
    if fit_bounds:
        fit_model.x.bounds = (xy_position[0]-fit_bounds[0], xy_position[0]+fit_bounds[0])
        fit_model.y.bounds = (xy_position[1]-fit_bounds[1], xy_position[1]+fit_bounds[1])

    flux = λ if λ else 1
    fit_model.flux.value = flux + rng.uniform(-np.sqrt(flux), +np.sqrt(flux))

    fitted = fitter(fit_model, x_grid[cutout_slices], y_grid[cutout_slices], img[cutout_slices],
                    acc=fit_accuracy, maxiter=1000, weights=weights[cutout_slices])
    fitted.snr = np.sum(img[cutout_slices])/np.sqrt(np.sum(img_variance[cutout_slices]))

    return fitted


def fit_models(input_model: Fittable2DModel,
               pixelphase=0.,
               n_sources=4,
               img_size=128,
               img_border=16,
               σ=0,
               λ=None,
               fitshape=(7, 7),
               model_oversampling=2,
               model_degree=5,
               model_mode='grid',
               fit_accuracy=1e-8,
               use_weights=True,
               fitter_name='LM',
               return_imgs=False,
               fit_bounds=None,
               seed=0) -> dict:

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
    img, xy_sources = gen_image(input_model, n_sources, img_size, img_border, pixelphase, σ, λ, rng)

    fit_model = make_fit_model(input_model, model_mode, fitshape, model_degree, model_oversampling)

    y_grid, x_grid = np.indices(img.shape)

    res = np.full((len(xy_sources), 7), np.nan)
    residual = img.copy()

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
                                  fit_bounds,
                                  rng)

        res[i] = (xy_position[0], xy_position[1],
                  fitted.x.value, fitted.y.value,
                  fitted.flux.value, fitted.snr,
                  fitter.fit_info[iter_name])
        residual -= fitted(x_grid, y_grid)

    x_dev = res[:, 0] - res[:, 2]
    y_dev = res[:, 1] - res[:, 3]
    dev = np.sqrt(x_dev**2 + y_dev**2)
    ret = {'x':    res[:, 0], 'y': res[:, 1],
           'dev':  dev, 'x_dev': x_dev, 'y_dev': y_dev,
           'flux': res[:, 4], 'snr': res[:, 5],
           'fititers': res[:, 6]}
    if return_imgs:
        ret |= {'img': img, 'residual': residual}

    return ret


def fit_models_dictarg(kwargs_dict):
    return kwargs_dict | fit_models(**kwargs_dict)


def dictoflists_to_listofdicts(dict_of_list):
    return [dict(zip(dict_of_list.keys(), vals)) for vals in product(*dict_of_list.values())]


def create_arg_list(dict_of_lists):
    dict_of_lists = dict_of_lists.copy()
    seed_start = dict_of_lists.pop('seed_start', 0)
    list_of_dict = dictoflists_to_listofdicts(dict_of_lists)
    for i, entry in zip(count(seed_start), list_of_dict):
        entry['seed'] = i
    return list_of_dict


def transform_dataframe(results: pd.DataFrame):
    # we get a row for each call to fit_model. Each row contains possibly the results for multiple sources
    # we want a row for each individual measurement, duplicating the parameters
    multicolumns = pd.Index({'x', 'y', 'dev', 'x_dev', 'y_dev', 'flux', 'snr', 'fititers'})
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
        return super().__repr__() + f' oversampling: {self.oversampling}'

    def __str__(self):
        return super().__str__() + f' oversampling: {self.oversampling}'


def make_anisocado_model(oversampling=2, degree=5, seed=0, offaxis=(0, 0)):
    img = AnalyticalScaoPsf(pixelSize=0.004/oversampling, N=80*oversampling+1, seed=seed).shift_off_axis(*offaxis)
    return AnisocadoModel(img, oversampling=oversampling, degree=degree)


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
    grouped = results.groupby([results.input_model.astype('str'),
                               'use_weights',
                               'n_sources'],
                              as_index=False)

    #fig, axes = plt.subplots(len(grouped), sharex='all')
    plt.figure()
    for i, ((model, weight, n_sources), modelgroup) in enumerate(grouped):
        xs = [i[0] for i in modelgroup.groupby('fitshape').mean().index]
        ys = modelgroup.groupby('fitshape').mean().dev
        errs = modelgroup.groupby('fitshape').std().dev
        style = '-' if weight==True else '--'
        plt.plot(xs, ys, style,
                 label=f'{type(modelgroup.input_model.iloc[0]).__name__} weights: {weight}, N: {n_sources}')
        plt.fill_between(xs, ys - errs, ys + errs, alpha=0.3)

    plt.xlabel('Cutout size of fit around star')
    plt.ylabel('mean position deviation')
    plt.legend()


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
                     fmt='o', label=f'{repr(group.input_model.iloc[0])} σ={σ}')

    plt.legend(fontsize='small')
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Peak pixel count')

    plt.ylabel('Mean Deviation in measured Position')
    #plt.title("everything's a line on a loglog-plot lol")


def plot_noise_vs_weights(results: pd.DataFrame):
    grouped = results.groupby(['σ',
                               results.input_model.apply(lambda model: repr(model)),
                               results.use_weights],
                              as_index=False)

    fig = plt.figure()

    for i, ((σ, model, weights), group) in enumerate(grouped):
        # for each noisetype, plot magnitude vs deviation
        noisegroup = group.groupby('λ')
        model = noisegroup.input_model.first().iloc[0].__class__.__name__
        # plt.errorbar(noisegroup.first().index, noisegroup.mean().dev, noisegroup.std().dev,
        #             alpha=0.8, errorevery=(i,len(grouped)), fmt='o', label=f'input_model: {model}, modeldegree: {degree}, weights: {weights}')
        xs = noisegroup.first().index
        ys = noisegroup.dev.mean()
        errs = noisegroup.dev.std()
        plt.plot(xs, ys, '-', alpha=0.9, label=f'input_model: {model}, σ: {σ}, weights: {weights}')
        plt.fill_between(xs, ys - errs, ys + errs, alpha=0.3)

    # noise_σ = results.noise.apply(lambda noise: noise.std).sort_values()
    ## precision = α*fwhm/snr
    # fwhm = 2*sqrt(2*log(2)) * 5
    # peak_snr = 1./noise_σ
    # precission = fwhm/peak_snr
    # plt.plot(noise_σ, precission, ':', label='α=1 Peak SNR Prediction')
    # plt.plot(noise_σ, fwhm/results.snr, ':', label='SNR sum prediction')

    plt.legend(fontsize='small')
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('λ')
    plt.ylabel('Deviation in measured Position')
    plt.title("")
    plt.tight_layout()


if __name__ == '__main__':
    from tqdm.cli import tqdm

    #gauss = Gaussian2D(x_stddev=5., y_stddev=5.)
    #airy = AiryDisk2D(radius=5.)
    anisocado = make_anisocado_model(2,5)

    size_1D = 2j
    xy_pairs = np.mgrid[0:0.5:size_1D, 0:0.5:size_1D].transpose((1, 2, 0)).reshape(-1, 2)

    dl = {'input_model': [anisocado],
          'n_sources': [4],
          'img_border': [30],
          'img_size': [30*6],
          'pixelphase': xy_pairs,
          'fitshape': [(31, 31)],
          'σ': [0, 50],
          'λ': np.linspace(30_000, 200_000, 3),
          'model_oversampling': [2],
          'model_degree': [3],
          'model_mode': ['grid'],
          'fit_accuracy': [1.49012e-08],
          'use_weights': [False, True],
          'return_imgs': [True]
          }

    from thesis_lib.util import DebugPool
    with DebugPool() as p:
    #with mp.Pool() as p:
        results = pd.DataFrame.from_records(
            p.map(fit_models_dictarg, tqdm(create_arg_list(dl))))
    results = transform_dataframe(results)
    print(results.dev.describe())
    #results = results[['noise', 'residual', 'use_weights', 'pixelphase']]
    #results.noise = results.noise.transform(lambda n: (n.gauss_std, n.poisson_std))

    #plot_fitshape(results[results.n_sources1d == 1])
    plot_fitshape(results)
    plot_xy_deviation(results)
    plot_phase_vs_deviation(results)
    plot_phase_vs_deviation3d(results)
    plot_noise_vs_weights(results)
    plt.show()
