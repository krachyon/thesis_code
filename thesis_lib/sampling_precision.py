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

from astropy.modeling import Fittable2DModel
from photutils import FittableImageModel, BasicPSFPhotometry, MADStdBackgroundRMS, DAOGroup
from astropy.modeling.functional_models import Gaussian2D, AiryDisk2D, Moffat2D, Const2D
import numpy as np
import numbers
from astropy.table import Table
from typing import Type, Union, List
import matplotlib.pyplot as plt
from astropy.modeling import fitting
import multiprocess as mp
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize
from collections import namedtuple
import os
from anisocado import AnalyticalScaoPsf


Result = namedtuple('Result',
                    ['img', 'n_sources1d',
                     'tab', 'pixelphase', 'fitshape', 'noise',
                     'model_oversampling', 'model_degree', 'model_mode', 'model_name',
                     'fit_accuracy'])


def gen_image(model, N1d, size, border, pixelphase, noise=None):
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
               fit_accuracy=1e-8) -> Result:

    # it's pretty hard to get a different random state in each process...
    np.random.seed(int.from_bytes(os.urandom(4), 'little'))

    fitshape = np.array(fitshape)
    img, xy_sources = gen_image(input_model, n_sources1d, img_size, img_border, pixelphase, noise)


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

    res = Table(names='x, y, x_fit, y_fit, flux_fit'.split(', '))
    for xy_position in xy_sources:
        cutout_slices = get_cutout_slices(xy_position, fitshape)

        # initialize model parameters to sensible values +jitter and add +- 1.1 pixel bounds
        # TODO fixing the parameters blows up the fit.
        setattr(fit_model.left, xname, xy_position[0] + np.random.uniform(-0.1, 0.1))
        #getattr(fit_model.left, xname).bounds = (xy_position[0]-1.1, xy_position[0]+1.1)
        setattr(fit_model.left, yname, xy_position[1] + np.random.uniform(-0.1, 0.1))
        #getattr(fit_model.left, yname).bounds = (xy_position[1]-1.1, xy_position[1]+1.1)

        fitted = fitter(fit_model, x[cutout_slices], y[cutout_slices], img[cutout_slices], acc=fit_accuracy, maxiter=100_000)
        res.add_row((xy_position[0],
                     xy_position[1],
                     getattr(fitted.left, xname),
                     getattr(fitted.left, yname),
                     getattr(fitted.left, fluxname)))

    res['x_dev'] = res['x']-res['x_fit']
    res['y_dev'] = res['y']-res['y_fit']
    res['dev']   = np.sqrt(res['x_dev']**2 + res['y_dev']**2)
    return Result(img,
                  n_sources1d,
                  res,
                  pixelphase,
                  fitshape,
                  noise,
                  model_oversampling,
                  model_degree,
                  model_mode,
                  type(input_model).__name__,
                  fit_accuracy)


def fit_models_dictarg(kwargs_dict):
    return fit_models(**kwargs_dict)


class AnisocadoModel(FittableImageModel):
    pass


# TODO oversampling=1 does not work...
# TODO also with oversampling=8 the fit diverges...
def make_anisocado_model(oversampling=8):
    img = AnalyticalScaoPsf(pixelSize=0.004/oversampling, N=80*oversampling+1).psf_on_axis
    return AnisocadoModel(img, oversampling=oversampling, degree=3)


class Repr:
    def __repr__(self):
        items = [item for item in self.__dict__.items() if not item[0].startswith('__')]
        item_string = ', '.join([f'{item[0]} = {item[1]}' for item in items])
        return f'{type(self)}({item_string})'


class GaussNoise(Repr):
    def __init__(self, center, std):
        self.center = center
        self.std = std

    def __call__(self, data):
        return data + np.random.normal(self.center, self.std, data.shape)


class PoissonNoise(Repr):
    def __init__(self, max_count):
        self.max_count = max_count

    def __call__(self, data):
        return np.random.poisson(self.max_count*data)/self.max_count


class CombinedNoise(Repr):
    def __init__(self, center, std, max_count):
        self.gauss = GaussNoise(center, std)
        self.poisson = PoissonNoise(max_count)

    def __call__(self, data):
        return self.gauss(self.poisson(data))


def plot_xy_deviation(results: List[Result], model):
    cmap = get_cmap('turbo')
    plt.figure()
    plt.title(repr(model))
    for res in results:
        phase = np.sqrt(res.pixelphase[0]**2+res.pixelphase[1]**2)
        plt.plot(res.tab['x_dev'], res.tab['y_dev'], 'o', markeredgewidth=0, color=cmap(phase/np.sqrt(2)), alpha=0.5)
        plt.xlabel('x deviation')
        plt.ylabel('y deviation')
    plt.gca().axis('equal')
    plt.colorbar(ScalarMappable(norm=Normalize(0, np.sqrt(2)), cmap=cmap)).set_label('euclidean pixelphase')


def plot_phase_vs_deviation(results: List[Result], model):
    plt.figure()
    plt.title(repr(model))


    phases = [np.sqrt(res.pixelphase[0]**2+res.pixelphase[1]**2) for res in results]
    sigmas = [np.mean(res.tab['dev']) for res in results]
    plt.xlabel('euclidean pixelphase')
    plt.ylabel('euclidean xy-deviation')

    sigmas, phases = np.array(sigmas), np.array(phases)
    order = np.argsort(phases)
    plt.plot(phases[order], sigmas[order])


def plot_phase_vs_deviation3d(results: List[Result], model):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title(repr(model))

    phases_3d = np.array([res.pixelphase for res in results])
    sigmas = [np.mean(res.tab['dev']) for res in results]

    fig.tight_layout()
    ax.plot_trisurf(phases_3d[:, 0], phases_3d[:, 1], sigmas, cmap='gist_earth')
    ax.set_xlabel('x phase')
    ax.set_ylabel('y phase')
    ax.set_zlabel('euclidean  deviation')


def plot_fitshape(results: List[Result]):
    dat = Table(rows=[(np.mean(res.tab['dev']), res.fitshape[0], res.model_name) for res in results],
              names=['dev','fitshape','model_name']).group_by(['fitshape','model_name'])

    mean = dat.groups.aggregate(np.mean).group_by('model_name').groups
    std = dat.groups.aggregate(np.std).group_by('model_name').groups


    fig, axes = plt.subplots(len(mean), sharex=True)
    for i in range(len(mean)):
        axes[i].errorbar(mean[i]['fitshape'], mean[i]['dev'], std[i]['dev'], fmt='o', label=mean.keys[i]['model_name'])
        axes[i].legend()


if __name__ == '__main__':

    model = Gaussian2D(x_stddev=2., y_stddev=2.)  # ,theta=2*np.pi/8)
    # model = AiryDisk2D(radius=4.5)
    # model = Moffat2D(gamma=2.5, alpha=2.5)
    # model = make_anisocado_model()

    size_1D = 5j
    xy_pairs = np.mgrid[0:1:size_1D, 0:1:size_1D].transpose((1, 2, 0)).reshape(-1, 2)

    from thesis_lib.util import DebugPool
    with DebugPool() as p:
    #with mp.Pool(10) as p:
        results = p.map(lambda phase: fit_models(input_model=model,
                                                 pixelphase=phase,
                                                 fitshape=(31, 31),
                                                 model_oversampling=2,
                                                 model_degree=5,
                                                 model_mode='same',
                                                 fit_accuracy=1e-10,
                                                 noise=lambda data: CombinedNoise(0, 1e-10, 20)(data)),
                        xy_pairs)

    plot_xy_deviation(results, model)
    plot_phase_vs_deviation(results, model)
    plot_phase_vs_deviation3d(results, model)
    plt.show()
