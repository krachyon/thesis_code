from numbers import Integral
from typing import Iterable

import numba
from more_itertools import chunked
import numpy as np
from astropy.table import Table
from photutils import FittableImageModel
from thesis_lib.scopesim_helper import make_anisocado_model
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from thesis_lib.testdata.generators import read_or_generate_image
from thesis_lib.scopesim_helper import download
from thesis_lib.testdata.recipes import scopesim_groups
import itertools
import timeit

#download()

xname, yname, fluxname = ['x_0', 'y_0', 'flux_0']

def make_objective_function(model: FittableImageModel,
                            guess_table: Table,
                            img: np.ndarray,
                            to_optimize: Iterable[Integral] = (0,),
                            fitshape: Integral = 31,
                            posbound: float = 0.2,
                            fluxbound: float = np.inf):

    xbounds, ybounds = get_fitbounds(img, guess_table, fitshape)

    img_to_fit = img[ybounds[0]:ybounds[1], xbounds[0]:xbounds[1]]
    base_img = np.zeros_like(img_to_fit)
    to_optimize = set(to_optimize)

    #x = np.arange(model._nx, dtype=float)
    #y = np.arange(model._ny, dtype=float)
    #interpolator = interp2d(x, y, model._data.T, k=3)
    interpolator = model.interpolator
    oversampling = model.oversampling
    norm = model._normalization_constant  # noqa
    # TODO ogrid?
    xs, ys = np.arange(xbounds[0], xbounds[1]).astype(float), np.arange(ybounds[0], ybounds[1]).astype(float)
    xs += model.origin[0]/oversampling[0]
    ys += model.origin[1]/oversampling[0]

    x0 = []
    lbounds = []
    ubounds = []
    for i, row in enumerate(guess_table):
        if i in to_optimize:
            x0.append(row[xname])
            lbounds.append(row[xname]-posbound)
            ubounds.append(row[xname]+posbound)
            x0.append(row[yname])
            lbounds.append(row[yname]-posbound)
            ubounds.append(row[yname]+posbound)
            x0.append(row[fluxname])
            lbounds.append(row[fluxname]-fluxbound)
            ubounds.append(row[fluxname]+fluxbound)
        else:
            f = row[fluxname]*norm
            base_img += f * interpolator(oversampling[0]*(xs-row[xname]),
                                         oversampling[1]*(ys-row[yname])).T

    def objective(values):
        # assert len(values) = 3 * len(to_optimize)
        eval_img = np.zeros_like(img_to_fit)
        # TODO maybe re-gridding this instead of transposing
        for x, y, flux in np.split(values, len(values)/3):
            eval_img += flux * norm * interpolator(oversampling[0]*(xs-x),
                                                   oversampling[1]*(ys-y)).T
        return (img_to_fit - (base_img + eval_img)).flatten()

    return objective, x0, (lbounds, ubounds)


def get_fitbounds(img: np.ndarray, initial_guess: Table, fitshape: int) -> tuple[np.ndarray, np.ndarray]:

    xvals = np.array(initial_guess[xname]).astype(int)
    yvals = np.array(initial_guess[yname]).astype(int)

    xbound = np.array([np.min(xvals) - fitshape,
                      np.max(xvals) + fitshape])
    ybound = np.array([np.min(yvals) - fitshape,
                      np.max(yvals) + fitshape])

    xbound = np.clip(xbound, 0, img.shape[1]) - 1
    ybound = np.clip(ybound, 0, img.shape[0]) - 1

    return xbound, ybound


def update_guess(guess_table, x_opt, to_optimize: Iterable):

    for idx, (x,y,f) in zip(to_optimize, chunked(x_opt, 3, strict=True)):
        row = guess_table[idx]
        row[xname] = x
        row[yname] = y
        row[fluxname] = f


if __name__ == '__main__':
    model = make_anisocado_model()
    rng = np.random.default_rng(seed=11)

    def recipe():
        return scopesim_groups(N1d=1, border=500, group_size=10, group_radius=12, jitter=12,
                               magnitude=lambda N: [20.5] * N,
                               custom_subpixel_psf=model, seed=10)


    img, table = read_or_generate_image('group_image_10', recipe=recipe)
    guess_table = table.copy()
    guess_table.rename_columns(['x', 'y', 'f'], [xname, yname, fluxname])
    guess_table.sort(fluxname, reverse=True)

    guess_table[xname] += rng.uniform(-0.01, 0.01, len(guess_table))
    guess_table[yname] += rng.uniform(-0.01, 0.01, len(guess_table))
    guess_table[fluxname] *= 1e15

    objective, x0, bounds = make_objective_function(model, guess_table, img, to_optimize=set(), posbound=0.05)
    # t=timeit.Timer(lambda: least_squares(objective, x0, bounds=bounds)).repeat(repeat=5, number=1)
    # print(t)
    for run, i in tqdm(list(itertools.product(range(3), range(9)))):
        to_optimize = {i, i+1}
        objective, x0, bounds = make_objective_function(model, guess_table, img, to_optimize=to_optimize, fitshape=51, posbound=0.3)
        x_found = least_squares(objective, x0, bounds=bounds)
        update_guess(guess_table, x_found['x'], to_optimize)

    for run, i in tqdm(list(itertools.product(range(2), range(8)))):
        to_optimize = {i, i+1, i+2}
        objective, x0, bounds = make_objective_function(model, guess_table, img, to_optimize=to_optimize, fitshape=51, posbound=0.1)
        x_found = least_squares(objective, x0, bounds=bounds)
        update_guess(guess_table, x_found['x'], to_optimize)

    for run, i in tqdm(list(itertools.product(range(2), range(7)))):
        to_optimize = {i, i+1, i+2, i+3}
        objective, x0, bounds = make_objective_function(model, guess_table, img, to_optimize=to_optimize, fitshape=101, posbound=0.05)
        x_found = least_squares(objective, x0, bounds=bounds)
        update_guess(guess_table, x_found['x'], to_optimize)


    #plt.imshow(img)
    #plt.plot(table['x'], table['y'], 'gx')
    #plt.plot(guess_table[xname], guess_table[yname], 'ro')
    #plt.show()

