import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import fitting
from astropy.modeling.functional_models import Gaussian1D, Trapezoid1D

fitter = fitting.LevMarLSQFitter()
import pandas as pd
import multiprocess as mp
from itertools import chain, product, count
from tqdm.auto import tqdm
from matplotlib.colors import Normalize

xs = np.linspace(-10, 10, 300)


def anderson(ys, σ, λ):
    # assume that each pixel has at least one count
    weights = 1 / np.sqrt((ys-ys.min())+1)
    return weights
    #return weights / weights.max()


def anderson_gauss(ys, σ, λ):
    # modify by taking gaussian noise into account
    weights = 1 / np.sqrt((ys-ys.min()) + 1 + σ**2)
    return weights


weights_ramp = Trapezoid1D(width=2.5, slope=0.5)(xs)
def anderson_gauss_ramp(ys, σ, λ):
    weights_statistics = 1 / np.sqrt((ys - ys.min()) + 1 + σ ** 2)
    return weights_statistics*weights_ramp


def anderson_gauss_double(ys, σ, λ):
    # modify by taking gaussian noise into account
    # TODO try if double sqrt helps
    weights = 1 / np.sqrt(np.sqrt((ys-ys.min()) + 1 + σ**2))
    return weights


def ones(ys, σ, λ):
    return np.ones_like(ys)


def fit(model, σ, λ, weight_calc, iters, seed_queue=None, return_arrays=True):

    model.amplitude = λ
    ys_normed = model(xs)


    fit_model = model.copy()
    xnames = ['mean', 'x_0']
    for name in xnames:
        if hasattr(fit_model, name):
            xname = name
    # for name in fit_model.param_names:
    #     if name != xname:
    #         fit_model.fixed[name] = True

    setattr(fit_model, xname, getattr(fit_model, xname) + np.random.uniform(-0.1, +0.1))

    if seed_queue:
        seed = seed_queue.get()
        np.random.seed(seed)

    ret = []
    for i in range(iters):
        ys = np.random.poisson(ys_normed) + np.random.normal(0, σ, ys_normed.shape)
        weights = weight_calc(ys, σ, λ)

        fitted = fitter(fit_model, xs, ys, acc=1e-8, maxiter=100_000, weights=weights)

        diff = getattr(fitted, xname).value  # should be 0
        # TODO returning multiple lines here is slow
        if return_arrays:
            ret.append({'deviation': diff, 'ys': ys, 'fitted': fitted,
                        'σ': σ, 'λ': λ,
                        'weight_calc': weight_calc.__name__, 'weights': weights})
        else:
            ret.append({'deviation': diff,
                        'σ': σ, 'λ': λ,
                        'weight_calc': weight_calc.__name__})

    return ret


# Todo make decorator?
def fit_dictarg(dict):
    return fit(**dict)


def dictoflists_to_listofdicts(dict_of_list):
    return list(dict(zip(dict_of_list.keys(), vals)) for vals in product(*dict_of_list.values()))



def plot_sigmalambda_vs_precission(results):

    grouped = results[results.weight_calc == 'ones'].groupby(['σ', 'λ'], as_index=False)
    ref_mean_dev = np.array(grouped.agg({'deviation': lambda g:g.abs().mean()}).deviation)
    figs = []

    for weightname, res in results[results.weight_calc != 'ones'].groupby('weight_calc'):
        grouped = res.groupby(['σ', 'λ'], as_index=False)

        aggregated = grouped.agg({'deviation': lambda g:g.abs().mean()})
        gauss_std = np.array(aggregated.σ)
        poisson_lamb = np.array(aggregated.λ)
        mean_dev = np.array(aggregated.deviation)

        shape = (len(np.unique(gauss_std)), len(np.unique(poisson_lamb)))
        y = gauss_std.reshape(shape)
        x = poisson_lamb.reshape(shape)
        z = (mean_dev/ref_mean_dev).reshape(shape)

        plt.figure()
        plt.pcolormesh(x, y, z, shading='auto', cmap='seismic', norm=Normalize(0, 2))
        plt.xlabel('poisson λ')
        plt.ylabel('gauss σ')
        plt.title(f'mean fit deviation {weightname}/flat')
        plt.colorbar()
        figs.append(plt.gcf())
    return figs


def plot_lambda_vs_precission(results):
    plt.figure()
    for weightname, res in results.groupby('weight_calc'):
        grouped = res.groupby('λ', as_index=False)
        mean = grouped.agg({'deviation': lambda g: g.abs().mean()})
        std = grouped.agg({'deviation': lambda g: g.abs().std()})
        xs = mean.λ
        ys = mean.deviation
        errs = std.deviation

        plt.plot(xs, ys, label=weightname)
        plt.fill_between(xs, ys - errs, ys + errs, alpha=0.4)
        plt.xlabel('Pixel Counts')
        plt.ylabel('Mean deviation')
        plt.ylim(1e-4, 1e-2)
        plt.yscale('log')
    plt.legend()
    return [plt.gcf()]


def plot_lambda_vs_precission_relative(results):
    plt.figure()
    ref_results = results[results.weight_calc == 'ones'].groupby('λ', as_index=False)
    ref_mean = ref_results.agg({'deviation': lambda g: g.abs().mean()})
    ref_std = ref_results.agg({'deviation': lambda g: g.abs().std()})

    for weightname, res in results[results.weight_calc != 'ones'].groupby('weight_calc'):
        grouped = res.groupby('λ', as_index=False)
        mean = grouped.agg({'deviation': lambda g: g.abs().mean()})
        std = grouped.agg({'deviation': lambda g: g.abs().std()})
        xs = mean.λ
        ys = mean.deviation/ref_mean.deviation
        errs = std.deviation/ref_mean.deviation # todo this is sketchy

        plt.plot(xs, ys, label=weightname)
        # plt.fill_between(xs, ys - errs, ys + errs, alpha=0.4)
    plt.axhline(1, ls=':', color='k')
    plt.xlabel('Peak pixel count')
    plt.ylabel('Relative deviation between weighted and unweighted fit')

    plt.legend()
    return [plt.gcf()]


def plot_weightshape(results, model):
    figs = []
    for wcal, group in results.groupby('weight_calc'):
        fig, axs = plt.subplots(2, 2)
        axs = axs.ravel()
        for i, (σ, λ) in enumerate([(σ, λ) for λ in [group.λ.max(), group.λ.min()]
                                  for σ in [group.σ.min(),group.σ.max()]]):

            res = group[(group.σ == σ) & (group.λ == λ)].iloc[0]
            current_model = model.copy()

            current_model.amplitude = λ
            noise = res.ys - current_model(xs)

            weights = res.weights
            # the order of magnitude the residual will have at each point
            effective_weights = (res.ys * weights)**2

            axs[i].plot(xs, noise, label='noise')
            axs[i].plot(xs, 10 * (res.fitted(xs) - current_model(xs)), label='10x deviation of fitted')
            axs[i].plot(xs, weights / weights.max() * noise.max(), label='scaled weights')
            axs[i].plot(xs, effective_weights/effective_weights.max() * noise.max(), label='effective weights scaled')
            axs[i].set_title(f'counts={λ}, gauss_σ={σ}')
            axs[i].legend()
        plt.suptitle(wcal)
        plt.tight_layout()
        figs.append(fig)
    return figs


def fillqueue(q):
    for i in count():
        q.put(np.int64(i))


if __name__ == '__main__':

    model = Gaussian1D()
    #fit(model,0,200_000,anderson,1)
    #model = Lorentz1D()
    manager = mp.Manager()
    q = manager.Queue(5000)

    params = {
        'λ': [λ for λ in np.linspace(50, 100_000, 300)],
        'σ': [0, 5, 50],
        'weight_calc': [anderson_gauss, ones, anderson],
        'model': [model],
        'iters': [100],
        'seed_queue': [q],
        'return_arrays': [False],
    }


    with mp.Pool() as p:
        queueproc = mp.Process(target=fillqueue, args=(q,))
        queueproc.start()
        records = chain(*p.imap_unordered(fit_dictarg, tqdm(dictoflists_to_listofdicts(params)), 1))
        #records = chain(*map(fit_dictarg, tqdm(dictoflists_to_listofdicts(params))))
        results = pd.DataFrame.from_records(records)
        queueproc.kill()


    for σ in results.σ.unique():
        plot_lambda_vs_precission(results[results.σ == σ])
        plt.title(f'Gauss_σ={σ}')
        plot_lambda_vs_precission_relative(results[results.σ == σ])
        plt.title(f'Gauss_σ={σ}')

    #plot_weightshape(results, model)


plt.show()
