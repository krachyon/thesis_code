from astropy.modeling import fitting
from astropy.modeling.functional_models import Gaussian1D, Lorentz1D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

fitter = fitting.LevMarLSQFitter()
import pandas as pd
import multiprocess as mp
from itertools import chain, product, count
from tqdm.auto import tqdm

matplotlib.use('Qt5Agg')


class CombinedNoise:

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


def inverse_var(ys, noise):
    inv_var = 1 / (max(noise.gauss_std**2, 1e-15) + ys*noise.poisson_std**2)
    return inv_var/inv_var.max()


def inverse_var_num(ys, noise):
    noise_img = noise(ys)
    inv_var = 1 / np.sqrt(noise_img - noise_img.min() + 0.001)
    return inv_var/inv_var.max()


def anderson(ys, noise):
    noised = noise(ys)
    weights = 1 / np.sqrt(noised-noised.min() + 0.0001)
    return weights / weights.max()


def negative_var(ys, noise):
    var = (noise.gauss_std**2 + ys*noise.poisson_std**2)
    return 1 - var/var.max()


def ones(ys, noise):
    return np.ones_like(ys)


def fit(model, noise, weight_calc, iters, seed_queue):
    seed = seed_queue.get()
    np.random.seed(seed)

    xs = np.linspace(-10, 10, 3000)
    ys_perfect = model(xs)
    weights = weight_calc(ys_perfect, noise)

    fit_model = model.copy()
    xnames = ['mean', 'x_0']
    for name in xnames:
        if hasattr(fit_model, name):
            xname = name
    # for name in fit_model.param_names:
    #     if name != xname:
    #         fit_model.fixed[name] = True

    #setattr(fit_model, xname, getattr(fit_model, xname) + np.random.uniform(-0.1, +0.1))

    ret = []
    for i in range(iters):
        ys = noise(ys_perfect)

        fitted = fitter(fit_model, xs, ys, acc=1e-8, maxiter=100_000, weights=weights)

        diff = getattr(fitted, xname).value  # should be 0
        # TODO returning multiple lines here is slow
        ret.append({'deviation': diff, 'ys': ys, 'fitted': fitted,
                    'gauss_std': noise.gauss_std, 'poisson_std': noise.poisson_std,
                    'weights': weight_calc.__name__})

    return ret


# Todo make decorator?
def fit_dictarg(dict):
    return fit(**dict)


def dictoflists_to_listofdicts(dict_of_list):
    return list(dict(zip(dict_of_list.keys(), vals)) for vals in product(*dict_of_list.values()))


def plot_sigmalambda_vs_precission(results):

    grouped = results[results.weights == 'ones'].groupby(['gauss_std', 'poisson_std'], as_index=False)
    ref_mean_dev = np.array(grouped.agg({'deviation': lambda g:g.abs().mean()}).deviation)

    for weightname, res in results[results.weights != 'ones'].groupby('weights'):
        grouped = res.groupby(['gauss_std', 'poisson_std'], as_index=False)

        aggregated = grouped.agg({'deviation': lambda g:g.abs().mean()})
        gauss_std = np.array(aggregated.gauss_std)
        poisson_std = np.array(aggregated.poisson_std)
        mean_dev = np.array(aggregated.deviation)

        shape = (len(np.unique(gauss_std)), len(np.unique(poisson_std)))
        y = gauss_std.reshape(shape)
        x = poisson_std.reshape(shape)
        z = (mean_dev-ref_mean_dev).reshape(shape)

        plt.figure()
        plt.pcolormesh(x,y,z,shading='auto')
        plt.xlabel('poisson std')
        plt.ylabel('gauss std')
        plt.title(f'fit deviation {weightname} - flat')
        plt.colorbar()


def plot_lambda_vs_precission(results):
    plt.figure()
    for weightname, res in results.groupby('weights'):
        grouped = res.groupby('poisson_std', as_index=False)
        mean = grouped.agg({'deviation': lambda g: g.abs().mean()})
        std = grouped.agg({'deviation': lambda g: g.abs().std()})
        xs = mean.poisson_std
        ys = mean.deviation
        errs = std.deviation

        plt.plot(xs, ys, label=weightname)
        plt.fill_between(xs, ys - errs, ys + errs, alpha=0.4)
        plt.xlabel('Poisson σ')
        plt.ylabel('Mean deviation')
    plt.legend()


def fillqueue(q):
    for i in count():
        q.put(np.int64(i))


if __name__ == '__main__':

    model = Gaussian1D()
    #model = Lorentz1D()
    manager = mp.Manager()
    q = manager.Queue(5000)

    # params = {
    #     'noise': [CombinedNoise(0, σ, 1/λ**2) for σ in [0, 1e-3] for λ in np.linspace(1e-4, 0.05, 100)],
    #     'weight_calc': [anderson, ones],
    #     'model': [model],
    #     'iters': [30],
    #     'seed_queue': [q]
    # }
    #
    # with mp.Pool() as p:
    #     queueproc = mp.Process(target=fillqueue, args=(q,))
    #     queueproc.start()
    #     records = chain(*p.imap_unordered(fit_dictarg, tqdm(dictoflists_to_listofdicts(params)), 5))
    #     results = pd.DataFrame.from_records(records)
    #     queueproc.kill()
    # #plot_sigmalambda_vs_precission(results)
    # plot_lambda_vs_precission(results[results.gauss_std == 0])
    # plt.title('Gauss_σ=0')
    # plot_lambda_vs_precission(results[results.gauss_std != 0])
    # plt.title('Gauss_σ=1e-15')
    # plt.show()

for i in range(10):
    q.put(0)



for weight_cal in [ones, inverse_var, anderson]:
    for poisson_σ in [1e-2, 1e-1]:
        noise = CombinedNoise(0, 1e-3, (1 / poisson_σ) ** 2)
        res = fit_dictarg({'noise': noise,
                'weight_calc': weight_cal,
                'model': model,
                'iters': 1,
                'seed_queue': q})[0]

        xs = np.linspace(-10, 10, 3000)

        noised_data = res['ys']-model(xs)

        plt.figure()
        plt.plot(xs, noised_data, label='noise')
        plt.plot(xs, 10*(res['fitted'](xs)-model(xs)), label='10x deviation of fitted')
        weights = weight_cal(model(xs), noise)
        plt.plot(xs, weights/weights.max()*noised_data.max(), label='scaled weights')
        plt.title(f'{weight_cal.__name__}, poisson_σ={poisson_σ}')
        plt.legend()


plt.show()
