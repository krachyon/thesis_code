from photutils.psf import EPSFModel
from astropy.nddata import NDData
from photutils import extract_stars, EPSFBuilder
import numpy as np
from astropy.modeling import fitting

from thesis_lib.testdata.recipes import convolved_grid, Gaussian2DKernel, model_add_grid
from thesis_lib.testdata.helpers import gauss2d
from thesis_lib.standalone_analysis.sampling_precision import get_cutout_slices


from astropy.modeling.models import Gaussian2D
import matplotlib.pyplot as plt


def fit_epsf(image: np.ndarray,
             x,
             y,
             detection: np.ndarray,
             model: EPSFModel,
             cutout_size: int):


    assert detection.shape == (2,)
    cutout_slices = get_cutout_slices(detection, cutout_size)

    xname = [name for name in model.param_names for pattern in ['x_0', 'x_mean'] if name.startswith(pattern)][0]
    yname = [name for name in model.param_names for pattern in ['y_0', 'y_mean'] if name.startswith(pattern)][0]
    fluxname = [name for name in model.param_names for pattern in ['flux', 'amplitude'] if name.startswith(pattern)][0]
    # nail down everything extra that could be changed by the fit, we only want to optimize position and flux


    getattr(model, xname).value = detection[0] + np.random.uniform(-0.2, 0.2)
    getattr(model, yname).value = detection[1] + np.random.uniform(-0.2, 0.2)
    getattr(model, fluxname).value = 1

    fitter = fitting.LevMarLSQFitter()
    fitted = fitter(model, x[cutout_slices], y[cutout_slices], image[cutout_slices])

    return fitted


def compute_residual(img, model):

    residual = img.copy()
    y, x = np.indices(img.shape)

    #fitted_models = []
    for detection in detections:
        fitted_model = fit_epsf(img, x, y, detection, model, cutout_size+2)
        residual -= fitted_model(x,y)
        #fitted_models.append(fitted_model)
    return residual


def normalize(img):
    return img-img.min()+1


if __name__ == "__main__":
    img_conv, input_table_conv = convolved_grid(N1d=5, border=100, perturbation=2., kernel=Gaussian2DKernel(x_stddev=10, y_stddev=12, x_size=101, y_size=101), seed=1327)
    img_conv -= 0.00
    img_mod, input_table_mod = model_add_grid(gauss2d(σ_x=10., σ_y=12), N1d=5, border=100, perturbation=2., seed=1327)

    cutout_size = 181
    stars_mod = extract_stars(NDData(img_mod), input_table_mod, size=(cutout_size, cutout_size))
    stars_conv = extract_stars(NDData(img_conv), input_table_mod, size=(cutout_size, cutout_size))

    epsf_analytic = Gaussian2D()

    epsf_fit_mod = EPSFBuilder(maxiters=8, oversampling=2, smoothing_kernel='quadratic').build_epsf(stars_mod)
    epsf_fit_conv = EPSFBuilder(maxiters=8, oversampling=2, smoothing_kernel='quadratic').build_epsf(stars_conv)


    detections = np.array((input_table_mod['x'], input_table_mod['y'])).T



    residual_ana_mod = compute_residual(img_mod, epsf_analytic)
    residual_ana_conv = compute_residual(img_conv, epsf_analytic)
    residual_epsf_mod = compute_residual(img_mod, epsf_fit_mod)
    residual_epsf_conv = compute_residual(img_conv, epsf_fit_conv)

    fig, axs = plt.subplots(2,2)
    im = axs[0,0].imshow(residual_epsf_conv)
    fig.colorbar(im, ax=axs[0,0])
    axs[0,0].set_title('EPSF fit to convolved')

    im = axs[0,1].imshow(residual_epsf_mod)
    fig.colorbar(im, ax=axs[0,1])
    axs[0,1].set_title('EPSF fit to model_eval')

    im = axs[1,0].imshow(residual_ana_conv)
    fig.colorbar(im, ax=axs[1,0])
    axs[1,0].set_title('Analytic fit to convolved')

    im = axs[1,1].imshow(residual_ana_mod)
    fig.colorbar(im, ax=axs[1,1])
    axs[1,1].set_title('Analytic fit to model_eval')

    fig.suptitle('Residuals')
    plt.tight_layout()

    plt.show()



