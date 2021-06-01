import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel, convolve_fft
from photutils.psf import EPSFModel, prepare_psf_model
import numpy as np
from astropy.table import Table
from astropy.modeling.models import Gaussian2D
from photutils.psf import *
from astropy.nddata import NDData

cutout_size = 30

σ = 8
from thesis_lib.testdata_generators import read_or_generate_image, convolved_grid

img, input_table = read_or_generate_image(
        lambda: convolved_grid(8, perturbation=8, kernel=Gaussian2DKernel(x_stddev=σ, x_size=201, y_size=201)),
        'prepare_model_test'
)

y, x = np.mgrid[-2*σ:2*σ+0.1:0.5, -2*σ:2*σ+.1:0.5]
epsf_data = Gaussian2D(x_stddev=σ, y_stddev=σ)(x, y)

epsf_ana_pre = EPSFModel(epsf_data, flux=1, normalize=False, oversampling=2)
epsf_ana = prepare_psf_model(epsf_ana_pre.copy(), renormalize_psf=False, fluxname='flux')

stars = extract_stars(NDData(img), input_table, size=cutout_size)
epsf_fit_pre, _ = EPSFBuilder(oversampling=2, maxiters=3, progress_bar=True)(stars)
epsf_fit = prepare_psf_model(epsf_fit_pre.copy(), renormalize_psf=False, fluxname='flux')

basic_phot = BasicPSFPhotometry(group_maker=DAOGroup(2),
                                bkg_estimator=None, psf_model=epsf_fit,
                                fitshape=[5, 5])

guess_table = input_table.copy()
guess_table.rename_columns(['x', 'y'], ['x_0', 'y_0'])
exact_result = basic_phot(img, guess_table)
#assert_allclose(input_table['x_0'], exact_result['x_fit'])
#assert_allclose(input_table['y_0'], exact_result['y_fit'])

perturbed_input = guess_table.copy()
perturbed_input['x_0'] += 2.
perturbed_input['y_0'] += 2.
perturbed_input['flux_0'] = 0

# fit should improve this
result = basic_phot(img, perturbed_input)
#assert_allclose(input_table['x_0'], result['x_fit'])
#assert_allclose(input_table['y_0'], result['y_fit'])

plt.imshow(img)
plt.plot(result['x_fit'], result['y_fit'], 'go', markersize=0.5)
plt.plot(exact_result['x_fit'], exact_result['y_fit'], 'go', markersize=2,
         fillstyle='none', markeredgewidth=0.5, markeredgecolor='red',)
plt.show()