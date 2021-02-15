from astropy.convolution import Gaussian2DKernel, convolve_fft
from photutils.psf import EPSFModel, prepare_psf_model
import numpy as np
from astropy.table import Table
from astropy.modeling.models import Gaussian2D
from photutils.psf import *

σ = 1
img_size = 512
names = ('x_0', 'y_0', 'flux_0')

# star position
x_float = np.array([400.])
y_float = np.array([300.])

data = np.zeros((img_size, img_size))

# distribute point source over neighbour pixels
x, x_frac = np.divmod(x_float, 1)
y, y_frac = np.divmod(y_float, 1)
y, x = x.astype(int), y.astype(int)
data[x, y] = (1 - x_frac) * (1 - y_frac)
data[x + 1, y] = (x_frac) * (1 - y_frac)
data[x, y + 1] = (1 - x_frac) * (y_frac)
data[x + 1, y + 1] = y_frac * x_frac

# convolve image to simulate PSF
kernel = Gaussian2DKernel(σ, σ)
image = convolve_fft(data, kernel)
input_table = Table((x_float.ravel(), y_float.ravel(), 1000 * np.ones(x.size)), names=names)

# build EPSF that matches exactly
# +0.1 for including end value; 0.5 -> 2x oversampling
x, y = np.mgrid[-2:2.1:0.5, -2:2.1:0.5]
epsf_data = Gaussian2D(x_stddev=σ, y_stddev=σ)(x, y)

epsf = prepare_psf_model(
    EPSFModel(epsf_data, flux=1, normalize=False, oversampling=2),
    renormalize_psf=False, fluxname='flux')

# fit should not touch this
basic_phot = BasicPSFPhotometry(group_maker=DAOGroup(2),
                                bkg_estimator=None, psf_model=epsf,
                                fitshape=[5, 5])
exact_result = basic_phot(image, input_table)
#assert_allclose(input_table['x_0'], exact_result['x_fit'])
#assert_allclose(input_table['y_0'], exact_result['y_fit'])

perturbed_input = input_table.copy()
perturbed_input['x_0'] += 0.1
perturbed_input['y_0'] += 0.1

# fit should improve this
result = basic_phot(image, perturbed_input)
#assert_allclose(input_table['x_0'], result['x_fit'])
#assert_allclose(input_table['y_0'], result['y_fit'])