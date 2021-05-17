import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import fitting
from photutils.psf import EPSFModel, FittableImageModel
from astropy.modeling.models import Gaussian2D, Const2D, Scale

y, x = np.mgrid[-20:21, -30:31]
model = Gaussian2D(x_stddev=3., y_stddev=6., x_mean=0.2, y_mean=0.3)
data = model(x, y)

fitter = fitting.LevMarLSQFitter()

to_fit_analytic = Gaussian2D(amplitude=1.1, x_mean=-0.2, y_mean=0.1, x_stddev=2., y_stddev=1., theta=0.)
to_fit_analytic.fixed['theta'] = True

oversampling = 1
ym, xm = np.mgrid[-20:20:1/oversampling, -30:30:1/oversampling]
shifted_model = model.copy()
shifted_model.x_mean = 0
shifted_model.y_mean = 0
to_fit_epsf = FittableImageModel(data=shifted_model(xm, ym), oversampling=oversampling, degree=1)
to_fit_epsf_3 = FittableImageModel(data=shifted_model(xm, ym), oversampling=oversampling, degree=3)

fitted_analytic = fitter(to_fit_analytic, x, y, data)
fitted_epsf = fitter(to_fit_epsf, x, y, data)

residual = data - fitted_epsf(x, y)

pass