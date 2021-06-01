from photutils.psf import EPSFModel, EPSFStar, FittableImageModel
from astropy.nddata import NDData
from photutils import extract_stars
import numpy as np
from astropy.table import Table
from typing import Tuple
from astropy.modeling import fitting


from thesis_lib.testdata_generators import read_or_generate_image, benchmark_images, convolved_grid, kernel_size, Gaussian2DKernel, gauss2d, model_add_grid
from thesis_lib.photometry import make_epsf_fit, make_epsf_combine
from thesis_lib.util import make_gauss_kernel

from astropy.modeling.models import Gaussian2D, Const2D, Scale
import matplotlib.pyplot as plt

# image_name = 'scopesim_grid_16_perturb2_lowpass_mag18_24'
# image_recipe = benchmark_images[image_name]
#
# img, input_table = read_or_generate_image(image_recipe, image_name)
# #TOD HAX
# input_table['x'] += 0.5
# input_table['y'] += 0.5

# img, input_table = read_or_generate_image(
#     lambda: convolved_grid(seed=1327, N1d=16, perturbation=2., kernel=Gaussian2DKernel(y_stddev=1., x_stddev=1.5, x_size=10, y_size=10)),
#     'gauss5_pert2')
# img -= 0.001

img, input_table = read_or_generate_image(lambda: model_add_grid(gauss2d(σ_x=1.5, σ_y=1.), perturbation=2.),
                                          'gauss_model')

cutout_size = 31
stars = extract_stars(NDData(img), input_table, size=(cutout_size, cutout_size))


#epsf = make_epsf_fit(stars, 5, 2, 'quadratic')
#epsf.data[epsf.data<=np.median(epsf.data)] = 0
#epsf.compute_interpolator()

#epsf = make_epsf_combine(stars, oversampling=2)
#epsf = (Scale() & Scale()) | epsf

epsf = Gaussian2D()
#epsf.theta = 0
#epsf.fixed['theta'] = True
#epsf.tied['x_stddev'] = lambda model: model.left.y_stddev
#epsf += Const2D()
epsf.shape=np.array((31,31))
epsf.oversampling=1

#oversampling = 8
#yg, xg = np.mgrid[-cutout_size/2+0.5:cutout_size/2+0.5:1/oversampling, -cutout_size/2+0.5:cutout_size/2+0.5:1/oversampling]
#epsf = EPSFModel(data=Gaussian2D(y_stddev=1., x_stddev=1.5)(xg, yg), oversampling=oversampling)


detections = np.array((input_table['x'], input_table['y'])).T

# def fit_epsf(image: np.ndarray,
#              detection: np.ndarray,
#              model: EPSFModel,
#              cutout_size: int) -> np.ndarray:
#
#
#     assert detection.shape == (2,)
#     low = np.round(detection - cutout_size/2+0.5).astype(int)
#     high = np.round(detection + cutout_size/2+0.5).astype(int)
#     cutout = image[low[0]:high[0], low[1]:high[1]]
#
#     extend = model.shape/model.oversampling
#     x, y = np.mgrid[-extend[0]/2:extend[0]/2:(high[0]-low[0])*1j, -extend[1]/2:extend[1]/2:(high[1]-low[1])*1j]
#
#     fitter = fitting.LevMarLSQFitter()
#     fitted = fitter(model, x, y, cutout)
#
#     return cutout, fitted(x,y), fitted

def get_cutout(image, position, cutout_size):
    assert position.shape == (2,)

    # clip values less than 0 to 0, greater does not matter, that's ignored
    low = np.clip(np.round(position - cutout_size/2+0.5).astype(int), 0, None)
    high = np.clip(np.round(position + cutout_size/2+0.5).astype(int), 0, None)

    # xy swap
    return image[low[1]:high[1], low[0]:high[0]]

def test_get_cutout():
    testimg = np.arange(0, 100).reshape(10, -1)

    assert np.all(get_cutout(testimg, np.array((0, 0)), 1) == np.array([[0]]))
    assert np.all(get_cutout(testimg, np.array((0, 0)), 2) == np.array([[0,1], [10,11]]))
    # goin of the edge to the upper right
    assert np.all(get_cutout(testimg, np.array((0, 0)), 3) == np.array([[0,1], [10,11]]))

    assert np.all(get_cutout(testimg, np.array((1, 1)), 1) == np.array([[11]]))

    assert np.all(get_cutout(testimg, np.array((0.9, 0.9)), 2) == np.array([[0,1], [10,11]]))
    assert np.all(get_cutout(testimg, np.array((1., 1.)), 2) == np.array([[0,1], [10,11]]))

    assert np.all(get_cutout(testimg, np.array((1.1, 1.1)), 2) == np.array([[11,12], [21,22]]))

    assert np.all(get_cutout(testimg, np.array((5,5)),1000) == testimg)


def fit_epsf(image: np.ndarray,
             detection: np.ndarray,
             model: EPSFModel,
             cutout_size: int) -> np.ndarray:


    assert detection.shape == (2,)
    cutout = get_cutout(image, detection, cutout_size)

    y, x = np.indices(cutout.shape)
    x = x - cutout.shape[1]/2 + 0.5
    y = y - cutout.shape[0]/2 + 0.5

    fitter = fitting.LevMarLSQFitter()
    fitted = fitter(model, x, y, cutout)

    return cutout, fitted(x,y), fitted


y, x = np.indices(img.shape)

residuals = []
fitted_models = []
for detection in detections:
    cutout, evaluated, fitted_model = fit_epsf(img, detection, epsf, cutout_size+1)
    residual = cutout - evaluated
    fitted_models.append(fitted_model)
    residuals.append(residual)

plt.clf()
plt.imshow(np.sum(np.array(residuals), axis=0))
plt.colorbar()
plt.show()

# from photutils import BasicPSFPhotometry, DAOGroup, MMMBackground
# phot=BasicPSFPhotometry(DAOGroup(1),MMMBackground(),epsf,[31,31])
# guess_table = input_table.copy()
# guess_table.rename_columns(['x','y'],['x_0', 'y_0'])
# phot(img,guess_table)




