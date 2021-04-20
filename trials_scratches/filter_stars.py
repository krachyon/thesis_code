from photutils.psf import EPSFModel, EPSFStar
from astropy.nddata import NDData
from photutils import extract_stars
import numpy as np
from astropy.table import Table
from typing import Tuple
from astropy.modeling import fitting


from thesis_lib.testdata_generators import read_or_generate_image, benchmark_images
from thesis_lib.photometry import make_epsf_fit
from thesis_lib.util import make_gauss_kernel

import matplotlib.pyplot as plt

image_name = 'scopesim_grid_16_perturb2_lowpass_mag18_24'
image_recipe = benchmark_images[image_name]

img, input_table = read_or_generate_image(image_recipe, image_name)

stars = extract_stars(NDData(img), input_table, size=(21, 21))[:30]

epsf = make_epsf_fit(stars, 2, 2, make_gauss_kernel(0.7))

plt.imshow(epsf.data)
plt.show()


# def quality_of_fit(image: np.ndarray,
#                    detection: np.ndarray,
#                    model: EPSFModel,
#                    cutout_size: int) -> np.ndarray:

detection = np.array((245.6, 245.6))
cutout_size = 17
image = img
model = epsf

assert detection.shape == (2,)
low = np.round(detection - cutout_size/2+0.5).astype(int)
high = np.round(detection + cutout_size/2+0.5).astype(int)
cutout = image[low[0]:high[0], low[1]:high[1]]

extend = model.shape/model.oversampling
x, y = np.mgrid[-extend[0]/2:extend[0]/2:(high[0]-low[0])*1j, -extend[1]/2:extend[1]/2:(high[1]-low[1])*1j]

fitter = fitting.LevMarLSQFitter()
fitted = fitter(model, x, y, cutout)

plt.subplot(1, 3, 1)
plt.imshow(cutout)
plt.subplot(1,3,2)
plt.imshow(fitted(x,y))
plt.subplot(1,3,3)
plt.imshow(cutout - fitted(x,y))
#    return fitted

#p=quality_of_fit(img, np.array((245.6, 245.6)), epsf, 15)