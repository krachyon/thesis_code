import anisocado
import numpy as np
from photutils import CircularAperture, aperture_photometry
import numpy.lib.recfunctions as rf

img = anisocado.AnalyticalScaoPsf().psf_on_axis


def radial_average(img, oversampling=1):
    max_radius = min(img.shape)
    rs = np.linspace(1, max_radius/2, max_radius*oversampling)
    y, x = np.array(img.shape)/2-0.5
    apertures = [CircularAperture((x, y), r=r) for r in rs]
    tab = aperture_photometry(img, apertures, method='exact')
    # every aperture has it's own column; exclude first three (id,x,y)
    cumulative_flux = rf.structured_to_unstructured(tab.as_array()).ravel()[3:]

    rs_reduced = rs[1:]
    radial_average = np.diff(cumulative_flux)/np.diff(rs**2)

    return rs_reduced, radial_average


def xcut(img):
    if img.shape[1]%2 == 0:
        start = int(img.shape[1]/2-1)
        stop = start+2
        return np.sum(img[:, start:stop], axis=1)/2
    else:
        return img[:, img.shape[1]//2].ravel()


def ycut(img):
    if img.shape[0]%2 == 0:
        start = int(img.shape[0]/2-1)
        stop = start+2
        return np.sum(img[start:stop, :], axis=0)/2
    else:
        return img[img.shape[0]//2, :].ravel()


import matplotlib.pyplot as plt

plt.plot(*radial_average(img, oversampling=2), label='radial profile')
x_profile = xcut(img)
x_indices = (np.indices(x_profile.shape)-np.mean(np.indices(x_profile.shape))).ravel()
plt.plot(x_indices, x_profile, label='x cut')

y_profile = ycut(img)
y_indices = (np.indices(y_profile.shape)-np.mean(np.indices(y_profile.shape))).ravel()
plt.plot(y_indices, y_profile, label='y cut')

plt.legend()
plt.yscale('log')
plt.xlim(0, 255)
plt.show()