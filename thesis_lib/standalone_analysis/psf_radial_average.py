from photutils import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.centroids import centroid_quadratic
from typing import Callable
import numpy as np
import numpy.lib.recfunctions as rf


def psf_radial_reduce(img, reduction: Callable[[np.ndarray], float] = np.mean):
    # get center of image.
    xcenter, ycenter = centroid_quadratic(img)
    # last radius in pixel where ring is fully in image
    extent = np.min(img.shape)/2

    radii = np.linspace(0.1, extent, int(extent))
    values = []

    for r_in, r_out in zip(radii, radii[1:]):
        aper = CircularAnnulus([xcenter, ycenter], r_in, r_out)
        mask = aper.to_mask('center')
        values.append(reduction(mask.get_values(img)))


    return radii[:-1], np.array(values)/np.max(img)


def cumulative_flux(img, oversampling=1):
    extent = np.min(img.shape)/2
    rs = np.linspace(1, extent, int(extent*oversampling))
    xcenter, ycenter = centroid_quadratic(img)

    apertures = [CircularAperture((xcenter, ycenter), r=r) for r in rs]
    tab = aperture_photometry(img, apertures, method='exact')
    # every aperture has it's own column; exclude first three (id,x,y)
    cumulative_flux = rf.structured_to_unstructured(tab.as_array()).ravel()[3:]

    return rs, cumulative_flux