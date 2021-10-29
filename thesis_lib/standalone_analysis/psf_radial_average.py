from photutils import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.centroids import centroid_quadratic
from typing import Callable
import numpy as np


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
