import scopesim
from scopesim_templates.basic.stars import stars
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

micado = scopesim.OpticalTrain('MICADO')
micado['armazones_atmo_dispersion'].include = False
micado['micado_adc_3D_shift'].include = False
micado['relay_psf'].include = False
micado['micado_ncpas_psf'].include = False
micado.cmds["!SIM.sub_pixel.flag"] = True

filter_name = 'MICADO/filters/TC_filter_K-cont.dat'
spectral_types = ['A0V']

pixel_scale = 0.004 * u.arcsec/u.pixel

n_pixel = 1024 * u.pixel
max_coord = n_pixel - 1*u.pixel #  size 1024 to max index 1023



def to_as(px_coord):
    if not isinstance(px_coord, u.Quantity):
        px_coord *= u.pixel

    # shift bounds (0,1023) to (-511.5,511.5)
    coord_shifted = px_coord - max_coord/2
    return coord_shifted * pixel_scale


def to_pixel(as_coord):
    if not isinstance(as_coord, u.Quantity):
        as_coord *= u.arcsec

    shifted_pixel_coord = as_coord / pixel_scale
    return shifted_pixel_coord + max_coord/2


# define points
x_pixel = np.array([511.5, 5, 5,      1023-5, 1023-5, 511.5, 0, 1023]) * u.pixel
y_pixel = np.array([511.5, 5, 1023-5, 5,      1023-5, 1.   , 0, 1023]) * u.pixel

x_as = to_as(x_pixel)
y_as = to_as(y_pixel)

source = stars(filter_name=filter_name,
               amplitudes=[15]*len(x_as),
               spec_types=spectral_types*len(x_as),
               x=x_as,
               y=y_as)

micado.observe(source, random_seed=1, update=True)
observed_image = micado.readout()[0][1].data

plt.imshow(observed_image)
plt.plot(x_pixel, y_pixel, 'r.', markersize=5)
plt.show()