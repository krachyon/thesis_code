import anisocado
import scopesim
import scopesim_templates
import tempfile
import astropy
from astropy.table import Table
from astropy.modeling.functional_models import Gaussian2D
import numpy as np
import matplotlib.pyplot as plt
import os


pixel_scale = 0.004  # TODO get these from scopesim directly?
pixel_count = 1024
shift = [[0, 14]]
psf_wavelength=2.15
half_length = 512
filter_name = 'MICADO/filters/TC_filter_K-cont.dat'

@np.vectorize
def to_pixel_scale(pos):
    return pos / pixel_scale + 512


hdus = anisocado.misc.make_simcado_psf_file(
    shift, [psf_wavelength], pixelSize=pixel_scale, N=half_length)
image = hdus[2]
image.data = np.squeeze(image.data)  # remove leading dimension, we're only looking at a single picture, not a stack

x, y = image.data.shape
#TODO psf array is offset by one, hence need the -1 after coordinates?
image.data = image.data * Gaussian2D(x_stddev=5, y_stddev=5)(*np.mgrid[-x / 2:x / 2:x * 1j, -y / 2:y / 2:y * 1j] - 1)

filename = tempfile.NamedTemporaryFile('w', suffix='.fits').name
image.writeto(filename)

tmp_psf = anisocado.AnalyticalScaoPsf(N=half_length, wavelength=psf_wavelength)
strehl = tmp_psf.strehl_ratio

# Todo: passing a filename that does not end in .fits causes a weird parsing error
psf_effect = scopesim.effects.FieldConstantPSF(
    name='mypsf.fits',
    filename=filename,
    wavelength=psf_wavelength,
    psf_side_length=half_length,
    strehl_ratio=strehl, )

N1d = 30
N = N1d ** 2
spectral_types = ['A0V'] * N
border = 32
perturbation = 2

y = (np.tile(np.linspace(border, pixel_count - border, N1d), reps=(N1d, 1)) - pixel_count / 2) * pixel_scale
x = y.T
x += np.random.uniform(0, perturbation * pixel_scale, x.shape)
y += np.random.uniform(0, perturbation * pixel_scale, y.shape)

m = np.array([19]*N)

source = scopesim_templates.basic.stars.stars(filter_name=filter_name,
                                              amplitudes=m,
                                              spec_types=spectral_types,
                                              x=x.ravel(), y=y.ravel())



micado = scopesim.OpticalTrain('MICADO')

# the previous psf had that optical element so put it in the same spot.
# Todo This way of looking up the index is pretty stupid. Is there a better way?
element_idx = [element.meta['name'] for element in micado.optics_manager.optical_elements].index('default_ro')

micado.optics_manager.add_effect(psf_effect, ext=element_idx)

# disable old psf
# TODO - why is there no remove_effect with a similar interface?
#  Why do I need to go through a dictionary attached to a different class?
# TODO - would be nice if Effect Objects where frozen, e.g. with the dataclass decorator. Used ".included" first and
# TODO   was annoyed that it wasn't working...
micado['relay_psf'].include = False
micado['micado_ncpas_psf'].include = False

# TODO Apparently atmospheric dispersion is messed up. Ignore both dispersion and correction for now
micado['armazones_atmo_dispersion'].include = False
micado['micado_adc_3D_shift'].include = False

# TODO does this also apply to the custom PSF?
micado.cmds["!SIM.sub_pixel.flag"] = True


micado.observe(source, random_seed=999, updatephotometry_iterations=True)
observed_image = micado.readout()[0][1].data

table = Table((to_pixel_scale(x).ravel(), to_pixel_scale(y).ravel(), m), names=['x','y','m'])

plt.imshow(observed_image)
plt.plot(table['x'],table['y'],'ro',markersize=3)
plt.show()
