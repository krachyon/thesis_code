import scopesim
from scopesim_templates.basic.stars import stars
import os
import matplotlib.pyplot as plt
import numpy as np

micado = scopesim.OpticalTrain('MICADO')
micado['armazones_atmo_dispersion'].include = False
micado['micado_adc_3D_shift'].include = False

filter_name = 'MICADO/filters/TC_filter_K-cont.dat'
# That's what scopesim seemed to use for all stars.
spectral_types = ['A0V']

x = np.array([0, 0.004*500, -0.004*500, 0.004*500, -0.004*500])
y = np.array([0, 0.004*500, -0.004*500, -0.004*500, 0.004*500])

source = stars(filter_name=filter_name,
               amplitudes=[15]*5,
               spec_types=spectral_types*5,
               x=x,
               y=y)

micado.observe(source, random_seed=1, update=True)
observed_image = micado.readout()[0][1].data

plt.imshow(observed_image)
plt.plot(x/0.004 + 512, y/0.004 + 512, '.')
