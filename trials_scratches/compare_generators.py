import numpy as np

from astropy.convolution import Gaussian2DKernel, convolve_fft
import numba
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn
#from pyfftw.interfaces.numpy_fft import fftn, ifftn
import pyfftw.interfaces.numpy_fft
from scipy.signal import fftconvolve

def gauss2d(σ_x=1., σ_y=1., a=1.):
    # faster than gauss from astropy.modelling
    @numba.njit(fastmath=True)
    def inner(x: np.ndarray, y: np.ndarray):
        return a * np.exp(-x**2 / (2 * σ_x ** 2) + -y**2 / (2 * σ_y ** 2))
    return inner

# tunable parameters
size=1024
border=64
N1d=16
perturbation=7.
σ_x=10
σ_y=10


###
# common data
###

y, x = np.indices((size, size))
# creates two columns of y,x. Ignore Cthulhuiness
yx_sources = np.mgrid[0 + border:size - border:N1d * 1j, 0 + border:size - border:N1d * 1j].transpose(
        (1, 2, 0)).reshape(-1, 2)
yx_sources += np.random.uniform(0, perturbation, (N1d ** 2, 2))

###
# Create Image with convolution
###
# whether kernel is even or odd makes a big difference...
kernel_even = Gaussian2DKernel(x_stddev=σ_x, y_stddev=σ_y, x_size=1024, y_size=1024)
kernel_odd = Gaussian2DKernel(x_stddev=σ_x, y_stddev=σ_y, x_size=1025, y_size=1025)

data_conv = np.zeros((size, size))
x_int, x_frac = np.divmod(yx_sources[:, 1], 1)
y_int, y_frac = np.divmod(yx_sources[:, 0], 1)
x_int, y_int = x_int.astype(int), y_int.astype(int)

# distribute flux over neighbours
data_conv[y_int, x_int] = (1 - x_frac) * (1 - y_frac)
data_conv[y_int + 1, x_int] = (1 - x_frac) * (y_frac)
data_conv[y_int, x_int + 1] = (x_frac) * (1 - y_frac)
data_conv[y_int + 1, x_int + 1] = y_frac * x_frac

data_conv_even = convolve_fft(data_conv, kernel_even, normalize_kernel=True, fftn=fftn, ifftn=ifftn)
#data_conv_even = fftconvolve(data_conv, kernel_even.array, mode='same')
data_conv_even /= np.max(data_conv_even)
data_conv_odd = convolve_fft(data_conv, kernel_odd, normalize_kernel=True, fftn=fftn, ifftn=ifftn)
#data_conv_odd = fftconvolve(data_conv, kernel_odd.array, mode='same')
data_conv_odd /= np.max(data_conv_odd)

data_mod = np.zeros((size, size))

###
# Create Image with summing evaluations
###
model = gauss2d(σ_x=σ_x, σ_y=σ_y)

for yshift, xshift in yx_sources:
    data_mod += model(x - xshift, y - yshift)
data_mod /= np.max(data_mod)

###
# plot results
###
table = Table((yx_sources[:, 1], yx_sources[:, 0], np.ones(N1d ** 2)), names=('x', 'y', 'm'))

plt.clf()
plt.imshow(data_conv_odd-data_mod)
plt.colorbar()
plt.plot(table['x'], table['y'], 'ro', markersize=0.5)
plt.title('odd')
plt.figure()
plt.imshow(data_conv_even-data_mod)
plt.colorbar()
plt.plot(table['x'], table['y'], 'ro', markersize=0.5)
plt.title('even')
plt.show()
