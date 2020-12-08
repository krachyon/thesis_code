from astropy.modeling.models import Gaussian1D
from astropy.modeling import Parameter, Fittable1DModel

import numpy as np
from astropy.modeling import Fittable1DModel, Parameter, CompoundModel


class Mygauss(Fittable1DModel):
    amplitude = Parameter()
    mean = Parameter()
    stddev_1 = Parameter()

    @staticmethod
    def evaluate(x, amplitude, mean, stddev):
        return amplitude * np.exp((-(1 / (2. * stddev**2)) * (x - mean)**2))

    @staticmethod
    def fit_deriv(x, amplitude, mean, stddev):
        d_amplitude = np.exp((-(1 / (stddev**2)) * (x - mean)**2))
        d_mean = (2 * amplitude *
                  np.exp((-(1 / (stddev**2)) * (x - mean)**2)) *
                  (x - mean) / (stddev**2))
        d_stddev = (2 * amplitude *
                    np.exp((-(1 / (stddev**2)) * (x - mean)**2)) *
                    ((x - mean)**2) / (stddev**3))
        return [d_amplitude, d_mean, d_stddev]


a = Mygauss(1,1,1)
b = Mygauss(1,1,1)
#c = CompoundModel('+',a,b)
c = a+b