from astropy.modeling import FittableModel, Fittable2DModel, Model
from scipy.interpolate import interp1d
import pandas as pd


def read_scopesim_linearity(fname):
    df = pd.read_csv(fname, comment='#', sep=r'\s+')
    return df.to_numpy()


class SaturationModel(Model):
    n_inputs = 1
    n_outputs = 1
    fittable = True
    linear = False
    fit_deriv = None

    def __init__(self, xy_data, interpolation_kind='linear'):
        self.interpolator = interp1d(xy_data[:, 0], xy_data[:, 1], kind=interpolation_kind, fill_value='extrapolate')
        self.inverse_interp = interp1d(xy_data[:, 1], xy_data[:, 0], kind=interpolation_kind, fill_value='extrapolate')
        super().__init__()

    def evaluate(self, img):
        return self.interpolator(img)

    def inverse_eval(self, img):
        return self.inverse_interp(img)


if __name__ == '__main__':

    from astropy.modeling.functional_models import Gaussian2D
    import numpy as np
    import matplotlib.pyplot as plt

    y, x = np.mgrid[-10:10:101j, -10:10:101j]
    gauss = Gaussian2D()
    gauss.amplitude = 200000

    model = gauss | SaturationModel(read_scopesim_linearity('../MICADO/FPA_linearity.dat'))

    img = model(x,y)
    diff_img = gauss(x,y) - img

    plt.imshow(diff_img)
    plt.show()