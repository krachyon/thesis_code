import pytest
from thesis_lib.util import estimate_fwhm

from astropy.modeling.functional_models import Gaussian2D
import numpy as np
from photutils import EPSFModel


class TestEstimateFWHM:
    @classmethod
    def setup_class(cls):
        cls.model = Gaussian2D(x_stddev=4,y_stddev=4, x_mean=50, y_mean=50)
        cls.data = np.zeros((101,101))
        cls.model.render(cls.data)

        cls.epsf = EPSFModel(data=cls.data)
        cls.epsf_os2 = EPSFModel(data=cls.data, oversampling=2)

    def test_estimate_fwhm_basic(self):
        est = estimate_fwhm(self.epsf)
        assert np.isclose(est, self.model.x_fwhm)
        assert np.isclose(est, self.model.y_fwhm)

    def test_estimate_fwhm_oversampling(self):
        est = estimate_fwhm(self.epsf_os2)
        assert np.isclose(est, self.model.x_fwhm/2)
        assert np.isclose(est, self.model.y_fwhm/2)




