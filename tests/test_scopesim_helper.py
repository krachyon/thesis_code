import pytest
import numpy as np
import itertools
from thesis_lib import scopesim_helper
from thesis_lib.util import center_of_image, centroid
from thesis_lib.testdata_helpers import lowpass

@pytest.fixture(scope='session', params=[0, 5])
def anisocado_model(request):
    return scopesim_helper.make_anisocado_model(lowpass=request.param)


@pytest.fixture(scope='session', params=itertools.product((0,5), (511, 512)))
def psf_effect(request):
    return scopesim_helper.make_psf(N=request.param[1], transform=lowpass(request.param[0]))


def test_anisocado_model_centered(anisocado_model):
    data = anisocado_model.data
    actual = centroid(data)
    expected = center_of_image(data)

    assert np.all((np.array(actual) - np.array(expected)) < 1e-3)


def test_anisocado_psf(psf_effect):
    data = psf_effect.data
    actual = centroid(data)
    expected = center_of_image(data)

    assert np.all((np.array(actual) - np.array(expected)) < 1e-3)
