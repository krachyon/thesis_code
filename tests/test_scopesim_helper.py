import pytest
import numpy as np
import itertools
from thesis_lib import scopesim_helper
from thesis_lib.util import center_of_image
from thesis_lib.testdata_helpers import lowpass
from photutils.centroids import centroid_quadratic

from conftest import psf_effect_odd, psf_effect_even, anisocado_model


@pytest.mark.parametrize('anisocado_model', [0, 5], indirect=True)
def test_anisocado_model_centered(anisocado_model):

    data = anisocado_model.render()
    actual = centroid_quadratic(data)
    expected = center_of_image(data)

    # TODO ideally this should be tighter
    assert np.all(np.abs(np.array(actual) - np.array(expected)) < 0.0001)


# Lowpass seems to fail. Investigate
def test_anisocado_psf_odd(psf_effect_odd):
    data = psf_effect_odd.data
    actual = centroid_quadratic(data)
    expected = center_of_image(data)

    # TODO ideally this should be a lot tighter
    assert np.all(np.abs(np.array(actual) - np.array(expected)) < 0.001)


def test_anisocado_psf_even(psf_effect_even):
    data = psf_effect_even.data
    actual = centroid_quadratic(data, fit_boxsize=6)
    expected = center_of_image(data)

    # TODO ideally this should be a lot tighter, but honestly even arrays are probably bad anyway
    assert np.all(np.abs(np.array(actual) - np.array(expected)) < 0.001)
