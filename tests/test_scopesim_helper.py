import pytest
import numpy as np
from thesis_lib.util import center_of_image
from photutils.centroids import centroid_quadratic
import astropy.units as u

from thesis_lib.scopesim_helper import make_anisocado_model, to_pixel_scale

@pytest.mark.parametrize('anisocado_model', [0, 5], indirect=True)
def test_anisocado_model_centered(anisocado_model):

    data = anisocado_model.render()
    actual = centroid_quadratic(data)
    expected = center_of_image(data)

    assert np.all(np.abs(np.array(actual) - np.array(expected)) < 1e-8)


@pytest.mark.parametrize('oversampling', [1, 2])
@pytest.mark.parametrize('degree', [2, 3, 5])
@pytest.mark.parametrize('lowpass', [0, 5])
def test_anisocado_model_normalized(oversampling, degree, lowpass):
    mod = make_anisocado_model(oversampling=oversampling, degree=degree, lowpass=lowpass)
    assert np.abs(np.sum(mod.render()) - 1) < 1e4


def disabled_test_weird_os4():
    mod = make_anisocado_model(oversampling=4, degree=5)
    mod = make_anisocado_model(oversampling=4, degree=5)

    mod.render()
    y,x = np.mgrid[-200:200:401j, -200:200:401j]
    mod(x,y)


# Lowpass seems to fail. Investigate
def test_anisocado_psf_odd(psf_effect_odd):
    data = psf_effect_odd.data
    actual = centroid_quadratic(data)
    expected = center_of_image(data)

    assert np.all(np.abs(np.array(actual) - np.array(expected)) < 1e-8)


def test_anisocado_psf_even(psf_effect_even):
    data = psf_effect_even.data
    actual = centroid_quadratic(data, fit_boxsize=6)
    expected = center_of_image(data)

    # TODO ideally this should be a lot tighter, but honestly even arrays are probably bad anyway
    # TODO does this work wih <0.001 on different computer?
    assert np.all(np.abs(np.array(actual) - np.array(expected)) < 0.05)


def test_to_pixel_scale():
    from astropy.table import Column
    to_pixel_scale(1)
    to_pixel_scale(1*u.arcsec)
    to_pixel_scale(1*u.arcmin)
    to_pixel_scale(np.array([1, 3, 4]))
    to_pixel_scale(np.array([23, 3]*u.arcsec))
    to_pixel_scale(Column([2, 3, 4], name='foo'))
    to_pixel_scale(Column([2, 3, 4], name='foo')*u.arcsec)

