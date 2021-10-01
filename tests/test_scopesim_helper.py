import pytest
import numpy as np
from thesis_lib import scopesim_helper


def image_moment(image, x_order, y_order):
    y, x = np.indices(image.shape)
    return np.sum(x**x_order * y**y_order * image)

def centroid(image):
    m00 = image_moment(image, 0, 0)
    m10 = image_moment(image, 1, 0)
    m01 = image_moment(image, 0, 1)
    return m10/m00, m01/m00

@pytest.fixture(scope='session')
def anisocado():
    return scopesim_helper.make_anisocado_model()


@pytest.fixture(scope='session')
def anisocado_lowpass():
    return scopesim_helper.make_anisocado_model(lowpass=5)


def test_anisocado_model_centered(anisocado):
    data = anisocado.data

