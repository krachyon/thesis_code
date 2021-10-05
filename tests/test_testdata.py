import pytest
import numpy as np

from thesis_lib.testdata_helpers import lowpass
from thesis_lib.util import centroid, center_of_image
from thesis_lib.testdata_recipes import single_star_image

from thesis_lib.util import work_in
from thesis_lib.config import Config
from thesis_lib.astrometry_types import INPUT_TABLE_NAMES, X, Y
from thesis_lib.scopesim_helper import make_anisocado_model, make_gauss_model
from conftest import anisocado_model


@pytest.mark.parametrize('xsize', [51, 50, 101])
@pytest.mark.parametrize('ysize', [51, 50, 101])
def test_lowpass(xsize, ysize):
    img = np.ones((ysize, xsize))
    transform = lowpass(5)

    assert np.all(np.isclose(centroid(transform(img)), center_of_image(img)))


@pytest.fixture(scope='session')
def single_star(request):
    with work_in(Config.instance().scopesim_working_dir):
        img, table = single_star_image(custom_subpixel_psf=request.param)
    return img, table


def make_delta_model():
    from photutils import FittableImageModel
    data = np.zeros((400,400))
    ycenter, xcenter = center_of_image(data)
    data[int(ycenter),int(xcenter)] = 1
    return FittableImageModel(data, oversampling=2, degree=5)


@pytest.mark.parametrize('single_star', [None,
                                         make_delta_model(),
                                         make_anisocado_model(),
                                         make_anisocado_model(lowpass=5),
                                         make_gauss_model(5)], indirect=True)
def test_single_star_image(single_star):
    from photutils.centroids import centroid_quadratic
    img, table = single_star

    xcenter, ycenter = center_of_image(img)
    xcentroid, ycentroid = centroid_quadratic(img, fit_boxsize=31)
    xref, yref = table[INPUT_TABLE_NAMES[X]][0], table[INPUT_TABLE_NAMES[Y]][0]

    assert np.abs(xref-xcentroid) < 0.01
    assert np.abs(yref-ycentroid) < 0.01
    # currently stars are not placed dead center due to 0.5 pixel offset from scopesim
    #assert np.abs(xcenter-xref) < 0.01
    #assert np.abs(ycenter-yref) < 0.01
