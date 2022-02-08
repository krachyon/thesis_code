import pytest
import numpy as np
from photutils.centroids import centroid_quadratic, centroid_sources

from thesis_lib.testdata.helpers import lowpass
from thesis_lib.util import centroid, center_of_image, copying_lru_cache
from thesis_lib.testdata.recipes import single_star_image, scopesim_grid

from thesis_lib.util import work_in
from thesis_lib.config import Config
from thesis_lib.astrometry.types import INPUT_TABLE_NAMES, X, Y
from thesis_lib.scopesim_helper import make_anisocado_model, make_gauss_model


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


@copying_lru_cache(maxsize=2)
def make_delta_model():
    from photutils import FittableImageModel
    data = np.zeros((401, 401))
    ycenter, xcenter = center_of_image(data)
    data[int(ycenter),int(xcenter)] = 1
    return FittableImageModel(data, oversampling=2, degree=5)


# The following tests take quite long to run and tolerances are pretty low.
# Not sure if debugging this will leads somewhere

@pytest.mark.parametrize('single_star', [None,
                                         make_delta_model(),
                                         make_anisocado_model(),
                                         make_anisocado_model(lowpass=5),
                                         make_gauss_model(0.5)], indirect=True)
def test_single_star_image(single_star):
    img, table = single_star

    xcenter, ycenter = center_of_image(img)
    xcentroid, ycentroid = centroid_quadratic(img, fit_boxsize=5)
    xref, yref = table[INPUT_TABLE_NAMES[X]][0], table[INPUT_TABLE_NAMES[Y]][0]

    assert np.abs(xref-xcentroid) < 0.005
    assert np.abs(yref-ycentroid) < 0.005
    # currently stars are not placed dead center due to 0.5 pixel offset from scopesim
    #assert np.abs(xcenter-xref) < 0.01
    #assert np.abs(ycenter-yref) < 0.01


@pytest.mark.xfail
@pytest.mark.parametrize('single_star', [make_anisocado_model(lowpass=5)], indirect=True)
def test_single_star_image_scopesim_shift(single_star):
    """This test should not fail after scopesim fixes the coordinate transformation"""
    img, table = single_star

    xcenter, ycenter = center_of_image(img)
    xref, yref = table[INPUT_TABLE_NAMES[X]][0], table[INPUT_TABLE_NAMES[Y]][0]

    assert np.abs(xcenter-xref) < 0.01
    assert np.abs(ycenter-yref) < 0.01


@pytest.mark.parametrize(
    'generator_args',
     [{'N1d':3, 'border':128, 'perturbation':0},
      {'N1d':3, 'border':128, 'perturbation':2},
      {'N1d':3, 'border':128, 'perturbation':2, 'psf_transform': lowpass()},
      {'N1d':3, 'border':128, 'perturbation':2, 'custom_subpixel_psf': make_anisocado_model()},
      {'N1d':3, 'border':128, 'perturbation':2, 'custom_subpixel_psf': make_anisocado_model(lowpass=5)}
      ])
def test_grid(printer, generator_args):
    with work_in(Config.instance().scopesim_working_dir):
        img, table = scopesim_grid(**generator_args)

    xref, yref = table[INPUT_TABLE_NAMES[X]], table[INPUT_TABLE_NAMES[Y]]
    x, y = centroid_sources(img, xref, yref, box_size=7, centroid_func=centroid_quadratic)

    xdev = np.abs(x-xref)
    ydev = np.abs(y-yref)
    printer(f'{xdev=}\n{ydev=}')

    #TODO this is pretty shit accuracy sometimes...
    assert np.all(np.abs(xdev) < 0.15)
    assert np.all(np.abs(ydev) < 0.15)
    #assert np.abs(xref-xcentroid) < 0.005
    #assert np.abs(yref-ycentroid) < 0.005
    # currently stars are not placed dead center due to 0.5 pixel offset from scopesim
    #assert np.abs(xcenter-xref) < 0.01
    #assert np.abs(ycenter-yref) < 0.01

@pytest.mark.parametrize(
    'generator_args',
     [{'N1d':25, 'border':128, 'perturbation':2},
      {'N1d':25, 'border':128, 'perturbation':2, 'psf_transform': lowpass()},
      {'N1d':5, 'border':128, 'perturbation':2, 'custom_subpixel_psf': make_anisocado_model()},
      {'N1d':5, 'border':128, 'perturbation':2, 'custom_subpixel_psf': make_anisocado_model(lowpass=5)}
      ])
def test_grid_mean_devation(printer, generator_args):
    with work_in(Config.instance().scopesim_working_dir):
        img, table = scopesim_grid(**generator_args)

    xref, yref = table[INPUT_TABLE_NAMES[X]], table[INPUT_TABLE_NAMES[Y]]
    x, y = centroid_sources(img, xref, yref, box_size=7, centroid_func=centroid_quadratic)
    xdev_mean = np.mean(x-xref)
    ydev_mean = np.mean(y-yref)
    printer(f'{xdev_mean=} {ydev_mean=}')
    assert xdev_mean < 0.001
    assert ydev_mean < 0.001