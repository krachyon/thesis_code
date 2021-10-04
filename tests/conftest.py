import pytest

from thesis_lib.config import Config
import thesis_lib.astrometry_wrapper as astrometry_wrapper
from thesis_lib.scopesim_helper import download
from thesis_lib.testdata_recipes import convolved_grid, one_source_testimage, multi_source_testimage
from thesis_lib.testdata_generators import read_or_generate_image

from thesis_lib import scopesim_helper
from thesis_lib.testdata_helpers import lowpass

import numpy as np
from astropy.modeling.functional_models import Gaussian2D
from photutils import FittableImageModel

@pytest.fixture
def config_for_tests() -> Config:
    config = Config()
    config.cutout_size = 9
    config.fitshape = 9

    config.create_dirs()
    download()
    return config


@pytest.fixture
def session_single(config_for_tests) -> astrometry_wrapper.Session:
    image, table = read_or_generate_image('testsingle', config_for_tests, lambda: one_source_testimage())
    session = astrometry_wrapper.Session(config_for_tests, image, table)
    return session


@pytest.fixture
def session_multi(config_for_tests) -> astrometry_wrapper.Session:
    config_for_tests.photometry_iterations = 1
    image, table = read_or_generate_image('testmulti', config_for_tests, lambda: multi_source_testimage())
    session = astrometry_wrapper.Session(config_for_tests, image, table)
    return session

@pytest.fixture
def session_grid(config_for_tests):
    config_for_tests.photometry_iterations = 1
    config_for_tests.use_catalogue_positions = True
    config_for_tests.max_epsf_stars = 40

    image, table = read_or_generate_image('testgrid', config_for_tests, lambda: convolved_grid(N1d=10))
    session = astrometry_wrapper.Session(config_for_tests, image, table)
    # TODO set this? session.image_name
    # TODO add method to let it be picked up automatically? Maybe container for image, name, table?
    return session


@pytest.fixture(scope='session')
def anisocado_model(request):
    return scopesim_helper.make_anisocado_model(lowpass=request.param)


@pytest.fixture(scope='session')
def gauss_model(request):
    return scopesim_helper.make_gauss_model(request.param)


@pytest.fixture(scope='session', params=(0, 5))
def psf_effect_odd(request):
    σ = request.param
    if σ:
        return scopesim_helper.make_psf(N=511, transform=lowpass(σ))
    else:
        return scopesim_helper.make_psf(N=511)


@pytest.fixture(scope='session', params=(0, 5))
def psf_effect_even(request):  # This may not make sense
    σ = request.param
    if σ:
        return scopesim_helper.make_psf(N=512, transform=lowpass(σ))
    else:
        return scopesim_helper.make_psf(N=512)