import pytest

from thesis_lib.config import Config
import thesis_lib.astrometry_wrapper as astrometry_wrapper
from thesis_lib.scopesim_helper import download
from thesis_lib.testdata_recipes import convolved_grid
from thesis_lib.testdata_generators import read_or_generate_image

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
    session = astrometry_wrapper.Session(config_for_tests, 'testsingle')
    return session


@pytest.fixture
def session_multi(config_for_tests) -> astrometry_wrapper.Session:
    config_for_tests.photometry_iterations = 1
    session = astrometry_wrapper.Session(config_for_tests, 'testmulti')
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
