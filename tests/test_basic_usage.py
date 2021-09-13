from thesis_lib.config import Config
from thesis_lib.testdata_generators import read_or_generate_image
import thesis_lib.astrometry_wrapper as astrometry_wrapper


def test_oneline():
    config = Config()
    photometry_result = astrometry_wrapper.Session(config, 'testdummy').do_it_all()
    assert photometry_result


def test_session():

    session = astrometry_wrapper.Session(Config(), 'testdummy')

    # equivalent way of
    session.image = 'testdummy'

    image, input_table = read_or_generate_image('testdummy')
    session.image = image
    session.input_table = input_table

    session.find_stars()
    session.select_epsfstars_auto()
    session.make_epsf()
    # Here we could e.g. change starfinder and re_run find_stars()
    session.cull_detections()
    session.select_epsfstars_qof()
    session.make_epsf()

    photometry_result = session.do_astrometry()
