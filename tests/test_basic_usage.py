from thesis_lib.config import Config
from thesis_lib.testdata_generators import read_or_generate_image
import thesis_lib.astrometry_wrapper as astrometry_wrapper
from thesis_lib.scopesim_helper import download



class TestSession:

    @classmethod
    def setup_class(cls):
        config = Config()
        config.cutout_size = 9
        config.fitshape = 9
        config.create_dirs()

        cls.config = config
        download()


    def test_oneline(self):

        session = astrometry_wrapper.Session(self.config, 'testdummy')
        session.do_it_all()

        assert session.result_table
        assert len(session.result_table) == 1


    def test_session(self):

        session = astrometry_wrapper.Session(self.config, 'testdummy')

        # equivalent way of
        session.image = 'testdummy'

        image, input_table = read_or_generate_image('testdummy')
        session.image = image
        session.input_table = input_table

        session.find_stars()
        session.select_epsfstars_auto()
        session.make_epsf()
        # Here we could e.g. change starfinder and re_run find_stars()
        # TODO
        # session.cull_detections()
        # session.select_epsfstars_qof()
        session.make_epsf()
        session.do_astrometry()

        assert session.result_table
        assert len(session.result_table) == 1


    def test_builder_session(self):
        session = astrometry_wrapper.Session(self.config, 'testdummy')
        session.find_stars().select_epsfstars_auto().make_epsf().do_astrometry()

        assert session.result_table
        assert len(session.result_table) == 1
