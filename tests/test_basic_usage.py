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

        session = astrometry_wrapper.Session(self.config, 'testsingle')
        session.do_it_all()

        assert session.tables.result_table
        assert len(session.tables.result_table) == 1


    def test_session(self):

        session = astrometry_wrapper.Session(self.config, 'testsingle')

        # equivalent way of
        session.image = 'testsingle'

        image, input_table = read_or_generate_image('testsingle')
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

        assert session.tables.result_table
        assert len(session.tables.result_table) == 1


    def test_builder_session(self):
        session = astrometry_wrapper.Session(self.config, 'testsingle')
        session.find_stars().select_epsfstars_auto().make_epsf().do_astrometry()

        assert session.tables.result_table
        assert len(session.tables.result_table) == 1


    def test_multi_image(self):
        config = self.config.copy()
        config.photometry_iterations = 1
        session = astrometry_wrapper.Session(config, 'testmulti')
        session.find_stars()
        assert len(session.input_table) == len(session.tables.finder_table)
        session.select_epsfstars_auto().make_epsf().do_astrometry()
        assert len(session.tables.result_table) == len(session.input_table)