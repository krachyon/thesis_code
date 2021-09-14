from . import config
from .astrometry_types import ImageStats, INPUT_TABLE_NAMES, STARFINDER_TABLE_NAMES, PHOTOMETRY_TABLE_NAMES, adapt_table_names
from . astrometry_functions import calc_image_stats, extract_epsf_stars

import numpy as np
from typing import Optional, Union

from astropy.table import Table

from photutils import DAOStarFinder, DAOGroup

from .testdata_generators import read_or_generate_image


class Session:
    """responsibility: initialize, dispatch and keep photutils classes/parameters in sync """
    def __init__(self, config: config.Config,
                 image: Union[str, np.ndarray],
                 input_table: Optional[Table] = None):
        self.config = config

        self.image_stats: Optional[ImageStats] = None


        #TODO Explain. Make sure that all have the same column names available
        if config.use_catalogue_positions:
            assert input_table, 'Need an input catalogue to if we want to use catalogue positions'
        self.input_table = input_table

        # internals of properties
        self._image = None
        self._input_table = None
        self._finder_table = None
        self._result_table = None

        if image:
            self.image = image


        self.starfinder = DAOStarFinder(threshold=self.image_stats.threshold,
                                        fwhm=config.fwhm_guess,
                                        sigma_radius=config.sigma_radius,
                                        sharplo=config.sharplo,
                                        sharphi=config.sharphi,
                                        roundlo=config.roundlo,
                                        roundhi=config.roundhi,
                                        exclude_border=config.exclude_border)
        self.grouper = DAOGroup(config.separation_factor*config.fwhm_guess)
        self.fitter = None
        self.background = None

        self.epsfstars = None
        self.epsf = None

    ### Properties ###
    @property
    def image(self) -> np.ndarray:
        return self._image
    @image.setter
    def image(self, value: Union[str, np.ndarray]):
        if isinstance(value, str):
            image, input_table = read_or_generate_image(value)
            self._image = image
            self.input_table = input_table
        elif isinstance(value, np.ndarray):
            self._image = value
        else:
            raise TypeError("image needs to be array or string (for predefined images)")
        self.image_stats = calc_image_stats(self._image, self.config)

    @property
    def input_table(self):
        return self._input_table
    @input_table.setter
    def input_table(self, value: Table):
        for name in INPUT_TABLE_NAMES:
            assert name in value.colnames
        self._input_table = value

    @property
    def finder_table(self):
        return self._finder_table
    @finder_table.setter
    def finder_table(self, value: Table):
        for name in STARFINDER_TABLE_NAMES:
            assert name in value.colnames
        self._finder_table = value

    @property
    def result_table(self):
        return self._finder_table
    @result_table.setter
    def result_table(self, value: Table):
        for name in PHOTOMETRY_TABLE_NAMES:
            assert name in value.colnames
        self._finder_table = value

    ### End properties ###

    ### User Interface ###
    def find_stars(self) -> None:
        self.finder_table = self.starfinder(self.image)

    def select_epsfstars_auto(self) -> None:
        # TODO refactor calculation outside of class
        if self.config.use_catalogue_positions:
            table = self.input_table
        elif self.result_table:
            table = self.result_table
        elif self.finder_table:
            table = self.finder_table
        else:
            raise ValueError("no star table to extract stars from. Run find_stars, photometry or provide input table")

        adapted_table = adapt_table_names(table, to_names=INPUT_TABLE_NAMES)
        self.epsfstars = extract_epsf_stars(self.image, self.image_stats, adapted_table, self.config)

    def select_epfsstars_qof(self) -> None:
        return NotImplemented
    def select_epsfstars_manual(self) -> None:
        return NotImplemented

    def make_epsf(self):
        assert self.epsfstars, "call to select_epsfstars* method required"
        NotImplemented


    def do_it_all(self) -> PhotometryResult:
        """Fire and forget mode"""
        # equivalent
        self.find_stars()
        self.select_epsfstars_auto()
        self.make_epsf()
        # TODO
        # self.cull_detections()
        # self.select_epsfstars_qof()
        self.make_epsf()
        return self.do_astrometry()



