from __future__ import annotations  # makes the "-> __class__" annotation work...

import photutils

from . import config
from .astrometry_types import ImageStats,\
    INPUT_TABLE_NAMES, STARFINDER_TABLE_NAMES, PHOTOMETRY_TABLE_NAMES, GUESS_TABLE_NAMES, REFERENCE_NAMES, INPUT_TABLE_NAMES,\
    StarfinderTable, InputTable, ResultTable, TypeCheckedTable, GuessTable
from . astrometry_functions import calc_image_stats, extract_epsf_stars, perturb_guess_table, match_finder_to_reference, calc_extra_result_columns

import numpy as np
from typing import Optional, Union, TypeVar
import warnings

from astropy.table import Table

from photutils import DAOStarFinder, DAOGroup, EPSFBuilder, IterativelySubtractedPSFPhotometry, MMMBackground
from astropy.modeling.fitting import LevMarLSQFitter

from .testdata_generators import read_or_generate_image
from . import util


class TableSet(metaclass=util.ClassRepr):
    """responsibility: Keep different tables coherent, calculate extra columns"""

    def __init__(self, input_table: Optional[Table] = None):
        self._input_table: Optional[InputTable] = input_table
        self._finder_table: Optional[StarfinderTable] = None
        self._result_table: Optional[ResultTable] = None

    @property
    def finder_table(self):
        return self._finder_table

    @finder_table.setter
    def finder_table(self, value: StarfinderTable):
        if not isinstance(value, StarfinderTable):
            value = StarfinderTable(value)
        if self._input_table:
            value = match_finder_to_reference(finder_table=value, reference_table=self._input_table)
        self._finder_table = value

    @property
    def result_table(self):
        return self._result_table

    @result_table.setter
    def result_table(self, value: ResultTable):
        if not isinstance(value, ResultTable):
            value = ResultTable(value)
        if set(REFERENCE_NAMES.values()).issubset(value.colnames):
            value = calc_extra_result_columns(value)
        self._result_table = value

    @property
    def input_table(self):
        return self._input_table

    @input_table.setter
    def input_table(self, value: InputTable):
        if not isinstance(value, InputTable):
            value = InputTable(value)
        if self._finder_table:
            self._finder_table = match_finder_to_reference(finder_table=self._finder_table, reference_table=value)
        self._input_table = value

    def select_table_for_epsfstars(self, use_catalogue_positions: bool) -> InputTable:
        if use_catalogue_positions:
            return self._input_table
        elif self._result_table:
            return self._result_table.typecast(InputTable)
        elif self._finder_table:
            return self._finder_table.typecast(InputTable)
        else:
            raise ValueError("no star table to extract stars from. Run find_stars, photometry or provide input table")

    def select_table_for_photometry(self, use_catalogue_positions):
        if use_catalogue_positions:
            return self._input_table.typecast(GuessTable)
        elif self._result_table:
            return self._result_table.typecast(GuessTable)
        elif self._finder_table:
            return self._finder_table.typecast(GuessTable)
        else:
            raise ValueError("Not star table to run photometry on")



class Session:
    """responsibility: initialize, dispatch and keep photutils classes/parameters in sync """

    def __init__(self, config: config.Config,
                 image: Union[str, np.ndarray],
                 input_table: Optional[Table] = None):
        self.config = config
        self.random_seed = 0
        self.tables = TableSet(input_table)

        if config.use_catalogue_positions:
            assert input_table, 'Need an input catalogue to if we want to use catalogue positions'

        self._image = None
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
        self.fitter = LevMarLSQFitter()
        self.background = MMMBackground()
        self.photometry = None

        self.epsfstars: Optional[photutils.EPSFStars] = None
        self.epsf: Optional[photutils.EPSFModel] = None
        self.fwhm: Optional[float] = None

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



    ### End properties ###

    ### User Interface ###

    def find_stars(self) -> __class__:
        if self._image is None:
            raise ValueError('need to attach image first')
        self.tables.finder_table = self.starfinder(self.image)
        return self

    def select_epsfstars_auto(self, table_to_use: Optional[InputTable] = None) -> __class__:
        if not table_to_use:
            table_to_use = self.tables.select_table_for_epsfstars(self.config.use_catalogue_positions)
        self.epsfstars = extract_epsf_stars(self.image, self.image_stats, table_to_use, self.config)
        return self

    def select_epfsstars_qof(self) -> __class__:
        raise NotImplementedError

    def select_epsfstars_manual(self) -> __class__:
        raise NotImplementedError

    def make_epsf(self, epsf_guess=None) -> __class__:
        assert self.epsfstars, "call to select_epsfstars* method required"
        try:
            builder = EPSFBuilder(oversampling=self.config.oversampling,
                                  maxiters=self.config.epsfbuilder_iters,
                                  progress_bar=True,
                                  smoothing_kernel=self.config.smoothing)
            self.epsf, fitted_stars = builder.build_epsf(self.epsfstars, init_epsf=epsf_guess)
        except ValueError:
            warnings.warn('Epsf fit diverged. Some data will not be analyzed')
            raise
        return self

    def determine_psf_parameters(self) -> __class__:
        self.fwhm = util.estimate_fwhm(self.epsf)
        return self

    def do_astrometry(self, initial_guess: Optional[GuessTable] = None) -> __class__:
        assert self.epsf, 'Need to create an EPSF model before photometry is possible'
        self.photometry = IterativelySubtractedPSFPhotometry(
            finder=self.starfinder,
            group_maker=self.grouper,
            bkg_estimator=self.background,
            psf_model=self.epsf,
            fitter=self.fitter,
            niters=self.config.photometry_iterations,
            fitshape=self.config.fitshape
        )
        if initial_guess:
            guess_table = initial_guess
        else:
            guess_table = self.tables.select_table_for_photometry(self.config.use_catalogue_positions)
            # TODO ugly hack to make custom tables sliceable
        self.tables.result_table = self.photometry.do_photometry(self.image, init_guesses=Table(guess_table))
        return self

    def do_it_all(self) -> __class__:
        """Fire and forget mode"""
        # equivalent
        self.find_stars().select_epsfstars_auto().make_epsf()
        # TODO
        # self.determine_psf_parameters()
        # self.cull_detections()
        # self.select_epsfstars_qof()
        self.do_astrometry()
        return self



