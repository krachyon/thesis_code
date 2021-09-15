from __future__ import annotations  # makes the "-> __class__" annotation work...

from . import config
from .astrometry_types import ImageStats, adapt_table_names, \
    INPUT_TABLE_NAMES, STARFINDER_TABLE_NAMES, PHOTOMETRY_TABLE_NAMES, GUESS_TABLE_NAMES
from . astrometry_functions import calc_image_stats, extract_epsf_stars, perturb_guess_table

import numpy as np
from typing import Optional, Union, TypeVar
import warnings

from astropy.table import Table

from photutils import DAOStarFinder, DAOGroup, EPSFBuilder, IterativelySubtractedPSFPhotometry, MMMBackground
from astropy.modeling.fitting import LevMarLSQFitter

from .testdata_generators import read_or_generate_image



class Session:
    """responsibility: initialize, dispatch and keep photutils classes/parameters in sync """

    def __init__(self, config: config.Config,
                 image: Union[str, np.ndarray],
                 input_table: Optional[Table] = None):
        self.config = config
        self.image_stats: Optional[ImageStats] = None
        self.random_seed = 0


        # internals of properties
        self._image = None
        self._input_table = input_table
        self._finder_table = None
        self._result_table = None

        if config.use_catalogue_positions:
            assert input_table, 'Need an input catalogue to if we want to use catalogue positions'


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
        for name in INPUT_TABLE_NAMES.values():
            assert name in value.colnames
        self._input_table = value

    @property
    def finder_table(self):
        return self._finder_table
    @finder_table.setter
    def finder_table(self, value: Table):
        for name in STARFINDER_TABLE_NAMES.values():
            assert name in value.colnames
        self._finder_table = value

    @property
    def result_table(self):
        return self._result_table
    @result_table.setter
    def result_table(self, value: Table):
        for name in PHOTOMETRY_TABLE_NAMES.values():
            assert name in value.colnames
        self._result_table = value


    ### End properties ###

    ### User Interface ###

    def find_stars(self) -> __class__:
        self.finder_table = self.starfinder(self.image)
        return self

    def select_epsfstars_auto(self) -> __class__:
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
        return self

    def select_epfsstars_qof(self) -> None:
        return NotImplemented
    def select_epsfstars_manual(self) -> None:
        return NotImplemented

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

    def do_astrometry(self, initial_guess: Optional[Table] = None) -> __class__:
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
        elif self.config.use_catalogue_positions:
            assert self.input_table
            guess_table = perturb_guess_table(self.input_table, self.random_seed)
        elif self.result_table:
            guess_table = self.result_table
        elif self.finder_table:
            guess_table = self.finder_table
        else:
            raise ValueError('No star positions known yet')
        guess_table = adapt_table_names(guess_table, to_names=GUESS_TABLE_NAMES)
        self.result_table = self.photometry.do_photometry(self.image, init_guesses=guess_table)
        return self

    def do_it_all(self) -> __class__:
        """Fire and forget mode"""
        # equivalent
        self.find_stars()
        self.select_epsfstars_auto()
        self.make_epsf()
        # TODO
        # self.cull_detections()
        # self.select_epsfstars_qof()
        self.make_epsf()
        self.do_astrometry()
        return self



