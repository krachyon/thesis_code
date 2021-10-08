from __future__ import annotations  # makes the "-> __class__" annotation work...

import warnings
from typing import Optional, Union, Callable, Tuple

import multiprocess as mp
import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.table import Table

import photutils
from photutils import DAOStarFinder, DAOGroup, EPSFBuilder, IterativelySubtractedPSFPhotometry, MMMBackground
from thesis_lib import config
from .functions import calc_image_stats, extract_epsf_stars, perturb_guess_table, \
    match_finder_to_reference, calc_extra_result_columns, mark_outliers
from .types import REFERENCE_NAMES, StarfinderTable, InputTable, ResultTable, GuessTable, ReferenceTable, OUTLIER
from .. import util
from ..config import Config
from ..testdata.generators import read_or_generate_image


class TableSet(metaclass=util.ClassRepr):
    """responsibility: Keep different tables coherent, calculate extra columns"""

    def __init__(self, input_table: Optional[Table] = None):
        if input_table is not None:
            input_table = InputTable(input_table).validate()
        self._input_table: Optional[InputTable] = input_table
        self._finder_table: Optional[StarfinderTable] = None
        self._result_table: Optional[ResultTable] = None

    @property
    def finder_table(self):
        return self._finder_table

    @finder_table.setter
    def finder_table(self, value: StarfinderTable):
        if not isinstance(value, StarfinderTable):
            value = StarfinderTable(value).validate()
        if self._input_table:
            value = match_finder_to_reference(finder_table=value, reference_table=self._input_table)
        self._finder_table = value

    @property
    def result_table(self):
        return self._result_table

    @result_table.setter
    def result_table(self, value: ResultTable):
        if not isinstance(value, ResultTable):
            value = ResultTable(value).validate()
        if set(REFERENCE_NAMES.values()).issubset(value.colnames):
            value = calc_extra_result_columns(value)
            value = mark_outliers(value)
        self._result_table = value

    @property
    def valid_result_table(self):
        return self._result_table[~self._result_table[OUTLIER]]

    @property
    def input_table(self):
        return self._input_table

    @input_table.setter
    def input_table(self, value: InputTable):
        if not isinstance(value, InputTable):
            value = InputTable(value).validate()
        if self._finder_table:
            self._finder_table = match_finder_to_reference(finder_table=self._finder_table, reference_table=value)
        self._input_table = value

    def select_table_for_epsfstars(self, use_catalogue_positions: bool) -> InputTable:
        if use_catalogue_positions:
            return self._input_table
        elif self._result_table:
            return self._result_table.convert(InputTable)
        elif self._finder_table:
            return self._finder_table.convert(InputTable)
        else:
            raise ValueError("no star table to extract stars from. Run find_stars, photometry or provide input table")

    def select_table_for_photometry(self, config: config.Config):
        if config.use_catalogue_positions:
            # This is an ugly hack to make sure x_orig etc exists. Makes me question the whole tableType approach...
            ret = self._input_table.convert(ReferenceTable).convert(GuessTable)
            # TODO another hack to deal with the issue of different references/appertures for
            #  flux. The reference fluxes are way to low, which makes the optimizer just lower
            #  the model towards zero instead of increasing flux and adapting positions
            # TODO imho this is a sign that the objective function is bad. It should penalize
            #  a non-existent model if there's flux in the pixels
            ret.remove_column('flux_0')
            return perturb_guess_table(ret, config.perturb_catalogue_guess, config.seed)
        elif self._result_table:
            return self._result_table.convert(GuessTable)
        elif self._finder_table:
            return self._finder_table.convert(GuessTable)
        else:
            raise ValueError("Not star table to run photometry on")


class Session:
    """responsibility: initialize, dispatch and keep photutils classes/parameters in sync """

    def __init__(self, config: config.Config,
                 image: Union[str, np.ndarray],
                 input_table: Optional[Table] = None):
        self.config = config

        self.tables = TableSet(input_table)

        self._image = None
        self.image_name = 'unnamed'
        if image is not None:
            self.image = image

        if config.use_catalogue_positions:
            if self.tables.input_table is None:
                raise ValueError('Need an input catalogue to if we want to use catalogue positions')

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

    # ## Properties ## #
    @property
    def image(self) -> np.ndarray:
        return self._image
    @image.setter
    def image(self, value: Union[str, np.ndarray]):
        if isinstance(value, str):
            image, input_table = read_or_generate_image(value)
            self.image_name = value
            self._image = image
            # TODO This is done twice now. Could this have deleted something?
            self.tables.input_table = input_table
        elif isinstance(value, np.ndarray):
            self._image = value
        else:
            raise TypeError("image needs to be array or string (for predefined images)")
        self.image_stats = calc_image_stats(self._image, self.config)

    # ## End properties ## #

    # ## User Interface ## #

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
        if not self.epsfstars:
            raise ValueError("call to select_epsfstars* method required")
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
        self.grouper = DAOGroup(self.config.separation_factor*self.fwhm)
        return self

    def do_astrometry(self, initial_guess: Optional[GuessTable] = None) -> __class__:
        if not self.epsf:
            raise ValueError('Need to create an EPSF model before photometry is possible')
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
            guess_table = self.tables.select_table_for_photometry(self.config)
        self.tables.result_table = self.photometry.do_photometry(self.image, init_guesses=guess_table)
        return self

    def do_it_all(self) -> __class__:
        """Fire and forget mode"""
        # equivalent
        self.find_stars().select_epsfstars_auto().make_epsf()
        # TODO
        self.determine_psf_parameters()
        # self.cull_detections()
        # self.select_epsfstars_qof()
        self.do_astrometry()
        return self


def photometry_multi(image_recipe_template: Callable[[int], Callable[[], Tuple[np.ndarray, Table]]],
                     image_name_template: str,
                     n_images: int,
                     config=Config.instance(),
                     threads: Union[int, None, bool]=None) -> list[Session]:
    """
    """

    def inner(i):
        image_recipe = image_recipe_template(i)
        image_name = image_name_template+f'_{i}'
        image, input_table = read_or_generate_image(image_name, config, image_recipe)
        session = Session(config, image, input_table)
        session.do_it_all()

        # TODO maybe do this as part of the calc_additional function
        session.tables.result_table['σ_pos_estimated'] = \
            util.estimate_photometric_precision_full(image, input_table, session.fwhm)
        return session

    if threads is False:
        sessions = list(map(inner, range(n_images)))
    else:
        with mp.Pool(threads) as pool:
            sessions = pool.map(inner, range(n_images))

    return sessions