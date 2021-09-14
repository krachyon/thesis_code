# Global values and names
import dataclasses
from typing import Optional, Union, Tuple
import numpy as np
import os
import appdirs

import matplotlib.pyplot as plt

from photutils.psf import EPSFModel
from .util import ClassRepr


@dataclasses.dataclass(init=True, repr=True, eq=False, order=False)
class Config(metaclass=ClassRepr):
    """
    Container for all parameters for the EPSF photometry pipeline.
    """
    _instance = None

    @classmethod
    def instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance

    # names
    psf_name: str = 'anisocado_psf'
    image_folder: str = 'test_images'
    output_folder: str = 'output_files'

    # TODO allow choosing starfinder type, maybe background as well?
    # magic parameters for Starfinder
    clip_sigma: float = 3.0  # sigma_clipping to apply for star guessing
    threshold_factor: float = 3.  # how many stds brighter than image for star to be detected?
    fwhm_guess: float = 7.  # estimate of PSF fwhm
    sigma_radius: float = 1.5  # clipping radius of starfinder kernel
    separation_factor: float = 2.  # How far do stars need to be apart to be considered a group?
    sharplo: float = 0.2
    sharphi: float = 1.0
    roundlo: float = -1.0
    roundhi: float = 1.0
    exclude_border: bool = True

    # magic parameters for EPSFBuilder
    max_epsf_stars: int = 200
    cutout_size: int = 50  # TODO PSF is pretty huge, right?
    fitshape: Union[int, Tuple[int, int]] = 49  # this should probably be equal or slightly less than the epsf model dimension
    oversampling: int = 4
    epsfbuilder_iters: int = 5
    smoothing: Union[str, np.ndarray] = 'quartic'
    epsf_guess: Optional[EPSFModel] = None

    # photometry
    # use known star positions from input catalogue as initial guess for photometry?
    use_catalogue_positions: bool = False
    photometry_iterations: int = 3

    # TODO implement these
    disable_detector_saturation: bool = False
    save_plots: bool = False
    known_psf_model: Optional[EPSFModel] = None
    # TODO it would be nice to use this for starfinder/epsfstarfinding
    detector_saturation: float = np.inf
    # TODO do this with chdir context manager
    scopesim_working_dir: str = appdirs.user_cache_dir('scopesim_workspace')


    def create_dirs(self):
        for dirname in [self.image_folder, self.output_folder]:
            if not os.path.exists(dirname):
                os.mkdir(dirname)


# TODO where should that go?
# matplotlib
plt.rcParams['figure.figsize'] = (8.3, 5.8)  # A5 paper. Friggin inches...
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 8
plt.rcParams['figure.autolayout'] = True
