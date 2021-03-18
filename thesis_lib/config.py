# Global values and names
import dataclasses
from typing import Optional, Union, Tuple
import numpy as np
import os

import matplotlib.pyplot as plt

from photutils.psf import EPSFModel


class ClassRepr(type):
    """
    Use this as a metaclass to make a class (and not just an instance of it) print its contents.
    Kinda hacky and doesn't really consider edge-cases
    """

    def __new__(mcs, *args, **kwargs):
        return super().__new__(mcs, *args, **kwargs)

    def __repr__(cls):
        items = [item for item in cls.__dict__.items() if not item[0].startswith('__')]
        item_string = ', '.join([f'{item[0]} = {item[1]}' for item in items])
        return f'{cls.__name__}({item_string})'

    def __str__(cls):
        return repr(cls)


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
    separation_factor: float = 2.  # How far do stars need to be apart to be considered a group?

    # magic parameters for EPSFBuilder
    stars_to_keep: int = 200
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

    def create_dirs(self):
        for dirname in [self.image_folder, self.output_folder]:
            if not os.path.exists(dirname):
                os.mkdir(dirname)


# matplotlib
plt.rcParams['figure.figsize'] = (8.3, 5.8)  # A5 paper. Friggin inches...
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 8
plt.rcParams['figure.autolayout'] = True
