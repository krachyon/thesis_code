from . import config
import numpy as np

from typing import Optional, Union
from astropy.table import Table

from .testdata_generators import read_or_generate_image

class Session:

    def __init__(self, config: config.Config,
                 image: Optional[Union[str, np.ndarray]] = None,
                 input_table: Optional[Table] = None):
        self.config = config

        self.input_table = input_table
        self._image = None
        if image:
            self.image = image

        # TODO actually initialize them
        self.starfinder = None
        self.grouper = None

        self.epsf = None
        self.source_table = None

    @property
    def image(self):
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
            raise TypeError("image needs to be array or string")

    def do_it_all(self) -> PhotometryResult:
        """Fire and forget mode"""
        # equivalent
        self.image = config.image_name
        self.find_stars()
        self.select_epsfstars_auto()
        self.make_epsf()
        # TODO
        # self.cull_detections()
        # self.select_epsfstars_qof()
        self.make_epsf()
        return self.do_astrometry()



