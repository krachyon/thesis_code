from os import mkdir
from os.path import exists, join
from typing import Callable, Tuple, Optional

import multiprocess
import numpy as np
from astropy.io.fits import PrimaryHDU
from astropy.table import Table

from .definitions import predefined_images
from ..config import Config
from ..util import getdata_safer, work_in

# make sure concurrent calls with the same filename don't tread on each other's toes.
# only generate/write file once

manager = multiprocess.Manager()
file_locks = manager.dict()


def get_lock(filename_base):
    lock: multiprocess.Lock = file_locks.setdefault(filename_base, manager.Lock())
    return lock


def read_or_generate_image(filename_base: str,
                           config: Config = Config.instance(),
                           recipe: Optional[Callable[[], Tuple[np.ndarray, Table]]] = None):
    """
    For the 'recipe' either generate and write the image+catalogue or read existing output from disk
    :param directory: where to put/read data
    :param filename_base: what the files are called, minus extension
    :param recipe: function generating your image and catalogue
    :return: image, input_catalogue
    """

    if not exists(config.image_folder):
        mkdir(config.image_folder)
    image_name = join(config.image_folder, filename_base + '.fits')
    table_name = join(config.image_folder, filename_base + '.dat')
    lock = get_lock(filename_base)
    with lock:
        if exists(image_name) and exists(table_name):
            img = getdata_safer(image_name)
            table = Table.read(table_name, format='ascii.ecsv')
        else:
            with work_in(config.scopesim_working_dir):
                if not recipe:
                    recipe = predefined_images[filename_base]
                img, table = recipe()
            img = img.astype(np.float64, order='C', copy=False)
            PrimaryHDU(img).writeto(image_name, overwrite=True)
            table.write(table_name, format='ascii.ecsv')

    return img, table


def read_or_generate_helper(filename_base: str,
                            config: Config = Config.instance(),
                            recipe: Optional[Callable[[], np.ndarray]] = None):
    """
    For the 'recipe' either generate and write the image+catalogue or read existing output from disk
    :param directory: where to put/read data
    :param filename_base: what the files are called, minus extension
    :param recipe: function generating your image and catalogue
    :return: image, input_catalogue
    """
    if not exists(config.image_folder):
        mkdir(config.image_folder)
    image_name = join(config.image_folder, filename_base + '.fits')
    lock = get_lock(filename_base)
    with lock:
        if exists(image_name):
            img = getdata_safer(image_name)
        else:
            with work_in(config.scopesim_working_dir):
                img = recipe().astype(np.float64, order='C', copy=False)
            PrimaryHDU(img).writeto(image_name, overwrite=True)

    return img


