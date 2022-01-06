from os import mkdir
from os.path import exists, join, abspath
from typing import Callable, Tuple, Optional
import multiprocess
import filelock
import hashlib
import tempfile
from pathlib import Path

import numpy as np
from astropy.io.fits import PrimaryHDU
from astropy.table import Table

from .definitions import predefined_images
from ..config import Config
from ..util import getdata_safer, work_in

# make sure concurrent calls with the same filename don't tread on each other's toes.
# only generate/write file once


def get_lock(file_path: str) -> filelock.FileLock:
    tmp = Path(tempfile.gettempdir())/'thesis_lib_locks'
    tmp.mkdir(exist_ok=True)
    file_name = Path(file_path).name
    file_base = str(Path(file_path).parent)
    lockname = hashlib.md5(file_base.encode()).hexdigest() + file_name
    lock = filelock.FileLock(tmp/lockname)
    return lock


def read_or_generate_image(filename_base: str,
                           config: Config = Config.instance(),
                           recipe: Optional[Callable[[], Tuple[np.ndarray, Table]]] = None,
                           force_generate = False):
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
    lock = get_lock(abspath(image_name))
    with lock:
        if (exists(image_name) and exists(table_name)) and not force_generate:
            img = getdata_safer(image_name)
            table = Table.read(table_name, format='ascii.ecsv')
        else:
            with work_in(config.scopesim_working_dir):
                if not recipe:
                    recipe = predefined_images[filename_base]
                img, table = recipe()
            img = img.astype(np.float64, order='C', copy=False)
            PrimaryHDU(img).writeto(image_name, overwrite=True)
            table.write(table_name, format='ascii.ecsv', overwrite=True)

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
    lock = get_lock(abspath(image_name))
    with lock:
        if exists(image_name):
            img = getdata_safer(image_name)
        else:
            with work_in(config.scopesim_working_dir):
                img = recipe().astype(np.float64, order='C', copy=False)
            PrimaryHDU(img).writeto(image_name, overwrite=True)

    return img


