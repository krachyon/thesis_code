import scopesim
import anisocado
import tempfile
import numpy as np
from typing import Tuple
import multiprocessing
from typing import Optional
import os
from config import Config

# globals
pixel_scale = 0.004  # TODO get these from scopesim?
pixel_count = 1024
filter_name = 'MICADO/filters/TC_filter_K-cont.dat'

# generators should be able to run in parallel but scopesim tends to lock up on the initialization
scopesim_lock = multiprocessing.Lock()

@np.vectorize
def to_pixel_scale(pos):
    """
    convert position of objects from arcseconds to pixel coordinates
    :param pos:
    :return:
    """
    # TODO I'm pretty sure this is correct like this, see scopesim_placement for trying it out
    #  but the distortion seems to be exactly one pixel towards the edges so that seems fishy
    return pos / pixel_scale + 512


# noinspection PyPep8Naming
def make_psf(psf_wavelength: float = 2.15, shift: Tuple[int] = (0, 14), N: int = 512) -> scopesim.effects.Effect:
    """
    create a psf effect for scopesim to be as close as possible to how an anisocado PSF is used in simcado
    :param psf_wavelength:
    :param shift:
    :param N: ? Size of kernel?
    :return: effect object you can plug into OpticalTrain
    """
    hdus = anisocado.misc.make_simcado_psf_file(
        [shift], [psf_wavelength], pixelSize=pixel_scale, N=N)
    image = hdus[2]
    image.data = np.squeeze(image.data)  # remove leading dimension, we're only looking at a single picture, not a stack
    filename = tempfile.NamedTemporaryFile('w', suffix='.fits').name
    image.writeto(filename)

    # noinspection PyTypeChecker
    tmp_psf = anisocado.AnalyticalScaoPsf(N=N, wavelength=psf_wavelength)
    strehl = tmp_psf.strehl_ratio

    # Todo: passing a filename that does not end in .fits causes a weird parsing error
    return scopesim.effects.FieldConstantPSF(
        name=Config.psf_name,
        filename=filename,
        wavelength=psf_wavelength,
        psf_side_length=N,
        strehl_ratio=strehl, )
    # convolve_mode=''


def setup_optical_train(psf_effect: Optional[scopesim.effects.Effect] = None) -> scopesim.OpticalTrain:
    """
    Create a Micado optical train with custom PSF
    :return: OpticalTrain object
    """
    if not psf_effect:
        psf_effect = make_psf()

    # TODO Multiprocessing sometimes seems to cause some issues in scopesim, probably due to shared connection object
    # #  File "ScopeSim/scopesim/effects/ter_curves.py", line 247, in query_server
    # #     tbl.columns[i].name = colname
    # #  UnboundLocalError: local variable 'tbl' referenced before assignment
    # mutexing this line seems to solve it...
    with scopesim_lock:
        micado = scopesim.OpticalTrain('MICADO')

    # the previous psf had that optical element so put it in the same spot.
    # Todo This way of looking up the index is pretty stupid. Is there a better way?
    element_idx = [element.meta['name'] for element in micado.optics_manager.optical_elements].index('default_ro')

    micado.optics_manager.add_effect(psf_effect, ext=element_idx)

    # disable old psf
    # TODO - why is there no remove_effect with a similar interface? Why do I need to go through a dictionary attached to
    # TODO   a different class?
    # TODO - would be nice if Effect Objects where frozen, e.g. with the dataclass decorator. Used ".included" first and
    # TODO   was annoyed that it wasn't working...
    micado['relay_psf'].include = False
    micado['micado_ncpas_psf'].include = False

    # TODO Apparently atmospheric dispersion is messed up. Ignore both dispersion and correction for now
    micado['armazones_atmo_dispersion'].include = False
    micado['micado_adc_3D_shift'].include = False

    # TODO does this also apply to the custom PSF?
    micado.cmds["!SIM.sub_pixel.flag"] = True

    return micado


def download() -> None:
    """
    get scopesim file if not present
    :return:
    """
    if not os.path.exists('MICADO'):
        # TODO is it really necessary to always throw shit into the current wdir?
        print('''Simcado Data missing. Do you want to download?
        Attention: Will write into current working dir!''')
        choice = input('[y/N] ')
        if choice == 'y' or choice == 'Y':
            scopesim.download_package(["locations/Armazones",
                                       "telescopes/ELT",
                                       "instruments/MICADO"])
        else:
            exit(-1)
