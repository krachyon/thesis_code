import multiprocessing
import os
import tempfile
from typing import Callable, Tuple, Optional

import anisocado
from anisocado import AnalyticalScaoPsf
from photutils import FittableImageModel
from astropy.modeling.functional_models import Gaussian2D
from image_registration.fft_tools import upsample_image

import numpy as np
import scopesim

from .config import Config
from .util import work_in, centroid, center_of_image
import astropy.units as u

# globals
# TODO get these from scopesim?
pixel_count = 1024 * u.pixel
pixel_scale = 0.004 * u.arcsec/u.pixel

max_pixel_coord = pixel_count - 1 * u.pixel #  size 1024 to max index 1023

filter_name = 'MICADO/filters/TC_filter_K-cont.dat'

# generators should be able to run in parallel but scopesim tends to lock up on the initialization
scopesim_lock = multiprocessing.Lock()


def to_pixel_scale(as_coord):
    """
    convert position of objects from arcseconds to pixel coordinates
    Numpy/photutils center convention
    """
    if not isinstance(as_coord, u.Quantity):
        as_coord *= u.arcsec

    shifted_pixel_coord = as_coord / pixel_scale
    pixel = shifted_pixel_coord + max_pixel_coord / 2 - 0.5 * u.pixel
    return pixel.value


def pixel_to_mas(px_coord):
    """
    convert position of objects from pixel coordinates to arcseconds
    Numpy/photutils center convention
    """
    if not isinstance(px_coord, u.Quantity):
        px_coord *= u.pixel

    # shift bounds (0,1023) to (-511.5,511.5)
    coord_shifted = px_coord - max_pixel_coord / 2 + 0.5 * u.pixel
    mas = coord_shifted * pixel_scale
    return mas.value


# TODO take care of correcting centroid shift
# noinspection PyPep8Naming
def make_psf(psf_wavelength: float = 2.15,
             shift: Tuple[int] = (0, 14), N: int = 512,
             transform: Callable[[np.ndarray], np.ndarray] = lambda x: x) -> scopesim.effects.Effect:
    """
    create a psf effect for scopesim to be as close as possible to how an anisocado PSF is used in simcado
    :param psf_wavelength:
    :param shift:
    :param N: ? Size of kernel?
    :param transform: function to apply to the psf array
    :return: effect object you can plug into OpticalTrain
    """
    hdus = anisocado.misc.make_simcado_psf_file(
        [shift], [psf_wavelength], pixelSize=pixel_scale.value, N=N)
    image = hdus[2]
    image.data = np.squeeze(image.data)  # remove leading dimension, we're only looking at a single picture, not a stack

    # re-sample to shift center
    actual_center = np.array(centroid(image.data))
    expected_center = np.array(center_of_image(image.data))
    shift = expected_center - actual_center
    resampled = upsample_image(image.data, xshift=shift[0], yshift=shift[1]).real
    image.data = resampled
    image.data = transform(image.data)


    filename = tempfile.NamedTemporaryFile('w', suffix='.fits').name
    image.writeto(filename)

    # noinspection PyTypeChecker
    tmp_psf = anisocado.AnalyticalScaoPsf(N=N, wavelength=psf_wavelength)
    strehl = tmp_psf.strehl_ratio

    # Todo: passing a filename that does not end in .fits causes a weird parsing error
    return scopesim.effects.FieldConstantPSF(
        name=Config.instance().psf_name,
        filename=filename,
        wavelength=psf_wavelength,
        psf_side_length=N,
        strehl_ratio=strehl, )
    # convolve_mode=''


# TODO take care of pixel shift, best by setting origin according to computed centroid
class AnisocadoModel(FittableImageModel):
    def __repr__(self):
        return super().__repr__() + f' oversampling: {self.oversampling}'

    def __str__(self):
        return super().__str__() + f' oversampling: {self.oversampling}'

    @property
    # TODO base this on a cutoff for included flux or something...
    def bounding_box(self):
        return ((self.y_0-200, self.y_0+200), (self.x_0-200, self.x_0+200))


def make_anisocado_model(oversampling=2, degree=5, seed=0, offaxis=(0, 14), lowpass=0):
    img = AnalyticalScaoPsf(pixelSize=0.004/oversampling, N=400*oversampling+1, seed=seed).shift_off_axis(*offaxis)
    if lowpass != 0:
        y, x = np.indices(img.shape)
        img = img * Gaussian2D(x_stddev=lowpass, y_stddev=lowpass)(x-x.mean(), y-y.mean())
        img /= np.sum(img)

    origin = centroid(img)
    return AnisocadoModel(img, oversampling=oversampling, degree=degree, origin=origin)


def make_gauss_model(σ):
    data = Gaussian2D(x_stddev=σ, y_stddev=σ)(*np.mgrid[-100:100:400j, -100:100:400j])
    return FittableImageModel(data, oversampling=2, degree=5)


def setup_optical_train(psf_effect: Optional[scopesim.effects.Effect] = None,
                        custom_subpixel_psf: Optional[Callable] = None) -> scopesim.OpticalTrain:
    """
    Create a Micado optical train with custom PSF
    :return: OpticalTrain object
    """

    assert not (psf_effect and custom_subpixel_psf), 'PSF effect can only be applied if custom_subpixel_psf is None'

    # TODO Multiprocessing sometimes seems to cause some issues in scopesim, probably due to shared connection object
    # #  File "ScopeSim/scopesim/effects/ter_curves.py", line 247, in query_server
    # #     tbl.columns[i].name = colname
    # #  UnboundLocalError: local variable 'tbl' referenced before assignment
    # mutexing this line seems to solve it...
    with scopesim_lock:
        micado = scopesim.OpticalTrain('MICADO')

    if custom_subpixel_psf:
        micado.cmds["!SIM.sub_pixel.flag"] = "psf_eval"
        scopesim.rc.__currsys__['!SIM.sub_pixel.psf'] = custom_subpixel_psf

    else:
        micado.cmds["!SIM.sub_pixel.flag"] = True
        # the previous psf had that optical element so put it in the same spot.
        # Todo This way of looking up the index is pretty stupid. Is there a better way?
        element_idx = [element.meta['name'] for element in micado.optics_manager.optical_elements].index('default_ro')
        if not psf_effect:
            psf_effect = make_psf()

        micado.optics_manager.add_effect(psf_effect, ext=element_idx)


    # disable old psf
    # TODO - why is there no remove_effect with a similar interface?
    #  Why do I need to go through a dictionary attached to a different class?
    # TODO - would be nice if Effect Objects where frozen, e.g. with the dataclass decorator. Used ".included" first and
    # TODO   was annoyed that it wasn't working...
    micado['relay_psf'].include = False
    micado['micado_ncpas_psf'].include = False

    # TODO Apparently atmospheric dispersion is messed up. Ignore both dispersion and correction for now
    micado['armazones_atmo_dispersion'].include = False
    micado['micado_adc_3D_shift'].include = False

    return micado


def download(to_directory=Config.instance().scopesim_working_dir) -> None:
    """
    get scopesim files if not present in current directory
    :return: No
    """
    if not os.path.exists(to_directory):
        os.makedirs(to_directory)
    with work_in(to_directory):
        if not os.path.exists('./MICADO'):
            scopesim.download_package(["locations/Armazones",
                                       "telescopes/ELT",
                                       "instruments/MICADO"])
