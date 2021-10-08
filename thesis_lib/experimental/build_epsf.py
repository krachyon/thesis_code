import numpy as np
from image_registration.fft_tools import upsample_image

import photutils
from ..config import Config


def make_epsf_combine(stars: photutils.psf.EPSFStars,
                      oversampling: int = Config.instance().oversampling) -> photutils.psf.EPSFModel:
    """
    Alternative way of deriving an EPSF. Use median after resampling/scaling to just overlay images
    :param stars: candidate stars as EPSFStars
    :param oversampling: How much to scale
    :return: epsf model
    """
    # TODO to make this more useful
    #  - maybe normalize image before combination? Now median just picks typical
    #    value so we're restricted to most common stars
    #  - add iterations where the star positions are re-determined with the epsf and overlaying happens again
    #  BUG: This will not always yield a centered array for the EPSF

    avg_center = np.mean([np.array(st.cutout_center) for st in stars], axis=0)
    output_shape = np.array(stars[0].data.shape) * oversampling // 2 * 2 + 1
    scaling = output_shape / np.array(stars[0].data.shape)

    combined = np.median([upsample_image(star.data / star.data.max(),
                                         upsample_factor=scaling[0],
                                         output_size=output_shape,
                                         xshift=star.cutout_center[0] - avg_center[0],
                                         yshift=star.cutout_center[1] - avg_center[1]
                                         ).real
                          for star in stars], axis=0)

    combined -= np.min(combined)
    combined /= np.max(combined)

    origin = np.array(combined.shape) / 2 / oversampling
    # TODO What we return here needs to actually use the image in it's __call__ operator to work as a model
    # type: ignore
    return photutils.psf.EPSFModel(combined, flux=None,
                                   origin=origin,
                                   oversampling=oversampling,
                                   normalize=False)
