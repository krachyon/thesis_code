from pylab import *
import photutils as phot
from astropy.io import fits
import numpy as np
from photometry import FWHM_estimate
import astropy.table as table
import itertools
from astropy.stats import sigma_clipped_stats
import multiprocessing as mp

# TODO forget source selection, just go off of the reference data an check fit quality

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


oversampling = 4
psf_size = 201

psf_offsets = [[0, 0], [200, 0], [400, 0],
               [0, 200], [200, 200], [400, 200],
               [0, 400], [200, 400], [400, 400]]


def get_psf_subframe(data: np.ndarray):
    subframes = [data[:201, :201],
                 data[200:401, :201],
                 data[400:, :201],

                 data[:201, 200:401],
                 data[200:401, 200:401],
                 data[400:, 200:401],

                 data[:201, 400:],
                 data[200:401, 400:],
                 data[400:, 400:]]

    return subframes


def get_img_subframes(data: np.ndarray, segments=3):
    ys = np.linspace(0, data.shape[0]-1, segments+1)
    xs = np.linspace(0, data.shape[1]-1, segments+1)
    subframes = []
    offsets = []
    for (x_start, x_end), (y_start, y_end) in itertools.product(pairwise(xs), pairwise(ys)):

        subframes.append(
            data[int(np.floor(y_start)):int(np.ceil(y_end)),
                 int(np.floor(x_start)):int(np.ceil(x_end))]
        )
        offsets.append([int(np.floor(x_start)), int(np.floor(y_start))])
    return subframes, offsets


def psf_from_image(image: np.ndarray):
    offset = int(psf_size/2)
    origin = [offset/oversampling, offset/oversampling]
    model = phot.psf.EPSFModel(image, flux=1, oversampling=oversampling, origin=origin)
    return phot.prepare_psf_model(model, renormalize_psf=False)


def runme(image, psf, offset):
    # if np.isclose(np.sum(psf.psfmodel.data), 0):
    #     continue
    fwhm = FWHM_estimate(psf.psfmodel)
    mean, median, std = sigma_clipped_stats(image)

    grouper = phot.DAOGroup(3*fwhm)
    # values here are handfudged to get maximum amount of candidates
    # ommitted peakmax = 10_000
    finder = phot.IRAFStarFinder(threshold=median*0.5, fwhm=fwhm*1.5, brightest=100, minsep_fwhm=0.4)
    photometry = phot.IterativelySubtractedPSFPhotometry(
        group_maker=grouper,
        finder=finder,
        bkg_estimator=phot.MMMBackground(),
        aperture_radius=fwhm,
        fitshape=psf_size,
        psf_model=psf, niters=3)

    result = photometry(image)
    result['x_fit'] += offset[0]
    result['y_fit'] += offset[1]

    return result


def do_it(image_name: str, psf_name: str):

    image_data = fits.getdata(image_name)
    #image_data = image_data - phot.Background2D(image_data, (50,50)).background
    image_data[image_data < 0] = np.nan
    mask = np.zeros(image_data.shape, dtype=bool)
    mask[0:512, 0:512] = 1
    image_data[mask] = np.nan

    image_subframes, offsets = get_img_subframes(image_data)
    psf_data   = fits.getdata(psf_name)
    psf_subframes = get_psf_subframe(psf_data)
    psf_models = [psf_from_image(p) for p in psf_subframes]

    with mp.Pool() as p:
        results = p.starmap(runme, zip(image_subframes, psf_models, offsets))

    return image_data, psf_data, table.vstack(results)


if __name__ == '__main__':
    img, psf, res = do_it('test_images_naco/NACO.2018-08-12T00:10:49.488_NGC6441_P13_flt.subtr15.fits',
              'test_images_naco/PSF.NACO.2018-08-12T00:10:49.488_NGC6441_P13_flt.subtr15.clean.fits')

    ref = table.Table.read('test_images_naco/NACO.2018-08-12T00:10:49.488_NGC6441_P13_flt.subtr15.clean.xym', format='ascii')
    imshow(img, cmap='hot')
    plot(ref['XRAW']-1, ref['YRAW']-1, 'go', markersize=3, alpha=1, label='reference analysis')
    plot(res['x_fit'], res['y_fit'], 'b+', markersize=2, alpha=1, label='photutils')
    legend()
    show()


