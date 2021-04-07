from pylab import *
from astwro.pydaophot import Daophot, Allstar
import astropy.io.fits
import astropy
import photutils as phot
from astropy.visualization import simple_norm
from astropy.table import Table
import numpy as np

img_path = '../test_images_naco/NACO.2018-08-12T00:10:49.488_NGC6441_P13_flt.subtr15.fits'
#img_path = fits_image()
img = astropy.io.fits.getdata(img_path).astype(np.float64)

sharplo=0.3
sharphi=1.4
roundlo=-1.
roundhi=1.
gain, readnoise = 9., 1.7
thresh_sigma = 8
rel_error = 0.8  # some internal calculation in daophot...
fwhm = 5

dp = Daophot(image=img_path,
             options={'TH':thresh_sigma, 'FW':fwhm,
                      'LS':sharplo, 'LR':roundlo, 'HS':sharphi, 'HR':roundhi,
                      'GA': gain, 'RE':readnoise})
al = Allstar(dir=dp.dir)
find_res = dp.FInd(frames_av=1, frames_sum=1)


daophot_thresh = np.sqrt(find_res.sky/gain+readnoise**2) * thresh_sigma * rel_error

phot_res = phot.DAOStarFinder(fwhm=fwhm, threshold=daophot_thresh,
                              sharplo=sharplo,
                              sharphi=sharphi,
                              roundlo=roundlo,
                              roundhi=roundhi)(img)

ref = Table.read('../test_images_naco/NACO.2018-08-12T00:10:49.488_NGC6441_P13_flt.subtr15.clean.xym', format='ascii')


imshow(img, norm=simple_norm(img, 'log', percent=99.8))
plot(find_res.found_starlist['x']-1,find_res.found_starlist['y']-1, 'ro', markersize=3)
plot(phot_res['xcentroid'], phot_res['ycentroid'], 'b+', markersize=3)
plot(ref['XRAW'] - 1, ref['YRAW'] - 1, 'gx', markersize=3, alpha=1, label='reference analysis')
show()

