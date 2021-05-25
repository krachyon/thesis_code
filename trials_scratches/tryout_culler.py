from thesis_lib import *
from photutils import DAOStarFinder, extract_stars, EPSFBuilder
from photutils.psf.culler import ChiSquareCuller
from astropy.stats import sigma_clipped_stats
from astropy.nddata import NDData
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

scopesim_helper.download()
plt.ion()

name = 'gausscluster_N2000_mag22'
recipe = testdata_generators.benchmark_images[name]

img, input_table = testdata_generators.read_or_generate_image(recipe, name)

mean, median, std = sigma_clipped_stats(img)

finder = DAOStarFinder(threshold=median - 5*std, fwhm=3.5, sigma_radius=2.7)

image_no_background = img - median
all_stars = finder(img)
stars_tbl = all_stars.copy()

stars_tbl.rename_columns(['xcentroid', 'ycentroid'], ['x', 'y'])

stars = extract_stars(NDData(image_no_background), stars_tbl[:300], size=11)


epsf, fitted_stars = EPSFBuilder(oversampling=2,
                                 maxiters=6,
                                 progress_bar=True,
                                 smoothing_kernel=util.make_gauss_kernel(0.45)).build_epsf(stars)

culled = ChiSquareCuller(100, image_no_background, epsf)(all_stars)


all_stars.sort('model_chisquare')
stars_refined = extract_stars(NDData(image_no_background), stars_tbl[:100], size=11)
epsf_refined, _ = EPSFBuilder(oversampling=2,
                                 maxiters=6,
                                 progress_bar=True,
                                 smoothing_kernel=util.make_gauss_kernel(0.45)).build_epsf(stars_refined)

all_stars_refined = all_stars.copy()
all_stars_refined.remove_column('model_chisquare')
culled_refined = ChiSquareCuller(100, image_no_background, epsf_refined)(all_stars_refined)


plt.imshow(img, cmap='Greys_r', norm=LogNorm())
plt.colorbar()
#plt.plot(input_table['x'], input_table['y'], 'o', fillstyle='none', markeredgewidth=0.5, markeredgecolor='red')
chisquares = np.array(all_stars['model_chisquare'])
chisquares[np.isnan(chisquares)] = np.nanmax(chisquares)*1.2
plt.scatter(all_stars['xcentroid'], all_stars['ycentroid'], s=2, c=chisquares, cmap='inferno')
plt.colorbar()


plt.figure()
plt.imshow(img, cmap='Greys_r', norm=LogNorm())
plt.colorbar()
chisquares = np.array(all_stars_refined['model_chisquare'])
chisquares[np.isnan(chisquares)] = np.nanmax(chisquares)*1.2
plt.plot(input_table['x']+0.5, input_table['y']+0.5, 'o', fillstyle='none', markersize=4, markeredgewidth=0.3, markeredgecolor='green')
plt.scatter(all_stars_refined['xcentroid'], all_stars_refined['ycentroid'], s=2, c=chisquares, cmap='inferno')
plt.colorbar()


plt.figure()
plt.imshow(epsf.data, norm=LogNorm())
plt.figure()
plt.imshow(epsf_refined.data, norm=LogNorm())


plt.show()



