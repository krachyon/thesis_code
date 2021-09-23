import thesis_lib.testdata_definitions
from thesis_lib import *
from photutils import DAOStarFinder, extract_stars, EPSFBuilder
from photutils.psf.culler import ChiSquareCuller, CorrelationCuller
from astropy.stats import sigma_clipped_stats
from astropy.nddata import NDData
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from thesis_lib.saturation_model import SaturationModel, read_scopesim_linearity

scopesim_helper.download()

# takeaway:
# - appyling saturation after epsf fitting does not help, as the epsf is kind of adapted to it already
# - inverting before all EPSF photometry stuff seems more usefull
# - χ^2 is bad for sorting candidates, as it picks faint ones. need to compensate for brightness somehow...



#name = 'scopesim_grid_16_perturb2_mag18_24'
name = 'gausscluster_N2000_mag22'
recipe = thesis_lib.testdata_definitions.benchmark_images[name]

img, input_table = testdata_generators.read_or_generate_image(recipe, name)

saturation_model = SaturationModel(read_scopesim_linearity('MICADO/FPA_linearity.dat'))
img = saturation_model.inverse_eval(img)

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

culled = CorrelationCuller(100, image_no_background, epsf)(all_stars)

sorted_stars = all_stars.copy()
sorted_stars.rename_columns(['xcentroid', 'ycentroid'], ['x', 'y'])
sorted_stars['qof'] = sorted_stars['model_chisquare']/sorted_stars['flux']
sorted_stars.sort('qof')

stars_refined = extract_stars(NDData(image_no_background), sorted_stars[:100], size=11)
epsf_refined, _ = EPSFBuilder(oversampling=2,
                                 maxiters=6,
                                 progress_bar=True,
                                 smoothing_kernel=util.make_gauss_kernel(0.45)).build_epsf(stars_refined)

# todo this does not really seem to work as intended. Test fitting of combined model

combined = epsf_refined|saturation_model
combined.data = epsf_refined.data  # goddamnit, there needs to be a way to access these irregardles of compounding
combined.oversampling = epsf_refined.oversampling
combined.xname = 'x_0_0'
combined.yname = 'y_0_0'
combined.fluxname = 'flux_0'

all_stars_refined = all_stars.copy()
all_stars_refined.remove_column('model_chisquare')

culled_refined = CorrelationCuller(100, image_no_background, epsf_refined)(all_stars_refined)
# end todo

plt.imshow(img, cmap='Greys_r', norm=LogNorm())
plt.colorbar()
#plt.plot(input_table['x'], input_table['y'], 'o', fillstyle='none', markeredgewidth=0.5, markeredgecolor='red')
chisquares = np.array(all_stars['model_chisquare'])
chisquares[np.isnan(chisquares)] = np.nanmax(chisquares)*1.2
plt.scatter(all_stars['xcentroid'], all_stars['ycentroid'], s=2, c=chisquares, cmap='inferno')
plt.colorbar()


plt.figure()
plt.imshow(img, cmap='Greys_r', norm=LogNorm())
plt.colorbar(label="pixel count")
chisquares = np.array(all_stars_refined['model_chisquare'])
chisquares[np.isnan(chisquares)] = np.nanmax(chisquares)*1.2
plt.plot(input_table['x']+0.5, input_table['y']+0.5, 'o', fillstyle='none', markersize=7, markeredgewidth=0.6, markeredgecolor='green', label='input positions')
#plt.scatter(all_stars_refined['xcentroid'], all_stars_refined['ycentroid'], s=4, c=np.log(chisquares), cmap='inferno', label='detections with $χ^2$')
plt.scatter(all_stars_refined['xcentroid'], all_stars_refined['ycentroid'], s=4, c=chisquares, cmap='inferno', label='detections with $χ^2$')
plt.legend()
plt.colorbar(label='$χ^2$')
import pickle
with open('culler_performance.mplf', 'wb') as f:
    pickle.dump(plt.gcf(), f)

#plt.figure()
#plt.imshow(epsf.data, norm=LogNorm())
#plt.figure()
#plt.imshow(epsf_refined.data, norm=LogNorm())

plt.show()



