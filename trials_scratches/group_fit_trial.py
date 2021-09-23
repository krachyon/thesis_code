import numpy as np

import thesis_lib.testdata_definitions
from thesis_lib import *
from photutils import extract_stars, EPSFBuilder, BasicPSFPhotometry, DAOGroup, MADStdBackgroundRMS, DAOStarFinder
from astropy.nddata import NDData
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
cutout_size = 17

name_grid = 'scopesim_grid_16_perturb2_mag18_24'
recipe_grid = thesis_lib.testdata_definitions.benchmark_images[name_grid]
img_grid, input_table_grid = testdata_generators.read_or_generate_image(recipe_grid, name_grid)

name_group = 'scopesim_tight_group'
recipe_group = lambda:\
    testdata_generators.scopesim_groups(N1d=1,
                                        jitter=8.,
                                        border=400,
                                        magnitude=lambda N: np.random.normal(19, 1.5, N),
                                        group_size=9,
                                        group_radius=7
                                        )

img, input_table = testdata_generators.read_or_generate_image(recipe_group, name_group)

sigma_clipped_stats(img)
grid_img_no_background = img_grid - np.median(img_grid)


stars = extract_stars(NDData(grid_img_no_background), input_table_grid, size=cutout_size)

epsf, _ = EPSFBuilder(oversampling=2,
                      maxiters=6,
                      progress_bar=True,
                      smoothing_kernel=util.make_gauss_kernel(0.5)).build_epsf(stars)

mean, median, std = sigma_clipped_stats(img)

grouper = DAOGroup(60)
finder = DAOStarFinder(threshold=median-2*std, fwhm=2.)

phot = BasicPSFPhotometry(grouper, MADStdBackgroundRMS(), epsf, cutout_size+2, finder=finder)
phot_nogroup = BasicPSFPhotometry(DAOGroup(0.001), MADStdBackgroundRMS(), epsf, cutout_size+2, finder=finder)

init_guess = input_table.copy()
init_guess.rename_columns(['x','y'],['x_0','y_0'])

result_table = phot(img, init_guess)
result_table_nogroup = phot_nogroup(img, init_guess)

plt.figure()
plt.title('perfect guess')
plt.imshow(img)
plt.plot(result_table['x_fit'], result_table['y_fit'], 'ro', markersize=1., label='group')
plt.plot(result_table_nogroup['x_fit'], result_table_nogroup['y_fit'], 'bo', markersize=1., label='nogroup')
plt.legend()

plt.figure()
plt.title('starfinder')
result_table = phot(img)
result_table_nogroup = phot_nogroup(img)
plt.imshow(img)
plt.plot(result_table['x_fit'], result_table['y_fit'], 'ro', markersize=1., label='group')
plt.plot(result_table_nogroup['x_fit'], result_table_nogroup['y_fit'], 'bo', markersize=1., label='nogroup')
plt.legend()
plt.show()
