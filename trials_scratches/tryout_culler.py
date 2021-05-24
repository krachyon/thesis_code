from thesis_lib import *
from photutils import DAOStarFinder, extract_stars
from photutils.psf.culler import ChiSquareCuller
from astropy.stats import sigma_clipped_stats
from astropy.nddata import NDData

name = 'gausscluster_N2000_mag22'
recipe = testdata_generators.benchmark_images[name]

img, input_table = testdata_generators.read_or_generate_image(recipe, name)

mean, median, std = sigma_clipped_stats(img)
threshold = median + -5*std
finder = DAOStarFinder(threshold, 4)

image_no_background = img - median
all_stars = finder(img)
stars_tbl = all_stars.copy()

stars_tbl.rename_columns(['xcentroid', 'ycentroid'], ['x', 'y'])

stars = extract_stars(NDData(image_no_background), stars_tbl, size=14)

epsf = photometry.make_epsf_combine(stars, oversampling=2)

all_results = finder(img)
culled = ChiSquareCuller(100, img, epsf)(all_stars)
print(culled)
pass



