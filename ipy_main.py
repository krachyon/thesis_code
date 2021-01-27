from pylab import *
from astrometry_benchmark import *

image, input_table = read_or_generate('gauss_cluster_N1000')

Config.epsfbuilder_iters = 2

star_guesses = make_stars_guess(image,
                                threshold_factor=Config.threshold_factor,
                                clip_sigma=Config.clip_sigma,
                                fwhm_guess=Config.fwhm_guess,
                                cutout_size=Config.cutout_size)

epsf = make_epsf_fit(star_guesses,
                     iters=Config.epsfbuilder_iters,
                     oversampling=Config.oversampling,
                     smoothing_kernel=Config.smoothing)

result_table = do_photometry_epsf(image, epsf)

#plot_input_vs_photometry_positions(input_table, result_table, output_path='foobar')
plot_image_with_source_and_measured(image, input_table, result_table)

#image, input_table, result_table, epsf = photometry_full('gauss_cluster_N1000')