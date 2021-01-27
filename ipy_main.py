from pylab import *
from astrometry_benchmark import *

config = Config()

config.epsfbuilder_iters = 2
#image, input_table = read_or_generate('airy_grid_16_radius1_perturb_0')
image, input_table = read_or_generate('gauss_cluster_N1000')


star_guesses = make_stars_guess(image,
                                threshold_factor=config.threshold_factor,
                                clip_sigma=config.clip_sigma,
                                fwhm_guess=config.fwhm_guess,
                                cutout_size=config.cutout_size)

epsf = make_epsf_fit(star_guesses,
                     iters=config.epsfbuilder_iters,
                     oversampling=config.oversampling,
                     smoothing_kernel=config.smoothing)

result_table = do_photometry_epsf(image, epsf)

#plot_input_vs_photometry_positions(input_table, result_table, output_path='foobar')
plot_image_with_source_and_measured(image, input_table, result_table, output_path='foobar')

#image, input_table, result_table, epsf = photometry_full('gauss_cluster_N1000')