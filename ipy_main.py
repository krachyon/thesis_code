from pylab import *
from astrometry_benchmark import *
import multiprocessing

config = Config()
config.epsfbuilder_iters = 2
#image, input_table, result_table, epsf, star_guesses = photometry_full('airy_grid_16_radius5_perturb_2', config)

def star_return(filename='gauss_cluster_N1000', config=Config.instance()):
    image, input_table = read_or_generate(filename, config)

    star_guesses = make_stars_guess(image,
                                    threshold_factor=config.threshold_factor,
                                    clip_sigma=config.clip_sigma,
                                    fwhm_guess=config.fwhm_guess,
                                    cutout_size=config.cutout_size)
    return star_guesses


with multiprocessing.Pool() as p:
    results = p.starmap(star_return, [('airy_grid_16_radius5_perturb_2', config)])


# TODO fails with friggin index error in model again:
#config = Config()
#config.epsfbuilder_iters = 2
#image, input_table, result_table, epsf, star_guesses = photometry_full('gauss_cluster_N1000', config)