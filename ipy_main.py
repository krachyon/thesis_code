
from pylab import *
from astrometry_benchmark import *
import multiprocessing


config = Config()
config.epsfbuilder_iters = 2
res = photometry_full('airy_grid_16_radius5_perturb_2', config)


# TODO fails with friggin index error in model again:
#config = Config()
#config.epsfbuilder_iters = 2
#image, input_table, result_table, epsf, star_guesses = photometry_full('gauss_cluster_N1000', config)