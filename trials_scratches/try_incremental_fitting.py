from thesis_lib.scopesim_helper import make_anisocado_model
from thesis_lib.testdata.generators import read_or_generate_image
from thesis_lib.testdata.recipes import scopesim_groups
from thesis_lib import config
from thesis_lib.astrometry.wrapper import Session
from thesis_lib.scopesim_helper import download
from astropy.modeling.fitting import TRFLSQFitter
from thesis_lib.astrometry.plots import plot_image_with_source_and_measured, plot_xy_deviation
import matplotlib.pyplot as plt
#download()

epsf = make_anisocado_model()

cfg = config.Config()
cfg.use_catalogue_positions = True
cfg.separation_factor = 20
cfg.photometry_iterations = 1
cfg.perturb_catalogue_guess = 0.01


def recipe():
    return scopesim_groups(N1d=1, border=500, group_size=10, group_radius=10, jitter=12,
                           magnitude=lambda N: [20.5]*N,
                           custom_subpixel_psf=epsf, seed=10)

cfg.fithshape=91
cfg.bounds = {'x_0': (0.2, 0.2), 'y_0': (0.2, 0.2), 'flux_0': (10000, 10000)}

img, tab = read_or_generate_image(f'1x10_group', cfg, recipe)
trialsession = Session(cfg, image=img, input_table=tab)
trialsession.fitter = TRFLSQFitter()
trialsession.epsf = epsf
trialsession.determine_psf_parameters()
trialsession.do_astrometry_mine()

plot_image_with_source_and_measured(img, tab, trialsession.tables.result_table)
plot_xy_deviation(trialsession.tables.result_table)
plt.show()
