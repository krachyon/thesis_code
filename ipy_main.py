from pylab import *
from astrometry_benchmark import *
from plots_and_sanitycheck import *
from photometry import *
from util import *
from testdata_generators import *
from photutils.detection import IRAFStarFinder, DAOStarFinder
import multiprocessing
from astropy.modeling.functional_models import Gaussian2D, Ellipse2D
from scipy.signal import fftconvolve

plt.ion()





def foo():
    config = Config()
    config.epsfbuilder_iters = 2
    config.clip_sigma = 2.
    config.threshold_factor = 1.
    config.separation_factor = 2.

    image, input_table = read_or_generate_image('scopesim_cluster', config)
    mean, median, std = sigma_clipped_stats(image, sigma=config.clip_sigma)
    threshold = median + config.threshold_factor * std

    finder = IRAFStarFinder(threshold=threshold, fwhm=config.fwhm_guess, minsep_fwhm=1)
    prelim_stars = make_stars_guess(image, finder)

    prelim_epsf = make_epsf_combine(prelim_stars)
    fwhm_guess = FWHM_estimate(prelim_epsf)

    finder = IRAFStarFinder(threshold=threshold, fwhm=fwhm_guess, minsep_fwhm=1)

    # peaks_tbl = IRAFStarFinder(threshold, fwhm_guess, minsep_fwhm=1)(image)
    # peaks_tbl = finder(image)
    # peaks_tbl.rename_columns(['xcentroid', 'ycentroid'], ['x_fit', 'y_fit'])

    # plot_image_with_source_and_measured(image, input_table, peaks_tbl)

    stars = make_stars_guess(image, cutout_size=51, star_finder=finder)
    epsf = make_epsf_fit(stars, iters=config.epsfbuilder_iters, smoothing_kernel=make_gauss_kernel())

    epsf_mod = photutils.psf.prepare_psf_model(epsf, renormalize_psf=False)

    grouper = DAOGroup(config.separation_factor * fwhm_guess)

    shape = (epsf_mod.psfmodel.shape / epsf_mod.psfmodel.oversampling).astype(np.int64)

    epsf_mod.fwhm = astropy.modeling.Parameter('fwhm', 'this is not the way to add this I think')
    epsf_mod.fwhm.value = fwhm_guess

    finder = IRAFStarFinder(threshold=0.1 * threshold, fwhm=fwhm_guess, minsep_fwhm=0.1)

    init_guesses = finder(image)
    init_guesses.rename_columns(('xcentroid','ycentroid'), ('x_0','y_0'))
    photometry = BasicPSFPhotometry(
        finder=finder,
        group_maker=grouper,
        bkg_estimator=MADStdBackgroundRMS(),
        psf_model=epsf_mod,
        fitter=LevMarLSQFitter(),
        fitshape=shape,
    )

    res = photometry(image, init_guesses=init_guesses)
    init_guesses.rename_columns(('x_0','y_0'),('x','y'))
    plot_input_vs_photometry_positions(init_guesses, res)



#plot_image_with_source_and_measured(image, input_table, peaks_tbl)

#imshow(epsf.data)

#stars = make_stars_guess(image, threshold_factor=0.1, fwhm_guess=fwhm_guess, clip_sigma=0.1, cutout_size=41)
#imshow(concat_star_images(stars))


#res = photometry_full('scopesim_cluster', config)
#plot_image_with_source_and_measured(res.image, res.input_table, res.result_table)

def bar():
    # TODO fails with friggin index error in model again:
    config = Config()
    image, input_table = read_or_generate_image('gauss_cluster_N1000')
    mean, median, std = sigma_clipped_stats(image, sigma=config.clip_sigma)
    star_guesses = make_stars_guess(image, DAOStarFinder(median*3, 4.), cutout_size=51)
    config.epsf_guess = make_epsf_combine(star_guesses)
    config.epsfbuilder_iters = 4

#image, input_table, result_table, epsf, star_guesses = photometry_full('gauss_cluster_N1000', config)
def baz():
    config = Config.instance()

    # throw away border pixels to make psf fit into original image

    psf = read_or_generate_helper('anisocado_psf', config)
    psf = center_cutout(psf, (51, 51))  # cutout center of psf or else it takes forever to fit


    config.output_folder = 'output_cheating_astrometry'
    filename = 'scopesim_grid_16_perturb0'

    image, input_table = read_or_generate_image(filename, config)

    origin = np.array(psf.shape)/2
    # type: ignore
    epsf = photutils.psf.EPSFModel(psf, flux=None, origin=origin, oversampling=1, normalize=False)
    epsf = photutils.psf.prepare_psf_model(epsf, renormalize_psf=False)

    image, input_table = read_or_generate_image(filename, config)

    mean, median, std = sigma_clipped_stats(image, sigma=config.clip_sigma)
    threshold = median + config.threshold_factor * std

    fwhm = FWHM_estimate(epsf.psfmodel)

    finder = DAOStarFinder(threshold=threshold, fwhm=fwhm)

    grouper = DAOGroup(config.separation_factor*fwhm)

    shape = (epsf.psfmodel.shape/epsf.psfmodel.oversampling).astype(np.int64)

    epsf.fwhm = astropy.modeling.Parameter('fwhm', 'this is not the way to add this I think')
    epsf.fwhm.value = fwhm
    bkgrms = MADStdBackgroundRMS()

    photometry = BasicPSFPhotometry(
        finder=finder,
        group_maker=grouper,
        bkg_estimator=bkgrms,
        psf_model=epsf,
        fitter=LevMarLSQFitter(),
        fitshape=shape
    )

    result_table = photometry(image)
    star_guesses = make_stars_guess(image, finder, 51)

    plot_filename = os.path.join(config.output_folder, filename+'_photometry_vs_sources')
    plot_image_with_source_and_measured(image, input_table, result_table, output_path=plot_filename)

    if len(result_table) != 0:
        plot_filename = os.path.join(config.output_folder, filename + '_measurement_offset')
        plot_input_vs_photometry_positions(input_table, result_table, output_path=plot_filename)
    else:
        print(f"No sources found for {filename} with {config}")

    plt.figure()
    plt.imshow(epsf.psfmodel.data)
    save(os.path.join(config.output_folder, filename+'_epsf'), plt.gcf())

    plt.figure()
    plt.imshow(concat_star_images(star_guesses))
    save(os.path.join(config.output_folder, filename+'_star_guesses'), plt.gcf())

    res = PhotometryResult(image, input_table, result_table, epsf, star_guesses)


lowpass_config = Config()
lowpass_config.smoothing = util.make_gauss_kernel()
lowpass_config.output_folder = 'output_files_lowpass'
lowpass_config.use_catalogue_positions = True
lowpass_config.photometry_iterations = 1  # with known positions we know all stars on first iter
if not os.path.exists(lowpass_config.output_folder):
    os.mkdir(lowpass_config.output_folder)

lowpass_args = itertools.product(testdata_generators.lowpass_images.keys(), [lowpass_config])
res = list(starmap(photometry_full, lowpass_args))
