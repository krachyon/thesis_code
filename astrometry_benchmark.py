from photometry import make_epsf_fit
from testdata_generators import read_or_generate, gaussian_cluster
import testdata_generators

from config import Config

from photometry import make_stars_guess, make_epsf_fit, do_photometry_epsf
# show how epsf looks for given image/parameters

from plots_and_sanitycheck import plot_image_with_source_and_measured, plot_input_vs_photometry_positions, save

from scopesim_helper import download
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import util


def photometry_full(filename='gauss_cluster_N1000', config=Config.instance()):
    image, input_table = read_or_generate(filename, config)

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

    plot_filename = os.path.join(config.output_folder, filename+'_photometry_vs_sources')
    plot_image_with_source_and_measured(image, input_table, result_table, output_path=plot_filename)

    if len(result_table) != 0:
        plot_filename = os.path.join(config.output_folder, filename + '_measurement_offset')
        plot_input_vs_photometry_positions(input_table, result_table, output_path=plot_filename)
    else:
        print(f"No sources found for {filename} with {config}")

    plt.figure()
    plt.imshow(epsf.data)
    save(os.path.join(config.output_folder, filename+'_epsf'))

    # TODO can't return star_guesses directly because they're not pickle-able
    return image, input_table, result_table, epsf


if __name__ == '__main__':
    download()
    test_images = testdata_generators.images.keys()
    normal_config = Config.instance()
    gauss_config = Config()
    gauss_config.smoothing = util.make_gauss_kernel()
    gauss_config.output_folder = 'output_files_gaussian_smooth'

    args = [(image, normal_config) for image in test_images] + \
           [(image, gauss_config) for image in test_images]

    if not os.path.exists(normal_config.output_folder):
        os.mkdir(normal_config.output_folder)
    if not os.path.exists(gauss_config.output_folder):
        os.mkdir(gauss_config.output_folder)

    with mp.Pool(mp.cpu_count()) as pool:
        # call photometry_full(*args[0]), photometry_full(*args[1]) ...
        results = list(pool.starmap(photometry_full, args))


    plt.show()
    pass
