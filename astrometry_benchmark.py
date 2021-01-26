from photometry import make_epsf_fit
from testdata_generators import read_or_generate, gaussian_cluster
import testdata_generators

from config import Config

from photometry import make_stars_guess, make_epsf_fit, do_photometry_epsf
# show how epsf looks for given image/parameters

from plots_and_sanitycheck import plot_image_with_source_and_measured, plot_input_vs_photometry_positions

from scopesim_helper import download
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import util


def photometry_full(filename='gauss_cluster_N1000'):


    image, input_table = read_or_generate(filename)

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

    plot_filename = os.path.join(Config.output_folder, filename+'_photometry_vs_sources.pdf')
    plot_image_with_source_and_measured(image, input_table, result_table, output_path=plot_filename)

    if len(result_table) != 0:
        plot_filename = os.path.join(Config.output_folder, filename + '_measurement_offset.pdf')
        plot_input_vs_photometry_positions(input_table, result_table, output_path=plot_filename)
    else:
        print(f"No sources found for {filename} with {Config}")

    plt.figure()
    plt.imshow(epsf.data)
    plt.savefig(os.path.join(Config.output_folder, filename+'_epsf.pdf'))

    # TODO can't return star_guesses directly because they're not pickle-able
    return image, input_table, result_table, epsf


if __name__ == '__main__':
    download()
    test_images = testdata_generators.images.keys()

    if not os.path.exists(Config.output_folder):
        os.mkdir(Config.output_folder)
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(pool.map(photometry_full, test_images))

    Config.output_folder = 'output_files_gaussian_smooth'
    Config.smoothing = util.make_gauss_kernel()
    if not os.path.exists(Config.output_folder):
        os.mkdir(Config.output_folder)

    with mp.Pool(mp.cpu_count()) as pool:
        results_gauss = list(pool.map(photometry_full, test_images))


    plt.show()
    pass
