from photometry import make_epsf_fit
from testdata_generators import read_or_generate, gaussian_cluster
import testdata_generators

from config import Config

from photometry import make_stars_guess, make_epsf_fit, do_photometry_epsf
# show how epsf looks for given image/parameters

from plots_and_sanitycheck import plot_inputtable_vs_resulttable

from scopesim_helper import download
import os
import matplotlib.pyplot as plt
import multiprocessing as mp


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
                         smoothing_kernel='quartic')

    result_table = do_photometry_epsf(image, epsf)

    plot_filename = os.path.join(Config.output_folder, filename+'_photometry_vs_sources.pdf')
    plot_inputtable_vs_resulttable(image, input_table, result_table, output_path=plot_filename)

    plt.figure()
    plt.imshow(epsf.data)
    plt.savefig(os.path.join(Config.output_folder, filename+'_epsf.pdf'))

    # TODO can't return star_guesses directly because they're not pickleable
    return image, result_table, epsf



if __name__ == '__main__':
    download()
    test_images = testdata_generators.images.keys()
    photometry_full('airy_grid_17_radius5_perturb_2')
    #with mp.Pool(mp.cpu_count()) as pool:
    #    results = list(map(photometry_full, test_images))

    plt.show()
