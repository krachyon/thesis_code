import numpy as np
from astropy.convolution import Gaussian2DKernel, AiryDisk2DKernel

from thesis_lib.scopesim_helper import make_anisocado_model
from thesis_lib.testdata_helpers import lowpass, make_anisocado_kernel
from thesis_lib.testdata_recipes import gaussian_cluster, scopesim_cluster, convolved_grid, scopesim_grid, empty_image, \
    single_star_image

kernel_size = 201  # should be enough for getting reasonable results
misc_images = {
    'gauss_cluster_N1000': lambda: gaussian_cluster(N=1000),
    'gauss_cluster_N1000_low': lambda: gaussian_cluster(N=1000, psf_transform=lowpass()),
    'scopesim_cluster': lambda: scopesim_cluster(),
    'gauss_grid_16_sigma1_perturb_0': lambda: convolved_grid(N1d=16),
    'gauss_grid_16_sigma1_perturb_2': lambda: convolved_grid(N1d=16, perturbation=2.),
    'gauss_grid_16_sigma5_perturb_2':
        lambda: convolved_grid(N1d=16, perturbation=2.,
                               kernel=Gaussian2DKernel(x_stddev=5, x_size=kernel_size, y_size=kernel_size)),
    'airy_grid_16_radius2_perturb_0':
        lambda: convolved_grid(N1d=16, perturbation=0.,
                               kernel=AiryDisk2DKernel(radius=2, x_size=kernel_size, y_size=kernel_size)),
    'airy_grid_16_radius2_perturb_2':
        lambda: convolved_grid(N1d=16, perturbation=2.,
                               kernel=AiryDisk2DKernel(radius=2, x_size=kernel_size, y_size=kernel_size)),
    'airy_grid_16_radius5_perturb_2':
        lambda: convolved_grid(N1d=16, perturbation=2.,
                               kernel=AiryDisk2DKernel(radius=5, x_size=kernel_size, y_size=kernel_size)),
    'anisocado_grid_16_perturb_0':
        lambda: convolved_grid(N1d=16, perturbation=0.,
                               kernel=make_anisocado_kernel()),
    'anisocado_grid_16_perturb_2':
        lambda: convolved_grid(N1d=16, perturbation=2.,
                               kernel=make_anisocado_kernel()),
    'grid_16_no_convolve_perturb2':
        lambda: convolved_grid(N1d=16, perturbation=2., kernel=None),
    'scopesim_grid_16_perturb0':
        lambda: scopesim_grid(N1d=16, perturbation=0.),
    'scopesim_grid_16_perturb2':
        lambda: scopesim_grid(N1d=16, perturbation=2.),
    'empty_image':
        lambda: empty_image(),
}
benchmark_images = {
    'scopesim_grid_16_perturb2_mag18_24':
        lambda: scopesim_grid(N1d=16, perturbation=2.,
                              magnitude=lambda N: np.random.uniform(18, 24, N)),
    'scopesim_grid_16_perturb2_mag18_24_lowpass':
        lambda: scopesim_grid(N1d=16, perturbation=2.,
                              magnitude=lambda N: np.random.uniform(18, 24, N), psf_transform=lowpass()),
    'scopesim_grid_16_perturb2_lowpass_mag18_24_subpixel':
        lambda: scopesim_grid(N1d=16, perturbation=2.,
                              magnitude=lambda N: np.random.uniform(18, 24, N),
                              custom_subpixel_psf=make_anisocado_model()),
    'scopesim_grid_16_perturb2_lowpass_mag18_24_subpixel_lowpass':
        lambda: scopesim_grid(N1d=16, perturbation=2.,
                              magnitude=lambda N: np.random.uniform(18, 24, N),
                              custom_subpixel_psf=make_anisocado_model(lowpass=5)),
    'gausscluster_N2000_mag22':
        lambda: gaussian_cluster(2000, magnitude=lambda N: np.random.normal(22, 2, N)),
    'gausscluster_N2000_mag22_lowpass':
        lambda: gaussian_cluster(2000, magnitude=lambda N: np.random.normal(22, 2, N), psf_transform=lowpass()),
    'gausscluster_N2000_mag22_subpixel':
        lambda: gaussian_cluster(2000, magnitude=lambda N: np.random.normal(22, 2, N),
                                 custom_subpixel_psf=make_anisocado_model()),
    'gausscluster_N2000_mag22_lowpass_subpixel':
        lambda: gaussian_cluster(2000, magnitude=lambda N: np.random.normal(22, 2, N),
                                 custom_subpixel_psf=make_anisocado_model(lowpass=5)),
}
helpers = {
    'anisocado_psf':
        lambda: make_anisocado_kernel().array,
    'single_star_image':
        lambda: single_star_image()
}
predefined_images = benchmark_images | misc_images

# predefined recipes

# lowpass_images = {
#     'scopesim_grid_30_perturb2_lowpass_mag18-24':
#         lambda: scopesim_grid(N1d=30, perturbation=2., psf_transform=lowpass(),
#                               magnitude=lambda N: np.random.uniform(18, 24, N)),
#     'scopesim_groups_16_perturb_2_lowpass_radius_7':
#         lambda: scopesim_groups(N1d=16, jitter=2., psf_transform=lowpass(), magnitude=lambda N: N * [20],
#                                 group_radius=7, group_size=2),
#     'scopesim_groups_16_perturb_2_lowpass_radius_5':
#         lambda: scopesim_groups(N1d=16, jitter=2., psf_transform=lowpass(), magnitude=lambda N: N * [20],
#                                 group_radius=10, group_size=5),
#     'gausscluster_N2000_lowpass_mag22':
#         lambda: gaussian_cluster(2000, magnitude=lambda N: np.random.normal(22, 2, N), psf_transform=lowpass()),
#     'gausscluster_N4000_lowpass_expmag21':
#         lambda: gaussian_cluster(4000, magnitude=expmag, psf_transform=lowpass())
# }

