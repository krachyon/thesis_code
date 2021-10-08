import numpy as np
from astropy.nddata import NDData
from astropy.table import Table

from photutils import extract_stars, EPSFStars, EPSFBuilder
from ..standalone_analysis.sampling_precision import gen_image


def reference_image(input_model, fitshape, oversampling):

    xsize, ysize = fitshape

    # add one if odd
    n_y = ((ysize*oversampling)//2 * 2 + 1)*1j
    n_x = ((xsize*oversampling)//2 * 2 + 1)*1j

    y, x = np.mgrid[-ysize/2:ysize/2:n_y, -xsize/2:xsize/2:n_x]
    return input_model(x, y)


def epsf_from_model(input_model,
                    n_images,
                    stars_per_image,
                    fitshape,
                    oversampling=1,
                    σ=0,
                    λ=None,
                    smoothing='quartic',
                    epsf_iters=5,
                    seed=0):

    size = 128*int(np.ceil(np.sqrt(stars_per_image)))
    border = 32
    rng = np.random.default_rng(seed)
    stars = []
    for i in range(n_images):
        img, xy_list = gen_image(input_model, stars_per_image, size, border, 'random', σ, λ, rng)

        stars += list(extract_stars(NDData(img), Table(xy_list, names=['x', 'y']), size=np.array(fitshape)))

    stars = EPSFStars(stars)
    builder = EPSFBuilder(oversampling=oversampling, smoothing_kernel=smoothing, maxiters=epsf_iters)

    epsf, fitted_stars = builder(stars)
    return epsf, reference_image(input_model, fitshape, oversampling)


if __name__ == '__main__':

    from astropy.modeling.functional_models import AiryDisk2D
    import matplotlib.pyplot as plt

    model = AiryDisk2D(radius=3)
    epsf, reference = epsf_from_model(model, 100, 1, (21, 21), 2, σ=2, λ=1e5)

    epsf_norm = epsf.data/epsf.data.max()
    reference_norm = reference/reference.max()

    fig, axs = plt.subplots(1,3)
    axs = axs.ravel()

    im = axs[0].imshow(epsf_norm)
    axs[0].set_title('epsf normalized')
    fig.colorbar(im, ax=axs[0])

    im = axs[1].imshow(reference_norm)
    axs[1].set_title('ref normalized')
    fig.colorbar(im, ax=axs[1])

    # TODO this isn't really the juciy information, we want to know how good the centroid accuracy is
    im = axs[2].imshow(epsf_norm-reference_norm)
    axs[2].set_title('epsf-ref')
    fig.colorbar(im, ax=axs[2])


    plt.show()