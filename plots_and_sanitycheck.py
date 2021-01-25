import numpy as np

def concat_star_images(stars: photutils.psf.EPSFStars) -> np.ndarray:
    assert len(set(star.shape for star in stars)) == 1  # all stars need same shape
    N = int(np.ceil(np.sqrt(len(stars))))
    shape = stars[0].shape
    out = np.zeros(np.array(shape, dtype=int)*N)

    from itertools import product

    for row, col in product(range(N), range(N)):
        if (row+N*col) >= len(stars):
            continue
        xstart = row*shape[0]
        ystart = col*shape[1]

        xend = xstart + shape[0]
        yend = ystart + shape[1]
        i = row+N*col
        out[xstart:xend, ystart:yend] = stars[i].data
    return out


def verify_methods_with_grid(filename='output_files/grid_16.fits'):
    img = fits.open(filename)[0].data

    epsf_fit = make_epsf_fit(img)
    epsf_combine = make_epsf_combine(img)

    table_fit = do_photometry_epsf(epsf_fit, img)
    table_combine = do_photometry_epsf(epsf_combine, img)

    plt.figure()
    plt.title('EPSF from fit')
    plt.imshow(epsf_fit.data+0.01, norm=LogNorm())

    plt.figure()
    plt.title('EPSF from image combination')
    plt.imshow(epsf_combine.data+0.01, norm=LogNorm())

    plt.figure()
    plt.title('EPSF internal fit')
    plt.imshow(img, norm=LogNorm())
    plt.plot(table_fit['x_fit'], table_fit['y_fit'], 'r.', alpha=0.7)

    plt.figure()
    plt.title('EPSF image combine')
    plt.imshow(img, norm=LogNorm())
    plt.plot(table_combine['x_fit'], table_combine['y_fit'], 'r.', alpha=0.7)

    return epsf_fit, epsf_combine, table_fit, table_combine