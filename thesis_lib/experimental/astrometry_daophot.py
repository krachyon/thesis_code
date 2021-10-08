import os
from collections import namedtuple
from typing import Optional

import astropy.io
import astropy.table
import numpy as np
from astropy.io import fits
from astwro.pydaophot import Daophot, Allstar
from astwro.starlist import StarList, fileformats

from ..testdata import generators
from ..util import magnitude_to_flux

DaophotPhotometryResult = namedtuple('DaophotPhotometryResult',
                                     ('image', 'input_table', 'result_table'))


def run_daophot_photometry(image: np.ndarray,
                           input_table: Optional[astropy.table.Table],
                           daophot_options=(('FITTING RADIUS', '6.0'),)):
    """
    Perform psf photometry with daophot as per the astwro pydaophot tutorial.
    Warning: This will just lock up if an issue occurs, then you need to debug it by looking at
    the "commands" passed to daophot as stdin by the runner and see why it chocked
    :param image: what to analyze
    :param input_table: want to cheat on the initial positions?
    :param daophot_options: pass daophot options as list/tuple of key-val tuples
    :return: PhotometryResult, but only input_table, image and result table are filled
    """
    dp = Daophot(options=daophot_options)
    image_file_name = os.path.join(dp.dir, 'input.fits')
    hdu = fits.PrimaryHDU(image)
    hdu.data = hdu.data.astype(np.float32)
    hdu.writeto(image_file_name, overwrite=True)
    dp.image = image_file_name
    al = Allstar(dir=dp.dir)

    starlist_file_name = 'i.ap'
    detections_file_name = 'i.coo'

    find_res = dp.FInd(frames_av=1, frames_sum=1, starlist_file=detections_file_name)
    assert find_res.success

    if input_table:
        tab = input_table.copy()
        tab['x'] += np.random.uniform(-0.1, +0.1, size=len(tab['x']))
        tab['y'] += np.random.uniform(-0.1, +0.1, size=len(tab['y']))
        tab.rename_columns(('m',), ('rmag',))
        tab.add_column(np.arange(len(tab)) + 1, name='id')
        tab.add_column(np.ones(len(tab)) * 0.5, name='fsharp')
        tab.add_column(np.ones(len(tab)) * 0.5, name='round')
        tab.add_column(np.ones(len(tab)) * 0.5, name='mround')

        detections = StarList.from_table(tab)
        detections.DAO_type = fileformats.DAO.COO_FILE
        detections.DAO_hdr = find_res.found_starlist.DAO_hdr
        dp.write_starlist(detections, detections_file_name)

    res = dp.PHotometry(apertures=[8], IS=35, OS=50, stars=detections_file_name)
    assert res.success
    stars = res.photometry_starlist
    sorted_stars = stars.sort_values('mag')
    sorted_stars.renumber()

    dp.write_starlist(sorted_stars, starlist_file_name)

    pick_res = dp.PIck(faintest_mag=24, number_of_stars_to_pick=200)

    assert pick_res.success
    dp.set_options('VARIABLE PSF', 2)
    # psf_res = dp.PSf()
    psf_res = dp.PSf()
    assert psf_res.success
    alls_res = al.ALlstar(image_file=image_file_name, stars=psf_res.nei_file, subtracted_image_file='is.fits')

    sub_res = dp.SUbstar(subtract=alls_res.profile_photometry_file, leave_in=pick_res.picked_stars_file)

    for i in range(3):
        print("Iteration {}: Allstar chi: {}".format(i, alls_res.als_stars.chi.mean()))
        dp.image = os.path.join(dp.dir, 'is.fits')
        psf_res = dp.PSf()
        assert psf_res.success
        alls_res = al.ALlstar(image_file=image_file_name, stars=psf_res.nei_file, subtracted_image_file='is.fits')
        dp.image = image_file_name
        dp.SUbstar(subtract='i.als', leave_in='i.lst')
    print("Final:       Allstar chi: {}".format(alls_res.als_stars.chi.mean()))

    dp.image = os.path.join(dp.dir, 'is.fits')
    psf_res = dp.PSf()
    # how to point this at more stars?
    alls_res = al.ALlstar(image_file=image_file_name, stars='i.ap', subtracted_image_file='is.fits')
    assert alls_res.success

    result_table = astropy.table.Table.from_pandas(alls_res.als_stars)
    result_table.rename_columns(('x', 'y', 'mag'), ('x_fit', 'y_fit', 'm'))
    result_table['x_fit'] -= 1
    result_table['y_fit'] -= 1
    result_table['flux_fit'] = magnitude_to_flux(result_table['m'])
    return DaophotPhotometryResult(image, input_table, result_table=result_table)


if __name__ == '__main__':
    img, table = generators.read_or_generate_image('scopesim_grid_16_perturb2_mag18_24_lowpass')

    res = run_daophot_photometry(img, table)
