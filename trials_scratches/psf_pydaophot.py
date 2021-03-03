from astwro.sampledata import fits_image
from astwro.pydaophot import Daophot, Allstar
from astwro.starlist.ds9 import write_ds9_regions
import shutil
import os

frame = fits_image()
dp = Daophot(image=frame)
al = Allstar(dir=dp.dir)
res = dp.FInd(frames_av=1, frames_sum=1)
res = dp.PHotometry(apertures=[8], IS=35, OS=50)
stars = res.photometry_starlist
sorted_stars = stars.sort_values('mag')
sorted_stars.renumber()
dp.write_starlist(sorted_stars, 'i.ap')
pick_res = dp.PIck(faintest_mag=20, number_of_stars_to_pick=40)
dp.set_options('VARIABLE PSF', 2)
psf_res = dp.PSf()
alls_res = al.ALlstar(image_file=frame, stars=psf_res.nei_file, subtracted_image_file='is.fits')

sub_res = dp.SUbstar(subtract=alls_res.profile_photometry_file, leave_in=pick_res.picked_stars_file)

for i in range(3):
    print("Iteration {}: Allstar chi: {}".format(i, alls_res.als_stars.chi.mean()))
    dp.image = os.path.join(dp.dir, 'is.fits')
    psf_res = dp.PSf()
    alls_res = al.ALlstar(image_file=frame, stars=psf_res.nei_file, subtracted_image_file='is.fits')
    dp.image = frame
    dp.SUbstar(subtract='i.als', leave_in='i.lst')
print("Final:       Allstar chi: {}".format(alls_res.als_stars.chi.mean()))

dp.image = os.path.join(dp.dir, 'is.fits')
psf_res=dp.PSf()
# how to point this at more stars?
alls_res = al.ALlstar(image_file=frame, stars=psf_res.nei_file, subtracted_image_file='is.fits')
shutil.copy(os.path.join(dp.dir,'i.als'), 'i.als')
shutil.copy(os.path.join(dp.dir,'is.fits'), 'is.fits')
shutil.copy(os.path.join(dp.dir,'i.psf'), 'i.psf')
shutil.copy(frame, 'i.fits')
