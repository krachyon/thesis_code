import photutils
from multiprocessing.reduction import ForkingPickler

stars = photutils.psf.EPSFStars([1])

foo = ForkingPickler.loads(ForkingPickler.dumps(stars))

