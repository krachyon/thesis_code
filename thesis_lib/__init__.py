# This is needed as the default method 'fork' sometimes borks the ipython kernel...
import multiprocess
try:
    multiprocess.set_start_method('forkserver')
except RuntimeError:
    assert (multiprocess.get_start_method() == 'forkserver'), \
        'This package only works with the forkserver process startup method'

from . import astrometry_benchmark
from . import photometry
from . import plots_and_sanitycheck
from . import util
from . import testdata_generators
from . import scopesim_helper
from . import config