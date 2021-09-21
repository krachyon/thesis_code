# This is needed as the default method 'fork' sometimes borks the ipython kernel...
import multiprocess
# try:
#     multiprocess.set_start_method('forkserver')
# except RuntimeError:
#     assert (multiprocess.get_start_method() == 'forkserver'), \
#         'This package only works with the forkserver process startup method'
try:
    import bottleneck
    raise ImportError('bottleneck is installed, which is numerically inaccurate. '
                      'see https://github.com/pydata/bottleneck/issues/379 and '
                      'https://github.com/astropy/astropy/issues/11492 for details')
except ModuleNotFoundError:
    pass


from . import astrometry_benchmark
from . import photometry
from . import astrometry_plots
from . import util
from . import testdata_generators
from . import scopesim_helper
from . import config
from . import photometry_daophot
from . import parameter_tuning
from . import sampling_precision
from . import accuracy_of_epsf_derivation