# This is needed as the default method 'fork' sometimes borks the ipython kernel...
try:
    import bottleneck
    raise ImportError('bottleneck is installed, which is numerically inaccurate. '
                      'see https://github.com/pydata/bottleneck/issues/379 and '
                      'https://github.com/astropy/astropy/issues/11492 for details')
except ModuleNotFoundError:
    pass


from . import astrometry
from . import config
#from . import experimental
from . import scopesim_helper
from . import standalone_analysis
from . import testdata
from . import util

import multiprocess
import warnings
if multiprocess.get_start_method() == 'fork':
    warnings.warn('multiprocess startup method set to fork.'
                  'If you run this in a notebook, fork might break on you')

