# This is needed as the default method 'fork' sometimes borks the ipython kernel...
import multiprocess
try:
    multiprocess.set_start_method('fork')
except RuntimeError:
    # TODO this is due to instanciation of mp.Manager in testdata.generators. This is not how it's supposed to go.
    # fork doesn't always work in jupyter
    assert (multiprocess.get_start_method() == 'fork'), \
        'This package only works with the fork process startup method'
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

