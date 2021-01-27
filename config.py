# Global values and names
import dataclasses
import matplotlib.pyplot as plt


class ClassRepr(type):
    """
    Use this as a metaclass to make a class (and not just an instance of it) print its contents
    """
    def __new__(mcs, *args, **kwargs):
        return super().__new__(mcs, *args, **kwargs)

    def __repr__(cls):
        items = [item for item in cls.__dict__.items() if not item[0].startswith('__')]
        item_string = ', '.join([f'{item[0]} = {item[1]}' for item in items])
        return f'{cls.__name__}({item_string})'

    def __str__(cls):
        return repr(cls)


plt.rcParams['figure.figsize'] = (8.3, 5.8)  # A4 paper
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 9
plt.rcParams['figure.autolayout'] = True


@dataclasses.dataclass(init=True, repr=True, eq=False, order=False)
class Config(metaclass=ClassRepr):
    _instance = None

    @classmethod
    def instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance

    # names
    psf_name: str = 'anisocado_psf'
    output_folder: str = 'output_files'

    # magic parameters for Starfinder
    clip_sigma: float = 3.0             # sigma_clipping to apply for star guessing
    threshold_factor: float = 3.        # how many stds brighter than image for star to be detected?
    fwhm_guess: float = 2.5             # estimate of PSF fwhm
    separation_factor: float = 1.       # How far do stars need to be apart to be considered?

    # magic parameters for EPSFBuilder
    cutout_size: int = 50  # TODO PSF is pretty huge, right?
    oversampling: int = 4
    epsfbuilder_iters: int = 5
    smoothing = 'quartic'

    # photometry
    photometry_iterations: int = 3


