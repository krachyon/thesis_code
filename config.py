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
        item_string = ', '.join([" = ".join(item) for item in items])
        return f'{cls.__name__}({item_string})'


plt.rcParams['figure.figsize'] = (8.3, 5.8)  # A5 paper
plt.rcParams['figure.dpi'] = 200

@dataclasses.dataclass(repr=True, eq=False, order=False)
class Config(metaclass=ClassRepr):
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

    # photometry
    photometry_iterations: int = 3


