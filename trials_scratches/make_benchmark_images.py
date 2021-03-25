from thesis_lib import testdata_generators
from thesis_lib.scopesim_helper import download
import tempfile
import os

old_dir = os.path.abspath(os.getcwd())

# download scopesim stuff to temporary location
with tempfile.TemporaryDirectory() as dir:
    os.chdir(dir)
    download(ask=False)

    for fname, recipe in testdata_generators.benchmark_images.items():
        testdata_generators.read_or_generate_image(recipe, fname, os.path.join(old_dir, 'test_images'))
