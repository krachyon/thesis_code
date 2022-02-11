from thesis_lib.testdata.generators import read_or_generate_image
from thesis_lib.testdata.definitions import benchmark_images

for key in benchmark_images:
    read_or_generate_image(key)