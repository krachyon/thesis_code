from multiprocess import Pool
from itertools import zip_longest

from thesis_lib.testdata_definitions import predefined_images
from thesis_lib.testdata_generators import read_or_generate_image
from thesis_lib.config import Config


def main():
    with Pool() as p:
        config = Config.instance()
        p.starmap(read_or_generate_image, [(name, config, recipe) for (name, recipe) in predefined_images.items()])


if __name__ == '__main__':
    main()