from multiprocess import Pool

from thesis_lib.config import Config
from thesis_lib.testdata.definitions import predefined_images
from thesis_lib.testdata.generators import read_or_generate_image


def main():
    with Pool() as p:
        config = Config.instance()
        p.starmap(read_or_generate_image, [(name, config, recipe) for (name, recipe) in predefined_images.items()])


if __name__ == '__main__':
    main()