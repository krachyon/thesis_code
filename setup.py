from setuptools import setup

setup(
    name='thesis_lib',
    version='0.1',
    packages=['thesis_lib'],
    url='',
    license='GPL v3',
    author='Sebastian Meßlinger',
    author_email='sebastian.messlinger@posteo.de',
    description='Analysis scripts for master thesis',
    entry_points={
        'console_scripts':
            ['astrometry_benchmark=thesis_lib.scripts.astrometry_benchmark:main',
             'generate_all_test_images=thesis_lib.scripts.generate_testimages:main'
             ]
    }
)
