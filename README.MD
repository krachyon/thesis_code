# Usage
Run `astrometry_benchmark.py`, either with the ipython `%run` magic
or just with `python3 /path/to/astrometry_benchmark.py`. This will generate test images and perform
astrometry on them and create a bunch of plots in output_files* directories. Also will download
a bunch of scopesim files to the current directory, so ideally use a empty one so you don't have to clean up

New 'images' can be added in testdata_generators.py by creating a function returning the image and
input table and putting it into the 'images' dictionary with an appopriate filename. Will be automatically
picked up in the photometry_benchmark if you want to run it with all the other stuff

If you want to experiment with different parameters, for all images change/add the config
in `astrometry_benchmark.py` or just run `astrometry_benchmark.photometry_full` on your file/config
interactively for a single image.

The generated `.mplf` files contain pickled matplotlib figures, can be viewed with 
https://github.com/krachyon/matplotlib_viewer like image files or just run something like 
```python
import pickle
import matplotlib.pyplot as plt
with open('foo.mplf', 'rb') as f:
    fig = pickle.load(f)
fig.show()
plt.show()    
```
to open a single one.