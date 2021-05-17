from pylab import *


def rect(x):
    return (np.abs(x) < 0.5).astype(float)


# shift between pixel center (d=0) to pixel border(d=0.5)

def psf(x, d):
    return d * rect(x - 1) + (1 - d) * rect(x)

def mtf(f, d):
    # FT of psf, but frequency shift might be off...
    return sinc(f) * (d * exp(-2j * pi * f) + 1 - d)




x, d = np.mgrid[-2:2:500j, 0:0.5:200j]
imshow(psf(x, d))
title('psf of pixel grid')
xlabel('pixel phase')

figure()
imshow(np.abs(mtf(x, d)))
title('mtf of pixel grid')
xlabel('pixel phase')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
f, d = np.mgrid[0:1:1000j, 0:0.5:400j]
ax.plot_surface(f, d, np.abs(mtf(f, d)), cmap='Spectral')
xlabel('frequency')
ylabel('pixel phase')
title('MTF of pixel grid')

plt.show()

