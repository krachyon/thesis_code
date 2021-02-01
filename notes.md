# EPSF process

- The fitter overlays the images by finding the closest pixel in the oversampled coordinate system
  for each pixel in a star cutout image. For oversampling 4 this results in the 0.25 pixel "snapping"
  mentioned in the 2000 paper. All the cutouts just get medianed together. 
  If there are any gaps, i.e. pixels in the oversampled grid that are 
  not filled by any of the cutouts, they're filled at the end with a polynomial interpolation.
  Not sure what paper that came from.
  

- The quartic and quadratic smoothing kernel is the result of fitting a degree 2/4 polynomial to 
  an 5x5 array of zeros with a single one in the middle. This is apparently equivalent to a polnomial
  fit for every pixel (https://zipcpu.com/dsp/2018/01/16/interpolation-is-convolution.html)
  
- The smoothing isn't really a fixed thing, different papers use different approaches, or multistages
  e.g. first using a 3x3 boxcar smooth and then the quartic.


# Technical

- The indexing of compound models in astropy is a bit inconsistent depending on how you combine models
  This currently is an issue in photutils as they rely on compound models for fitting star groups.
  As you can't really know easily if a parameters will be called e.g. `x_0_0, x_0_1, x_1_0` or 
  `x_0 x_1 x_2` you have to guess and if you don't guess right you get index errors.