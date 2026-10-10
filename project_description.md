
We are developing a script for fully automated georegistration of PlanetScope imagery using OpenCV feature-based methods. 

PlanetScope imagery has relatively fine spatial resolution (3 m) but radiometric quality issues, mediocre cloud masks, and uncertain geolocation. 

I have uploaded a working pair of scripts for registering a different pair of datasets into the working directory - these were designed to register Landsat and EMIT.

We are creating a new script which will leverage the same basic logic, but will be updated to use spectral information from PlanetScope. Some caveats exist in this rewrite:

1. The old script assumed the images were cropped to cover the same area. The new data will not necessarily have exactly the same extents. The new script should check for overlap and update logic (e.g. pixel indexing) to allow for the possibility that extents do not match between source and target images. 

2. The old script used three normalized difference band ratios for estimating correspondences. PlanetScope does not include a SWIR band so some of these indices are not possible to compute. Instead, we will use every combination of normalized difference band ratio indices (e.g. Red vs. Green, Blue vs. NIR, etc.) among the blue, green, red, and NIR bands in each image. 

3. Some PlanetScope imagery has 8 bands (SuperDove) and some has only 4 bands (Dove). The script should verify which image type is loaded and make sure it is processing the correct bands in each, focusing only on the four bands which are common to both (blue, green, red, and NIR). 

Later, we will add utilities to help deal with snow, clouds, cloud shadows, and changes in sun angle, which can cause problems for PlanetScope imagery in particular. 