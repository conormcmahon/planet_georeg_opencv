# PlanetScope Registration

Script to georegister PlanetScope imagery using OpenCV and Scale Invariant Feature Transform (SIFT) keypoints and features. 

Two options exist for the script - one using a global affine transform, and one using a local transform with thin plate spline RBF. 

In the input spline version, an option exists to downsample keypoints. Keypoints are selectively removed only from image subregions which have more than one keypoint, and retained in regions more sparsely covered. 

This script has been locally tested using proprietary PlanetScope imagery. Will work on uploading a public dataset later on once I find and collate some public Planet data. 
