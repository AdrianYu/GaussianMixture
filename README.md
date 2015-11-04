# GaussianMixture
Gaussian mixture model based on EM for any shaped gaussians.

Single C++ header and depend on STL and Eigen3 only.
Support AIC/AICC/BIC to automantically determine the number of gaussians.
Use kmeans++ algorithm to initialize centers.
Support any shaped gaussians (full rank or not, using SVD to compute the pseudo-inverse).

# Compile
Any compiler that supports C++11 standard should be able to compile. For gcc, please specify -std=c++11 in compiling flags. To use OpenMP, please also add -fopenmp. Since I don't use many C++11 features, modifications should not be hard if you would like to convert the code to support lower version of the compilers.

# Usage
A very simple example can be found in Source.cpp.
Test data is in Release\data.
For futher details, please refer to Source.cpp and GaussianMixture.h.

If you have any problems, please don't hesitate to contact me at adrianandyu@gmail.com

# Special thanks
Wen was kind enough to point out the error about free parameters. Thanks!

