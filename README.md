# sph_jeans

Spherical Jeans Equations models from Aros et al. (2020)

-- Please cite Aros et al., 2020. MNRAS, 499, 4646-4665. if you use the methods included here. --

The code "sph_jeans.py" includes the methods to create an object "model()" which given a set of
parameters for the surface luminosity density creates a model for the mass distribution and
velocity dispersion a cluster by solving the Jeans Equations (Jeans, 1922).

The version included here assumes spherical symmetry, constant values for the velocity anisotropy and
the mass-to-light ratio and de-projects the surface luminosity density to make the velocity dispersion
models. The main output of the code is the line-of-sight velocity dispersion as well as the radial
and tangential velocity dispersion as cylindrical coordinates in the sky.

This version is from August 2020. 
