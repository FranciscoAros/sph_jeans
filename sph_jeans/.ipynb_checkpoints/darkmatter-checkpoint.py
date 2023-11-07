import numpy as np
from scipy.integrate import quad

###########################
#  Mass Density profiles  #
###########################
# NFW profile
def dm_density_nfw(r,rho_0,r_0):
    return rho_0*(r/r_0)**(-1.)*(1.+r/r_0)**(-2.)

# MOORE profile
def dm_density_moore(r,rho_0,r_0):
    return rho_0*(r/r_0)**(-1.5)*(1.+r/r_0)**(-1.5)

# BURKERT profile
def dm_density_burkert(r,rho_0,r_0):
    return rho_0*(1.+r/r_0)**(-1.)*(1.+(r/r_0)**(2.0))**(-1.)


#####################
#  Cumulative MASS  #
#####################

def MASS(r,dm_density,rho_0,r_0):
    dM = lambda x,rho_0,r_0: 4*np.pi*dm_density(x,rho_0,r_0)*x**(2.)

    M,errM = quad(dM, 0.0, r , args=(rho_0,r_0) )


    return M,errM
