import numpy as np
import emcee as mc

import sph_jeans.sph_jeans as sph

#################################################################
#
#  Assuming Isotropy only LOS velocities
#
#


#########################################
## ONLY STARS -> fitting for ml and beta

def str_lnprior(theta):
    ml = theta
    
    if 0.0 < ml < 10.0:
        return 0.0    
    else:
        return -np.inf

def str_lnlike(theta, data, M0):
    # data = [R,Slos,e_Slos]
    
    ml = theta
    
    M0.gen_mass(ml,0.0)
    M0.gen_vdisp(0.0)
    
    model_slos = M0.s_los(data[0])
    
    return -0.5*np.sum((data[1]-model_slos)**2/(data[2]**2)) 

def str_lnprob(theta, data, M0):
    lp = str_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + str_lnlike(theta, data, M0)


def fit_jeans_str(lumn_pars, data, init_guess, n_dim = 1, n_walkers = 20, n_steps = 500, progress=False):
    # data = [R,Slos,e_Slos]
    # init_guess = [ml]

    ## Init Jeans models
    mc_M0 = sph.model(lumn_pars)
    mc_M0.gen_nu()

    ## run MCMC
    pos  = [init_guess + 1e-4*np.random.randn(n_dim) for i in range(n_walkers)]
    
    sampler = mc.EnsembleSampler(n_walkers, n_dim, str_lnprob, args=(data,mc_M0))
    sampler.run_mcmc(pos, n_steps, progress=progress)

    samples = sampler.chain
    logLike = sampler.lnprobability

    return samples, logLike 


#########################################
## STARS + IMBH

def imbh_lnprior(theta):
    ml,logMbh = theta
    
    if 0.0 < ml < 10. and -1 < logMbh < 6:
        return 0.0    
    else:
        return -np.inf

def imbh_lnlike(theta, data, M0):
    # data = [R,Slos,e_Slos]
    
    ml,logMbh = theta
    
    M0.gen_mass(ml,10**logMbh)
    M0.gen_vdisp(0.0)
    
    model_slos = M0.s_los(data[0])
    
    return -0.5*np.sum((data[1]-model_slos)**2/(data[2]**2)) 

def imbh_lnprob(theta, data, M0):
    lp = imbh_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + imbh_lnlike(theta, data, M0)


def fit_jeans_imbh(lumn_pars, data, init_guess, n_dim = 2, n_walkers = 20, n_steps = 500, progress=False):
    # data = [R,Slos,e_Slos] 
    # init_guess = [ml, logMbh]
    
    ## Init Jeans models
    mc_M0 = sph.model(lumn_pars)
    mc_M0.gen_nu()

    ## run MCMC
    pos  = [init_guess + 1e-4*np.random.randn(n_dim) for i in range(n_walkers)]
    
    sampler = mc.EnsembleSampler(n_walkers, n_dim, imbh_lnprob, args=(data,mc_M0))
    sampler.run_mcmc(pos, n_steps, progress=progress)

    samples = sampler.chain
    logLike = sampler.lnprobability

    return samples, logLike 



#########################################
## STARS + Dark Mass (no IMBH) + FIX ML (STARS)

def dm_lnprior(theta):
    r0,logp0 = theta
    
    if 0.0 < r0 < 5.0 and -2 < logp0 < 8. :
        return 0.0    
    else:
        return -np.inf

def dm_lnlike(theta, data, M0, ML , dm_density):
    # data = [R,Slos,e_Slos]
    r0,logp0 = theta
    
    M0.gen_mass_with_dm(ML,0.0,dm_density,10**logp0,r0)
    M0.gen_vdisp(0.0)
    
    model_slos = M0.s_los(data[0])
    
    return -0.5*np.sum((data[1]-model_slos)**2/(data[2]**2)) 

def dm_lnprob(theta, data, M0, ML, dm_density):
    lp = dm_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + dm_lnlike(theta, data, M0, ML, dm_density)


def fit_jeans_dm(lumn_pars, data, init_guess, ML, dm_density, n_dim = 2, n_walkers = 20, n_steps = 500, progress=False):
    # data = [R,Slos,e_Slos] 
    # init_guess = [r0,logp0]
    
    ## Init Jeans models
    mc_M0 = sph.model(lumn_pars)
    mc_M0.gen_nu()

    ## run MCMC
    pos  = [init_guess + 1e-4*np.random.randn(n_dim) for i in range(n_walkers)]
    
    sampler = mc.EnsembleSampler(n_walkers, n_dim, dm_lnprob, args=(data,mc_M0,ML,dm_density))
    sampler.run_mcmc(pos, n_steps, progress=progress)

    samples = sampler.chain
    logLike = sampler.lnprobability

    return samples, logLike

