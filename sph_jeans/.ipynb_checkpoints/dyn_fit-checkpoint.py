import numpy as np
import emcee as mc

import sph_jeans.sph_jeans as sph
import sph_jeans.dark_matter_models as dm

#################################################################
#
#  USING ONLY LINE-OF-SIGHT VELOCITIES
#
#


#########################################
## ONLY STARS -> fitting for ml and beta

def str_lnprior(theta):
    ml,beta = theta
    
    if 0.5 < ml < 3.5 and -2 < beta < +1:
        return 0.0    
    else:
        return -np.inf

def str_lnlike(theta, data, M0):
    # data = [R,Slos,e_Slos]
    
    ml,beta = theta
    
    M0.gen_mass(ml,0.0)
    M0.gen_vdisp(beta)
    
    model_slos = np.zeros(data[0].size)
    
    for n in range(data[0].size):
        model_slos[n] = M0.s_los(data[0][n])
    
    return -0.5*np.sum((data[1]-model_slos)**2/(data[2]**2)) 

def str_lnprob(theta, data, M0):
    lp = str_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + str_lnlike(theta, data, M0)


def fit_jeans_str(lumn_pars, data, init_guess, n_dim = 2, n_walkers = 20, n_steps = 500, progress=False):
    # data = [R,Slos,e_Slos]
    # init_guess = [ml,beta]

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
    ml,beta,logMbh = theta
    
    if 0.5 < ml < 3.5 and -2 < beta < +1 and -1 < logMbh < 6:
        return 0.0    
    else:
        return -np.inf

def imbh_lnlike(theta, data, M0):
    # data = [R,Slos,e_Slos]
    
    ml,beta,logMbh = theta
    
    M0.gen_mass(ml,10**logMbh)
    M0.gen_vdisp(beta)
    
    model_slos = np.zeros(data[0].size)
    
    for n in range(data[0].size):
        model_slos[n] = M0.s_los(data[0][n])
    
    return -0.5*np.sum((data[1]-model_slos)**2/(data[2]**2)) 

def imbh_lnprob(theta, data, M0):
    lp = imbh_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + imbh_lnlike(theta, data, M0)


def fit_jeans_imbh(lumn_pars, data, init_guess, n_dim = 2, n_walkers = 20, n_steps = 500, progress=False):
    # data = [R,Slos,e_Slos] 
    # init_guess = [ml,beta,logMbh]
    
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
## STARS + DM (no IMBH)

def dm_lnprior(theta):
    ml,beta,r0,logp0 = theta
    
    if 0.5 < ml < 3.5 and -2 < beta < +1 and 0.2 < r0 < 2.2 and -1 < logp0 < 4.5 :
        return 0.0    
    else:
        return -np.inf

def dm_lnlike(theta, data, M0,dm_density):
    # data = [R,Slos,e_Slos]
    
    ml,beta,r0,logp0 = theta
    
    M0.gen_mass_with_dm(ml,0.0,dm_density,10**logp0,r0)
    M0.gen_vdisp(beta)
    
    model_slos = np.zeros(data[0].size)
    
    for n in range(data[0].size):
        model_slos[n] = M0.s_los(data[0][n])
    
    return -0.5*np.sum((data[1]-model_slos)**2/(data[2]**2)) 

def dm_lnprob(theta, data, M0, dm_density):
    lp = dm_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + dm_lnlike(theta, data, M0, dm_density)


def fit_jeans_dm(lumn_pars, data, init_guess, dm_density, n_dim = 4, n_walkers = 20, n_steps = 500, progress=False):
    # data = [R,Slos,e_Slos] 
    # init_guess = [ml,beta,r0,logp0]
    
    ## Init Jeans models
    mc_M0 = sph.model(lumn_pars)
    mc_M0.gen_nu()

    ## run MCMC
    pos  = [init_guess + 1e-4*np.random.randn(n_dim) for i in range(n_walkers)]
    
    sampler = mc.EnsembleSampler(n_walkers, n_dim, dm_lnprob, args=(data,mc_M0, dm_density))
    sampler.run_mcmc(pos, n_steps, progress=progress)

    samples = sampler.chain
    logLike = sampler.lnprobability

    return samples, logLike

