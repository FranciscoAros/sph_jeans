import numpy as np
import emcee as mc

import sph_jeans.sph_jeans as sph

#################################################################
#
#  USING ONLY PROPER MOTIONS
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
    # data = [R,Spmr,e_Spmr,Spmt,e_Spmt]
    
    ml,beta = theta
    
    M0.gen_mass(ml,0.0)
    M0.gen_vdisp(beta)
    
    model_spmr, model_spmt = M0.s_pms(data[0])
    
    aux_01 = -0.5*np.sum((data[1]-model_spmr)**2/(data[2]**2)) 
    aux_02 = -0.5*np.sum((data[3]-model_spmt)**2/(data[4]**2)) 
    
    return aux_01 + aux_02


def str_lnprob(theta, data, M0):
    lp = str_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + str_lnlike(theta, data, M0)


def fit_jeans_str(lumn_pars, data, init_guess, n_dim = 2, n_walkers = 20, n_steps = 500, progress=False):
    # data = [R,Spmr,e_Spmr,Spmt,e_Spmt]
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
    # data = [R,Spmr,e_Spmr,Spmt,e_Spmt]
    
    ml,beta,logMbh = theta
    
    M0.gen_mass(ml,10**logMbh)
    M0.gen_vdisp(beta)
    
    model_spmr, model_spmt = M0.s_pms(data[0])
    
    aux_01 = -0.5*np.sum((data[1]-model_spmr)**2/(data[2]**2)) 
    aux_02 = -0.5*np.sum((data[3]-model_spmt)**2/(data[4]**2)) 
    
    return aux_01 + aux_02

def imbh_lnprob(theta, data, M0):
    lp = imbh_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + imbh_lnlike(theta, data, M0)


def fit_jeans_imbh(lumn_pars, data, init_guess, n_dim = 3, n_walkers = 20, n_steps = 500, progress=False):
    # data = [R,Spmr,e_Spmr,Spmt,e_Spmt]
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
    # data = [R,Spmr,e_Spmr,Spmt,e_Spmt]
    
    ml,beta,r0,logp0 = theta
    
    M0.gen_mass_with_dm(ml,0.0,dm_density,10**logp0,r0)
    M0.gen_vdisp(beta)
    
    model_spmr, model_spmt = M0.s_pms(data[0])
    
    aux_01 = -0.5*np.sum((data[1]-model_spmr)**2/(data[2]**2)) 
    aux_02 = -0.5*np.sum((data[3]-model_spmt)**2/(data[4]**2)) 
    
    return aux_01 + aux_02 

def dm_lnprob(theta, data, M0, dm_density):
    lp = dm_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + dm_lnlike(theta, data, M0, dm_density)


def fit_jeans_dm(lumn_pars, data, init_guess, dm_density, n_dim = 4, n_walkers = 20, n_steps = 500, progress=False):
    # data = [R,Spmr,e_Spmr,Spmt,e_Spmt]
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

