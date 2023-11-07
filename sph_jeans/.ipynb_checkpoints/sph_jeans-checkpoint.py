import numpy as np
#import modules.darkmatter as dm


minLM, maxLM, dLM = -6.,5.0,0.1
min3D, max3D, d3D = -5.,4.0,0.1
minPJ, maxPJ, dPJ = -5.,4.0,0.1

#######################3
#       Constants     #3
#######################3
m_pc    = 1/(206265*1.4996e11)         # 1kpc  = 1e3*206265*1.496e11 m
kg_Msol = 1/(1.991e30)                 # 1Msol = 1.991e30 kg
G       = 6.67e-11*1e-6*(m_pc/kg_Msol) #(km^2/s^2)*(kpc/Msol)

#######################3
#       MODELO        #3
#######################3


class model:

    def __init__(self,pars):
        self.I0,self.a1,self.a2,self.s0,self.s1,self.s2,self.alph_1,self.alph_2 = pars

        #par2
        self.r  = np.logspace(-3.,8.0,1000)
        self.nu = np.zeros((self.r).size)

        #par3
        self.ml  = 0.0
        self.Mbh = 0.0

        self.p0  = 0.0  #scale density for dm profile
        self.r0  = 0.0  #scale radius for dm profile

        self.mass_tot = np.zeros((self.r).size)

        #par4
        self.beta = 0.0
        self.r_02 = 10.**np.arange(-2.,6.5,0.1)
        self.s2_r = np.zeros((self.r_02).size)

    ##################################################################################
    def get_pars(self):
        return [self.I0,self.a1,self.a2,self.s0,self.s1,self.s2,self.alph_1,self.alph_2]

    def get_nu(self,x):
        return np.interp(x,self.r,self.nu)

    def get_mass_tot(self,x):
        return np.interp(x,self.r,self.mass_tot)

    def get_s2r(self,x):
        return np.interp(x,self.r_02,self.s2_r)    
    
    #####################################################################################
    # Deprojection of the surface luminosity profile, based on the surface  
    # luminosity profile presented in Van der Marel & Anderson 2010, ApJ, 710.1063..1088
    #  -- equation (32) 
    #
    
    def gen_nu(self):
        I0,a1,a2,s0,s1,s2,alph_1,alph_2 = self.get_pars()
    
        r  = self.r
   
        for k in range(r.size):
    
            R  = r[k] + 10**np.arange(minLM,maxLM,dLM)
            dR = R[1:R.size] - R[0:R.size-1]
    
            aux_is = I0*(R/a1)**(-s0)*(1.+(R/a1)**alph_1)**(-s1/alph_1)*(1.+(R/a2)**alph_2)**(-s2/alph_2)
            aux_00 = (R**2 - r[k]**2)**(-0.5)
            aux_01 = (-s0)*(R/a1)**(-1.)*(1./a1)
            aux_02 = (-s1)*(1.+(R/a1)**(alph_1))**(-1.)*(R/a1)**(alph_1-1.)*(1./a1)
            aux_03 = (-s2)*(1.+(R/a2)**(alph_2))**(-1.)*(R/a2)**(alph_2-1.)*(1./a2)
    
            f     = aux_00*aux_is*(aux_01+aux_02+aux_03)
            f_mid = 0.5*(f[1:f.size] + f[0:f.size-1])
    
            self.nu[k] = (-1./np.pi)*np.sum(f_mid*dR)
    
    ############################################################################
    # Total cumulative mass as used on Aros et al. 2020.  
    # it considers a constant mass-to-light ratio and central IMBH.
    
    def gen_mass(self,ml,Mbh):
        self.ml  = ml
        self.Mbh = Mbh
        
        r  = self.r
        
        aux_m_stars = np.zeros(r.size)

        u = np.linspace(r[:-1],r[1:],100)
        v = ml*self.get_nu(u)*u**2
        
        du =      u[1:,:] - u[:-1,:]
        dv = 0.5*(v[1:,:] + v[:-1,:])

        aux_m_stars[1:] = 4*np.pi*np.sum(du*dv,axis=0)
                
        #aux_m_stars = np.zeros(r.size)
        #
        #for k in range(r.size-1):
        #
        #    u = np.linspace(r[k],r[k+1],100)
        #    v = ml*self.get_nu(u)*u**2
        #
        #    du =      u[1:] - u[:u.size-1]
        #    dv = 0.5*(v[1:] + v[:v.size-1])
        #
        #    aux_m_stars[k+1] = 4*np.pi*np.sum(du*dv)

        self.mass_tot = np.cumsum(aux_m_stars) + Mbh
    
    
    # Modified version: includes parameters for a Dark Matter density profile
    # [dm_density(r,p0,r0)] defined in module dark_matter.
    
    def gen_mass_with_dm(self,ml,Mbh,dm_density,p0,r0):
        self.ml  = ml
        self.Mbh = Mbh

        self.p0 = p0
        self.r0 = r0


        r  = self.r

        aux_m_stars = np.zeros(r.size)
        aux_m_darkm = np.zeros(r.size)
        
        u = np.linspace(r[:-1],r[1:],100)
        v = ml*self.get_nu(u)*u**2
        w = dm_density(u,p0,r0)*u**2
        
        du =      u[1:,:] - u[:-1,:]
        dv = 0.5*(v[1:,:] + v[:-1,:])
        dw = 0.5*(w[1:,:] + w[:-1,:])
        
        aux_m_stars[1:] = 4*np.pi*np.sum(du*dv,axis=0)
        aux_m_darkm[1:] = 4*np.pi*np.sum(du*dw,axis=0)
        
        #for k in range(r.size-1):

        #    u = np.linspace(r[k],r[k+1],100)
        #    v = ml*self.get_nu(u)*u**2
        #    w = dm_density(u,p0,r0)*u**2
        #
        #
        #    du =      u[1:] - u[:u.size-1]
        #    dv = 0.5*(v[1:] + v[:v.size-1])
        #    dw = 0.5*(w[1:] + w[:w.size-1])

        #    aux_m_stars[k+1] = 4*np.pi*np.sum(du*dv)
        #    aux_m_darkm[k+1] = 4*np.pi*np.sum(du*dw)


        self.mass_tot = np.cumsum(aux_m_stars) + np.cumsum(aux_m_darkm) + Mbh


    ############################################################################
    # Solution to the spherically symmetric Jeans Equation, assuming constant
    # velocity anisotropy (beta). It calculates the integral in eq. 4.216 from
    # Binney & Tremaine (2008)
    
    def gen_vdisp(self,beta):
        self.beta = beta
        r  = self.r_02

        ## integration
        x  = np.logspace(np.log10(r),np.log10(r)+max3D,1000)[1:-1,:]
        dx = x[1:,:] - x[:-1,:]

        Ms = self.get_mass_tot(x)
        js = self.get_nu(x)

        ds    = js*x**(2.*beta-2)*G*Ms
        f_mid = 0.5*(ds[1:,:]+ds[:-1,:])*dx       
        
        self.s2_r = np.sum(f_mid,axis=0)*(self.get_nu(r))**(-1)*(r)**(-2*beta)


    ############################################################################
    # 2D projection of 3D models
    # 

    def I_surf(self,x):
        r  = np.logspace(np.log10(x),np.log10(x)+maxPJ,1000)[1:-1,:]
        dr = r[1:,:] - r[:-1,:]
        
        js  = self.get_nu(r)

        #integrando Isurf
        A  = r/(r**2-x**2)**0.5*js         
        dI = 0.5*(A[1:,:]+A[:-1,:])*dr
        
        return 2*np.sum(dI,axis=0)


    def s_los(self,x):
        r  = np.logspace(np.log10(x),np.log10(x)+maxPJ,1000)[1:-1,:]
        dr = r[1:,:] - r[:-1,:]

        s2r = self.get_s2r(r)
        js  = self.get_nu(r)

        #integrando Slos
        A = r/(r**2-x**2)**0.5*js        
        B_los = (1.-self.beta*(x/r)**2)*s2r       
        
        f_los = A*B_los

        dS_los = 0.5*(f_los[1:,:]+f_los[:-1,:])*dr
        dI     = 0.5*(A[1:,:]+A[:-1,:])*dr

        aux_slos = np.sum(dS_los,axis=0)
        aux_I = np.sum(dI,axis=0)
        
        #results   
        S_los = np.sqrt(aux_slos/aux_I)

        return S_los

    def s_pms(self,x):
        r  = np.logspace(np.log10(x),np.log10(x)+maxPJ,1000)[1:-1,:]
        dr = r[1:,:] - r[:-1,:]

        s2r = self.get_s2r(r)
        js  = self.get_nu(r)

        #integrando Slos
        A = r/(r**2-x**2)**0.5*js
        
        B_rad = (1.-self.beta*(1-(x/r)**2))*s2r
        B_tan = (1.-self.beta)*s2r
        
        f_rad = A*B_rad
        f_tan = A*B_tan
        
        dS_rad = 0.5*(f_rad[1:,:]+f_rad[:-1,:])*dr
        dS_tan = 0.5*(f_tan[1:,:]+f_tan[:-1,:])*dr
        dI     = 0.5*(A[1:,:]+A[:-1,:])*dr

        aux_srad = np.sum(dS_rad,axis=0)
        aux_stan = np.sum(dS_tan,axis=0)
        aux_I = np.sum(dI,axis=0)
        
        #results   
        S_rad = np.sqrt(aux_srad/aux_I)
        S_tan = np.sqrt(aux_stan/aux_I)

        return [S_rad,S_tan]

    def s_all(self,x):
        r  = np.logspace(np.log10(x),np.log10(x)+maxPJ,1000)[1:-1,:]
        dr = r[1:,:] - r[:-1,:]

        s2r = self.get_s2r(r)
        js  = self.get_nu(r)

        #integrando Slos
        A = r/(r**2-x**2)**0.5*js
        
        B_los = (1.-self.beta*(x/r)**2)*s2r
        B_rad = (1.-self.beta*(1-(x/r)**2))*s2r
        B_tan = (1.-self.beta)*s2r
        
        f_los = A*B_los
        f_rad = A*B_rad
        f_tan = A*B_tan
        
        dS_los = 0.5*(f_los[1:,:]+f_los[:-1,:])*dr
        dS_rad = 0.5*(f_rad[1:,:]+f_rad[:-1,:])*dr
        dS_tan = 0.5*(f_tan[1:,:]+f_tan[:-1,:])*dr
        dI     = 0.5*(A[1:,:]+A[:-1,:])*dr

        aux_slos = np.sum(dS_los,axis=0)
        aux_srad = np.sum(dS_rad,axis=0)
        aux_stan = np.sum(dS_tan,axis=0)
        aux_I = np.sum(dI,axis=0)
        
        #results   
        S_los = np.sqrt(aux_slos/aux_I)
        S_rad = np.sqrt(aux_srad/aux_I)
        S_tan = np.sqrt(aux_stan/aux_I)

        return [S_los,S_rad,S_tan]