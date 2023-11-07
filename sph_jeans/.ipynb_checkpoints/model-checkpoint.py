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
        self.r  = 10.**np.arange(-3.,8.0,0.1)
        self.nu = np.zeros((self.r).size)

        #par3
        self.ml  = 0.0
        self.Mbh = 0.0

        self.p0  = 0.0  #scale density for dm profile
        self.r0  = 0.0  #scale radius for dm profile

        self.mass = np.zeros((self.r).size)

        #par4
        self.beta = 0.0
        self.r_02 = 10.**np.arange(-2.,6.5,0.1)
        self.s2_r = np.zeros((self.r_02).size)

    def get_pars(self):
        return [self.I0,self.a1,self.a2,self.s0,self.s1,self.s2,self.alph_1,self.alph_2]

    #@classmethod
    def set_dist(self,dist):
        self.dist = dist

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
    # Modified version: includes parameters for a Dark Matter density profile
    # [dm_density(r,p0,r0)] defined in module darkmatter.
    #

    def gen_mass(self,ml,Mbh,dm_density,p0,r0):
        self.ml  = ml
        self.Mbh = Mbh

        self.p0 = p0
        self.r0 = r0

        I0,a1,a2,s0,s1,s2,alph_1,alph_2 = self.get_pars()

        r  = self.r
        aux_m_stars = np.zeros(r.size)
        aux_m_darkm = np.zeros(r.size)

        for k in range(r.size-1):

            u = np.linspace(r[k],r[k+1],100)
            v = np.zeros(u.size)
            w = np.zeros(u.size)

            for j in range(u.size):
                v[j] = ml*self.get_nu(u[j])*u[j]**2
                w[j] = dm_density(u[j],p0,r0)*u[j]**2

            du =      u[1:] - u[:u.size-1]
            dv = 0.5*(v[1:] + v[:v.size-1])
            dw = 0.5*(w[1:] + w[:w.size-1])

            aux_m_stars[k+1] = 4*np.pi*np.sum(du*dv)
            aux_m_darkm[k+1] = 4*np.pi*np.sum(du*dw)


        self.mass = np.cumsum(aux_m_stars) + np.cumsum(aux_m_darkm) + Mbh

    #
    #
    ############################################################################

    def gen_vdisp(self,beta):
        self.beta = beta
        r  = self.r_02

        for k in range(r.size):
            x  = r[k] + 10**np.arange(min3D,max3D,d3D)
            dx = x[1:x.size] - x[0:x.size-1]

            Ms = np.zeros(x.size)
            js = np.zeros(x.size)

            for j in range(x.size):
                Ms[j] = self.get_mass(x[j])
                js[j] = self.get_nu(x[j])

            dx = x[1:x.size]-x[0:x.size-1]
            ds = js*x**(2.*beta-2)*G*Ms

            f_mid = 0.5*(ds[1:x.size]+ds[0:x.size-1])*dx

            self.s2_r[k] = np.sum(f_mid)*(self.get_nu(r[k]))**(-1)*(r[k])**(-2*beta)



    #@staticmethod
    def get_nu(self,x):
        idx_nu = self.r[self.r<x].size - 1

        x1 = self.r[idx_nu]
        x2 = self.r[idx_nu+1]

        y1 = self.nu[idx_nu]
        y2 = self.nu[idx_nu+1]

        return (y2-y1)/(x2-x1)*(x-x1) + y1

    def get_mass(self,x):
        idx_nu = self.r[self.r<=x].size - 1

        x1 = self.r[idx_nu]
        x2 = self.r[idx_nu+1]

        y1 = self.mass[idx_nu]
        y2 = self.mass[idx_nu+1]

        return (y2-y1)/(x2-x1)*(x-x1) + y1

    def get_s2r(self,x):
        idx_nu = self.r_02[self.r_02<x].size - 1

        x1 = self.r_02[idx_nu]
        x2 = self.r_02[idx_nu+1]

        y1 = self.s2_r[idx_nu]
        y2 = self.s2_r[idx_nu+1]

        return (y2-y1)/(x2-x1)*(x-x1) + y1


    def I_surf(self,x):
        r  = x + 10**np.arange(minPJ,maxPJ,dPJ)
        dr = r[1:] - r[:r.size-1]

        js  = np.zeros(r.size)
        for j in range(r.size):
            js[j]  = self.get_nu(r[j])

        #integrando Slos
        A  = r/(r**2-x**2)**0.5*js
        dI = 0.5*(A[1:]+A[:r.size-1])*dr

        #results
        return 2*np.sum(dI)


    def s_los(self,x):
        r  = x + 10**np.arange(minPJ,maxPJ,dPJ)
        dr = r[1:] - r[:r.size-1]

        s2r = np.zeros(r.size)
        js  = np.zeros(r.size)

        for j in range(r.size):
            s2r[j] = self.get_s2r(r[j])
            js[j]  = self.get_nu(r[j])

        #integrando Slos
        A = r/(r**2-x**2)**0.5*js
        B = (1.-self.beta*(x/r)**2)*s2r
        f = A*B

        dSlos = 0.5*(f[1:]+f[:r.size-1])*dr
        dI    = 0.5*(A[1:]+A[:r.size-1])*dr

        #results
        Slos = np.sqrt(np.sum(dSlos)/np.sum(dI))

        return Slos

    def s_pms(self,x):
        r  = x + 10**np.arange(minPJ,maxPJ,dPJ)
        dr = r[1:] - r[:r.size-1]

        s2r = np.zeros(r.size)
        js  = np.zeros(r.size)

        for j in range(r.size):
            s2r[j] = self.get_s2r(r[j])
            js[j]  = self.get_nu(r[j])

        #integrando Slos
        A = r/(r**2-x**2)**0.5*js

        B_rad = (1.-self.beta*(1-(x/r)**2))*s2r
        B_tan = (1.-self.beta)*s2r

        f_rad = A*B_rad
        f_tan = A*B_tan

        dS_rad = 0.5*(f_rad[1:]+f_rad[:-1])*dr
        dS_tan = 0.5*(f_tan[1:]+f_tan[:-1])*dr

        dI    = 0.5*(A[1:]+A[:-1])*dr

        #results
        S_rad = np.sqrt(np.sum(dS_rad)/np.sum(dI))
        S_tan = np.sqrt(np.sum(dS_tan)/np.sum(dI))


        return [S_rad,S_tan]
