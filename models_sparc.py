
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
#from scipy.special import betainc, gamma


import scipy.optimize
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.special import *
import galpynamics
import galpynamics.dynamic_component as dc

# -------------------------------------------
#           ***  Dimentions  ***
# -------------------------------------------

# r -- kpc
# v -- km/s
# M -- M_Sun
# rho -- M_Sun / pc^3

# H0 -- km/s/kpc
H0 = 0.073 # H0 = 73 km/s/Mpc

# G_N -- km^2 kpc/ M_Sun / s^2
G_N = 4.306 * 10**(-6)

rho_crit = 1.47741 * 10**(-7) # critical density of the Universe, M_Sun / pc^3 

# v(r) = kGrav * sqrt( (M(r)/4pi) /r )
# kGrav = sqrt[ 4 * pi * G_N * (M_Sun/kg) * (kpc/pc)^3 / (kpc/m)]  / (km/m)
# G_N = 6.67 * 10^-11 m^3 / kg / s^2

kGrav = 232.652 

local_inf = 10**4

kpc_to_km = 3.0856776e16  # kpc to km 


plummer=lambda r,ra: (1+(r/ra)**2)**(-5./2.) #plammer profile with ra half-light radius

xiPlum = lambda R, ra: 3/4*ra*(1+(R/ra)**2.)**(-2.) 

@np.vectorize
def betainc(a,b,t):
    f = lambda u: u**(a-1) * (1 - u)**(b-1) 
    return quad(f, 0, t )[0]

# --------------------------------------------
#        ***  Dark matter models  ***
# --------------------------------------------


# FDM_fixedmass: mass_test is correct, 
# but param_compare fails to reproduce initial delta parameter


class Model:
    
    def __init__(self, **kwards):
        self.ndim = 2
        self.parameters = ['$v_{200}$', '$c_{200}$']

    def initial(self, r_inf, v_inf):
        v200_init = v_inf
        c200_init = 0 
        return [v200_init, c200_init]  

    def bounds(self):
        return [[10, 500], [0.0, 1000]] #we use this values as lowe and upper bounds in flat priors

    def priors(self):
        return ['uniform', 'uniform'] #prior functional types

    def sigma(self, *args):
        return [0, 0]
   
    def g(self, x, theta = None):  
        return self.mu(x, theta) / x
    
    def velocity(self, r, theta, *args):
        v200, c200 = theta[:2]
        #c200 = 10 ** logc200
        x = r * 10 * H0 * c200 / v200
        return  v200 * (self.g(x, theta) / self.g(c200, theta)) ** 0.5
 
    def inverse_transformation(self, theta):

        v200, c200 = theta[:2]
        #c200 = 10 ** logc200
        r0 = v200 / (10 * H0 * c200)     
        rho0 = 200 * rho_crit * c200**3 / (3 * self.mu(c200, theta))  
        return r0, rho0
    
    

    def direct_transformation(self, theta):

        r0, rho0 = theta[:2]
   
        def eq(x):
            return rho0 * self.mu(x, theta) - 200 * rho_crit * x**3 / 3

        #x = np.arange(10**-3, 50, 10**-3)
        #y = np.array([eq(_x) for _x in x])
        #plt.plot(x, y)
        #plt.show()

        c200 = scipy.optimize.broyden1(eq, [100], f_tol=1e-6)[0] 
        # the equation has several solutions(2), large initial value allows to find correct solution
        v200 = 10 * H0 * c200 * r0
        #logc200 = np.log10(c200)

        return v200, c200
    
    def sigma_los(self, theta):
        rr = np.logspace(-3, 2, num=500)
        v200, c200 = theta[:2]
        b_a, ra = theta[-2:]
        r200 = v200 / (10 * H0)
        rs = r200 / c200

        plum_arr = plummer(rr, ra)

        mu_arr = np.array([self.mu(r_i/rs, theta) for r_i in rr])

        mu200 = self.mu(c200, theta)

        integrand = (v200**2 * r200 / mu200 ) * mu_arr * plum_arr * (rr)**(2*b_a-2) 

        sigma2r = -((rr)**(-2*b_a) * plum_arr**(-1))[:-1] * scipy.integrate.cumtrapz(integrand[::-1],rr[::-1])[::-1]
        slos_arr = np.array([])
        for idx,r in enumerate(rr):
            y = 2 * rr[idx+1:-1] * plum_arr[idx+1:-1] / np.sqrt((rr[idx+1:-1])**2 - (r)**2) * sigma2r[idx+1:] * (1-b_a * (r/rr[idx+1:-1])**2)
            slos_arr=np.append(slos_arr, np.sqrt(np.array(scipy.integrate.trapz(y, rr[idx+1:-1]))/xiPlum(r, ra)))
        slos = interp1d(rr, slos_arr)
    
        return slos





class Burkert(Model):

    def mu(self, x, theta = None):
        return (np.log(1 + x**2))/4 + np.log(1 + x)/2 - np.arctan(x)/2

    def density(self, r, theta):
        r0, rho0 = self.inverse_transformation(theta)
        return ( rho0 / ( 1 + (r/r0)) ) / (1 + (r/r0)**2) 




class NFW(Model):

    def mu(self, x, theta = None):
        return np.log(1 + x) - x / (1 + x)

    def density(self, r, theta):
        r0, rho0 = self.inverse_transformation(theta)[:2]
        return rho0 / ((r/r0) * (1 + r/r0) ** 2)



class coreNFW(NFW):

    def __init__(self, **kwards):
        super().__init__()
        self.ndim = 4
        self.parameters += ['$n$', r'$\beta_c$']

    def initial(self, r_inf, v_inf):
        #v200_init = v_inf
        #c200_init = 1 
        n_init = 0
        beta_init = 0.1
        return super().initial(r_inf, v_inf) + [n_init, beta_init]  

    def bounds(self):
        #return [[10, 500], [0, 1000], [0, 1], [0, 1]]
        return super().bounds() + [[0, 1], [0, 1]]

    def priors(self):
        return ['uniform', 'uniform', 'uniform', 'uniform']

    def sigma(self, *args):
        return [0, 0, 0, 0]

    def mu(self, x, theta = None):
        v200, c200, n, beta = theta[:4]
        return super().mu(x) * (np.tanh(x/beta))**n

    def density(self, r, theta):
        r0, rho0, n, beta = self.inverse_transformation(theta)

        rc = beta * r0
        x = r / r0 
        f = np.tanh(r/rc)

        return super().density(r, theta) * (np.tanh(r/rc))**n +\
           n * (f**(n-1)) * (1 - f**2) * rho0 * r0**3 * (np.log(1 + x) - x / (1 + x)) / (r**2 * rc)

    def inverse_transformation(self, theta):
        #v200, c200, n, beta = theta[:self.ndim]
        v200, c200, n, beta = theta[:self.ndim]
        r0, rho0 = super().inverse_transformation(theta) 
        return r0, rho0, n, beta

    def direct_transformation(self, theta):
        r0, rho0, n, beta = theta[:self.ndim]        
        v200, c200 = super().direct_transformation(theta)
        #logc200 = np.log10(c200)
        #return v200, c200, n, beta
        return v200, c200, n, beta

    
    def sigma_los(self, theta):

        rr = np.logspace(-3, 2, num=500)

        v200, c200, n, beta, b_a, ra = theta
        r200 = v200 / (10 * H0)
        rs = r200 / c200

        plum_arr = plummer(rr, ra)

        mu_arr = np.array([self.mu(r_i/rs, [v200, c200, n, beta]) for r_i in rr])

        mu200 = self.mu(c200, [v200, c200, n, beta])

        integrand = v200**2 * r200/rs * plum_arr * (rr/rs)**(2*b_a-2) * mu_arr / mu200  

        sigma2r = -((rr/rs)**(-2*b_a)*plum_arr**(-1))[:-1]*scipy.integrate.cumtrapz(integrand[::-1],rr[::-1]/rs)[::-1]
        slos_arr = np.array([])

        for idx,r in enumerate(rr):

            y = 2 * rr[idx+1:-1]/rs * plum_arr[idx+1:-1]/np.sqrt((rr[idx+1:-1]/rs)**2 - (rr[idx]/rs)**2)*sigma2r[idx+1:]*(1-b_a*(rr[idx]/rr[idx+1:-1])**2)
            slos_arr=np.append(slos_arr, np.sqrt(np.array(scipy.integrate.trapz(y, rr[idx+1:-1]/rs))/xiPlum(r, ra)))
        slos = interp1d(rr, slos_arr)

        return slos


class FDM_core(Model):

    def f_sol(self, x):
        return 1 / (1 + 0.091*x**2)**8
  
    def I_sol(self, x):
        a = 9.1 / 100
        return (np.sqrt(a) * x * ( -3465 + 48580 *a * x**2 + 92323 * a**2 * x**4 + 101376 * a**3 * x**6 +  65373 * a**4 * x**8 + 23100 * a**5 * x**10 + 3465 * a**6 * x**12) / (1 + a * x**2)**7 + \
            3465* np.arctan(np.sqrt(a) * x)) / (215040 * a**(3/2))

    def mu(self, x, theta = None):
        return self.I_sol(x) 

    def density(self, r, theta):
        r0, rho0 = self.inverse_transformation(theta)
        return rho0 / (1 + 0.091 * (r/r0)**2)**8



class FDM(Model):
    
    def __init__(self, **kwards):
        super().__init__()
        self.ndim = 4
        #self.parameters = ['$v_{200}$', '$c_{200}$', r'$\alpha$', r'$\beta$']
        self.parameters += [ r'$\alpha$', r'$\beta$']
        # alpha = ra/rs     beta = rn/rs

    def initial(self, r_inf, v_inf):
        #v200_init = v_inf  
        alpha_init = 3
        #c200_init = 10
        beta_init = 5
        #return [v200_init, c200_init, alpha_init, beta_init]
        return super().initial(r_inf, v_inf) + [alpha_init, beta_init]

    def bounds(self):
        #return [[10, 500], [0, 1000], [1, 7], [1, 1000]]
        return super().bounds() + [[1, 7], [1, 1000]]

    def priors(self):
        return ['uniform' for _ in range(self.ndim)]

    def sigma(self, *args):
        return [0 for _ in range(self.ndim)]

    def f_sol(self, x):
        return 1 / (1 + 0.091*x**2)**8

    def f_tail(self, x):
        return 1 / (x * (1 + x)**2)

 
    def I_sol(self, x):
        a = 9.1 / 100
        return (np.sqrt(a) * x * ( -3465 + 48580 *a * x**2 + 92323 * a**2 * x**4 + 101376 * a**3 * x**6 +  65373 * a**4 * x**8 + 23100 * a**5 * x**10 + 3465 * a**6 * x**12) / (1 + a * x**2)**7 + \
            3465* np.arctan(np.sqrt(a) * x)) / (215040 * a**(3/2))
    
    def I_tail(self, x1, x2):
        if x1 >= x2:
            return 0
        return 1 / (1 + x2) - 1 / (1 + x1) + np.log(1 + x2) - np.log(1 + x1)
    
    def mu(self, x, theta):
        v200, c200, alpha, beta = theta[:4]
        K = self.f_sol(alpha) * beta**3 / ( self.f_tail(alpha / beta))
        return self.I_sol(min(alpha, x)) + K * self.I_tail(alpha / beta, x / beta) 


    def velocity(self, r, theta):
        #v200, c200, alpha, beta = theta[:4]
        v200, c200, alpha, beta = theta[:4]
        #c200 = 10 ** logc200
        x = r * 10 * H0 * c200 / v200
        return  np.array([v200 * (self.g(z, theta) / self.g(c200, theta)) ** 0.5 for z in x])


    def inverse_transformation(self, theta):
        #v200, c200, alpha, beta = theta[:4]
        v200, c200, alpha, beta = theta[:4]
        #c200 = 10 ** logc200
        rs = v200 / (10 * H0 * c200)
        rhos = 200 * rho_crit * c200**3 / (3 * self.mu(c200, theta))
        rNFW = beta * rs      
        return rs, rhos, alpha, rNFW


    def direct_transformation(self, theta):# rs, rhos, alpha, rNFW
        rs, rhos, alpha, rNFW = theta[:4] 
        beta = rNFW / rs
        theta0 = [None, None, alpha, beta]
 
        def eq(x):
            return rhos * self.mu(x, theta0)  - 200 * rho_crit * x**3 / 3

        #x = np.arange(10**-2, 100, 10**-2)
        #y = np.array([eq(_x) for _x in x])
        #plt.plot(x, y)
        #plt.show()

        c200 = scipy.optimize.broyden1(eq, [1000], f_tol=1e-10)[0]
        v200 = 10 * H0 * c200 * rs  
        #logc200 = np.log10(c200)
        return v200, c200, alpha, beta


    def density(self, r, theta):

        #v200, c200, alpha, beta = theta[:4]
        v200, c200, alpha, beta = theta[:4]
        #c200 = 10 ** logc200
        rs, rhos, alpha, rNFW = FDM().inverse_transformation(theta)
        ra = rs * alpha
        if r < ra:
            return rhos/(1 + 0.091 * (r/rs)**2)**8
        else:
            rhoNFW =  (rhos / (1 + 0.091 * (ra/rs)**2)**8 ) * (ra/rNFW) * (1 + ra/rNFW)**2 
            return rhoNFW/ (r/rNFW) / (1 + r/rNFW)**2 




class FDM_scaled(FDM):

    def __init__(self, **kwards):
        self.ndim = 4
        self.parameters = ['$v_{200}$', '$log_{10}m_{22}$', r'$\alpha$', r'$\delta$']
        # self.parameters = ['$v_{200}$', '$m_{22}$', r'$\alpha$']

    def initial(self, r_inf, v_inf):
        v200_init = 100 #v_inf  
        logm22_init = 0
        alpha_init = 3
        delta_init = 1
        return [v200_init, logm22_init, alpha_init, delta_init]
        # return [v200_init, m22_init, alpha_init]

    def bounds(self):
        return [[10, 500], [-3,3], [1, 7], [0.5, 1.5]]
        # return [[10, 500], [0.01, 100], [1, 7]]



    def scaling(self, theta): # transfrom this 
        
        v200, logm22, alpha, delta = theta[:self.ndim]
        m22 = 10 ** logm22

        #c200 = (0.0537 * (1/invm22) * delta * v200**2 * rho_crit**(1/3)) / (H0**2)
        c200 = (0.0537 * m22 * delta * v200**2 * rho_crit**(1/3)) / (H0**2)

        if c200 < alpha:
            beta = 1 # the profile containe only FDM core in this case and do not depends on beta 
        else:
            def eq(b):
                theta = [v200, c200, alpha, b]
                #return self.mu(c200, theta) - (0.350877 * (1/invm22)**2 * rho_crit * v200**4) / (c200 * H0**4)
                return self.mu(c200, theta) - (0.350877 * m22**2 * rho_crit * v200**4) / (c200 * H0**4)

            #x = np.arange(0.001, 100, 0.1) 
            #y = np.array([eq(_x) for _x in x])
            #plt.plot(x, y)
            #plt.show()

            #beta = scipy.optimize.broyden1(eq, [10], f_tol=1e-6)[0]  # !!! non-convergence error
            beta = scipy.optimize.toms748(eq, 1/local_inf, local_inf)
            #print('beta ', beta)

        #logc200 = np.log10(c200)
        return v200, c200, alpha, beta
   
    

    def scaling_rrho(self, theta):

        rs, logm22, alpha, delta = theta[:self.ndim]
        m22 = 10 ** logm22

        #rhos = 1.9 * 0.01 / ((1/invm22)**2 * rs**4)
        rhos = 1.9 * 0.01 / (m22**2 * rs**4)

        #M200 = 5.42 * 10**9 / ((1/invm22)**3 * delta**3 * rs**3) 
        M200 = 5.42 * 10**9 / (m22**3 * delta**3 * rs**3) 
        #r200 = 186.7 / ((1/invm22) * delta * rs * rho_crit**(1/3)) / 1000
        r200 = 186.7 / (m22 * delta * rs * rho_crit**(1/3)) / 1000

        c200 = r200 / rs
 
        if c200 < alpha:
            beta = 1
        else:
            def eq(b):
                theta = [None, None, alpha, b]
                return 4 * np.pi * rhos * rs**3 * self.mu(c200, theta) - M200 / 10**9

            #x = np.arange(10**-3, 1000, 0.1)
            #y = np.array([eq(_x) for _x in x])
            #plt.plot(x, y)
            #plt.show()

            beta = scipy.optimize.broyden1(eq, [10], f_tol=1e-6)[0]

        rNFW = rs * beta
        return rs, rhos, alpha, rNFW


    def inv_scaling(self, theta): 
        v200, c200, alpha, beta = theta[:4]
        #c200 = 10 ** logc200
        #invm22 = 1 / ( (1.68819 * np.sqrt(c200) * H0**2 * np.sqrt(self.mu(c200, theta))) / (np.sqrt(rho_crit) * v200**2) )
        m22 = (1.68819 * np.sqrt(c200) * H0**2 * np.sqrt(self.mu(c200, theta))) / (np.sqrt(rho_crit) * v200**2)
        logm22 = np.log10(m22)
        delta = (18.6335 * c200 * H0**2) / (m22 *  rho_crit**(1/3) * v200**2)
        #delta = (18.6335 * c200 * H0**2) / ((1/invm22) *  rho_crit**(1/3) * v200**2)
        return v200, logm22, alpha, delta


    def inv_scaling_rrho(self, theta):
        rs, rhos, alpha, rNFW = theta[:4]
        #invm22 = 1 / (0.13784 / (np.sqrt(rhos) *rs**2))
        m22 = 0.13784 / (np.sqrt(rhos) *rs**2)
        logm22 = np.log10(m22)

        # find r200
        def eq(r200):
            theta0 = [None, r200/rs, alpha, rNFW/rs]
            return rhos * rs**3 * self.mu(r200 / rs, theta0)  - 200 * rho_crit * r200**3 / 3

        #x = np.arange(10**-2, 1000, 10**-2)
        #y = np.array([eq(_x) for _x in x])
        #plt.plot(x, y)
        #plt.show()

        r200 = scipy.optimize.broyden1(eq, [100], f_tol=1e-12)[0]

        delta =  0.186335 / (m22 * r200 * rho_crit**(1/3) * rs)
        #delta =  0.186335 / ((1/invm22) * r200 * rho_crit**(1/3) * rs)

        return rs, logm22, alpha, delta


    def check_parameters(self, theta):
        
        v200, logm22, alpha, delta = theta[:self.ndim]
        m22 = 10 ** logm22
        c200 = (0.0537 * m22 * delta * v200**2 * rho_crit**(1/3)) / (H0**2)

        theta0 = [None, None, alpha, local_inf]
        if self.mu(c200, theta0) - (0.350877 * m22**2 * rho_crit * v200**4) / (c200 * H0**4) < 0:
            return False
        theta1 = [None, None, alpha, 1 / local_inf]
        if self.mu(c200, theta1) - (0.350877 * m22**2 * rho_crit * v200**4) / (c200 * H0**4) > 0:
            return False
        return True

         
    def velocity(self, r, theta):
        
        if self.check_parameters(theta):
            theta0 = self.scaling(theta[:self.ndim]) # v200, c200, alpha, beta
            return super().velocity(r, theta0)
        else:
            return - local_inf


    #def velocity(self, r, theta):
    #    try: 
    #        theta0 = self.scaling(theta[:self.ndim]) # v200, c200, alpha, beta
    #        return super().velocity(r, theta0)
    #    except Exception:
    #        return -1000
        

    def density(self, r, theta):

        theta0 = self.scaling(theta[:self.ndim]) # v200, c200, alpha, beta
        return super().density(r, theta0)


    def inverse_transformation(self, theta): # v200, m22, alpha

        theta0 = self.scaling(theta) # v200, logc200, alpha, beta
        theta_r4 = super().inverse_transformation(theta0)
        theta_r2 = self.inv_scaling_rrho(theta_r4) 

        return theta_r2     # rs, logm22, alpha


    def direct_transformation(self, theta): # rs, m22, alpha
    
        theta0 = self.scaling_rrho(theta) # rs, rhos, alpha, rNFW
        theta1 = super().direct_transformation(theta0)
        theta2 = self.inv_scaling(theta1)

        return theta2 # v200, logm22, alpha
    

    def sigma_los(self, theta):

        rr = np.logspace(-3, 2, num=500)
        if self.check_parameters(theta):

            v200, logm22, alpha, delta, b_a, ra = theta
            v200, c200, alpha, beta = self.scaling(theta)
            r200 = v200 / (10 * H0)
            rs = r200 / c200
            
            plum_arr = plummer(rr, ra)

            mu_arr = np.array([self.mu(r_i/rs, [v200, c200, alpha, beta]) for r_i in rr])

            mu200 = self.mu(c200, [v200, c200, alpha, beta])

            integrand = v200**2 * r200/rs * plum_arr * (rr/rs)**(2*b_a-2) * mu_arr / mu200  

            sigma2r = -((rr/rs)**(-2*b_a)*plum_arr**(-1))[:-1]*scipy.integrate.cumtrapz(integrand[::-1],rr[::-1]/rs)[::-1]
            slos_arr = np.array([])

            for idx,r in enumerate(rr):

                y = 2 * rr[idx+1:-1]/rs * plum_arr[idx+1:-1]/np.sqrt((rr[idx+1:-1]/rs)**2 - (rr[idx]/rs)**2)*sigma2r[idx+1:]*(1-b_a*(rr[idx]/rr[idx+1:-1])**2)
                slos_arr=np.append(slos_arr, np.sqrt(np.array(scipy.integrate.trapz(y, rr[idx+1:-1]/rs))/xiPlum(r, ra)))
        else:
            
            slos = interp1d(rr, np.full_like(rr, local_inf))
            return slos
        slos = interp1d(rr, slos_arr)

        return slos



class DC14(Model):

    def __init__(self, **kwards):
        self.ndim = 2
        self.parameters = ['$v_{200}$', '$c_{200}$']
        kwards.setdefault('di', None)
        self.mlums = kwards['mlums']
        self.di = kwards['di']
        

    @staticmethod
    def transform(Xstar):
        if Xstar<=-4.1:
            return 1.,3.,1.
        
        alpha = 2.94 - np.log10(10**((Xstar+2.33)*(-1.08)) + (10**((Xstar+2.33)*2.29)))
        beta = 4.23 + 1.34*Xstar + 0.26*Xstar**2
        gamma = -0.06 + np.log10(10**((Xstar+2.56)*(-0.68)) + 10**(Xstar+2.56))
        
        return alpha, beta, gamma

    def mu(self, x, Xstar):

        alpha, beta, gamma = self.transform(Xstar)
        epsl = x**alpha/(1 + x**alpha)
        a = (3 - gamma)/alpha
        b = (beta - 3)/alpha

        return betainc(a, b+1, epsl) + betainc(a+1,b, epsl)

    def g(self, x, Xstar):  
        return self.mu(x, Xstar) / x    
    
    def velocity(self, r, theta, *args):
        
        v200, c200 = theta[:2]
        d = theta[2]
        r200 = v200/(10*H0)
        m200 = 4 * np.pi/3 * 200 * rho_crit * r200**3 *10**9 #10**9 -- pc^-3 to kpc^-3 factor
        
         
        #ndim_mlums = len(self.mlums)
        Y = np.array(theta[self.ndim + 2 :])
        mstar = sum(Y*self.mlums[1:])
        if self.di!=None:
            mstar = mstar * (d/self.di)**2
        #mbar = self.mlums[0] * (d/self.di)**2 + mstar

        #X = np.log10(mstar/(mbar+m200))
        Xstar = np.log10(mstar/(m200)) #like in Allaert-17
        #c200 = 10 ** logc200
        x = r * 10 * H0 * c200 / v200
        if Xstar>-1.3:
            return local_inf
        return  v200 * (self.g(x, Xstar) / self.g(c200, Xstar)) ** 0.5



    


class FDM_fixedmass(FDM):

    def __init__(self, **kwards):
        self.ndim = 3
        self.parameters = ['$v_{200}$', r'$\alpha$', r'$\delta$']
        self.m22 = kwards['m22']

    def initial(self, r_inf, v_inf):
        v200_init = 100  
        alpha_init = 3
        delta_init = 1
        return [v200_init, alpha_init, delta_init]


    def bounds(self):
        return [[10, 500], [1, 7], [0.5, 1.5]]


    def scaling(self, theta): # transfrom this 
        
        v200, alpha, delta = theta[:self.ndim]
        c200 = (0.0537 * self.m22 * delta * v200**2 * rho_crit**(1/3)) / (H0**2)

        if c200 < alpha:
            beta = 1 # the profile containe only FDM core in this case and do not depends on beta 
        else:
            def eq(b):
                theta = [v200, c200, alpha, b]
                return self.mu(c200, theta) - (0.350877 * self.m22**2 * rho_crit * v200**4) / (c200 * H0**4)

            #x = np.arange(0.001, 1000, 0.1) 
            #y = np.array([eq(_x) for _x in x])
            #plt.plot(x, y)
            #plt.show()

            #beta = scipy.optimize.broyden1(eq, [10], f_tol=1e-6)[0]  # !!! non-convergence error
            beta = scipy.optimize.toms748(eq, 1/local_inf, local_inf)
            #print('beta ', beta)

        return v200, c200, alpha, beta
  
    
    def scaling_rrho(self, theta): # rs, alpha, delta

        rs, alpha, delta = theta[:self.ndim]
        rhos = 1.9 * 0.01 / (self.m22**2 * rs**4)
        M200 = 5.42 * 10**9 / (self.m22**3 * delta**3 * rs**3) 
        r200 = 186.7 / (self.m22 * delta * rs * rho_crit**(1/3)) / 1000

        c200 = r200 / rs
 
        if c200 < alpha:
            beta = 1
        else:
            def eq(b):
                theta = [None, None, alpha, b]
                return 4 * np.pi * rhos * rs**3 * self.mu(c200, theta) - M200 / 10**9

            #x = np.arange(10**-3, 1000, 0.1)
            #y = np.array([eq(_x) for _x in x])
            #plt.plot(x, y)
            #plt.show()

            beta = scipy.optimize.broyden1(eq, [10], f_tol=1e-6)[0]

        rNFW = rs * beta
        return rs, rhos, alpha, rNFW


    def inv_scaling(self, theta): # v200, logc200, alpha, beta
        v200, c200, alpha, beta = theta[:4]
        #c200 = 10 ** logc200
        #m22 = (1.68819 * np.sqrt(c200) * H0**2 * np.sqrt(self.mu(c200, theta))) / (np.sqrt(rho_crit) * v200**2)
        #logm22 = np.log10(m22)
        delta = (18.6335 * c200 * H0**2) / (self.m22 *  rho_crit**(1/3) * v200**2)
        return v200, alpha, delta


    def inv_scaling_rrho(self, theta):   # the problem is here
        rs, rhos, alpha, rNFW = theta[:4] 
        #m22 = 0.13784 / (np.sqrt(rhos) *rs**2)
        #logm22 = np.log10(m22)

        # find r200
        def eq(r200):
            theta0 = [None, r200/rs, alpha, rNFW/rs]
            return rhos * rs**3 * self.mu(r200 / rs, theta0)  - 200 * rho_crit * r200**3 / 3

        #x = np.arange(10**-2, 1000, 10**-2)
        #y = np.array([eq(_x) for _x in x])
        #plt.plot(x, y)
        #plt.show()

        #r200 = scipy.optimize.broyden1(eq, [100], f_tol=1e-12)[0]
        r200 = scipy.optimize.broyden1(eq, [1000], f_tol=1e-12)[0]

        delta =  0.186335 / (self.m22 * r200 * rho_crit**(1/3) * rs)
        #delta =  0.186335 / ((1/invm22) * r200 * rho_crit**(1/3) * rs)

        return rs, alpha, delta



    def check_parameters(self, theta):
        
        v200, alpha, delta = theta[:self.ndim]
        #m22 = 10 ** logm22
        c200 = (0.0537 * self.m22 * delta * v200**2 * rho_crit**(1/3)) / (H0**2)

        theta0 = [None, None, alpha, local_inf]
        if self.mu(c200, theta0) - (0.350877 * self.m22**2 * rho_crit * v200**4) / (c200 * H0**4) < 0:
            return False
        theta1 = [None, None, alpha, 1 / local_inf]
        if self.mu(c200, theta1) - (0.350877 * self.m22**2 * rho_crit * v200**4) / (c200 * H0**4) > 0:
            return False
        return True

         
    def velocity(self, r, theta):
        
        if self.check_parameters(theta):
            theta0 = self.scaling(theta[:self.ndim]) # v200, c200, alpha, beta
            return super().velocity(r, theta0)
        else:
            return - local_inf


    #def velocity(self, r, theta):
    #    try: 
    #        theta0 = self.scaling(theta[:self.ndim]) # v200, c200, alpha, beta
    #        return super().velocity(r, theta0)
    #    except Exception:
    #        return -1000
        

    def density(self, r, theta):

        theta0 = self.scaling(theta[:self.ndim]) # v200, logc200, alpha, beta
        return super().density(r, theta0)


    def inverse_transformation(self, theta): # v200, alpha, delta

        #print('v200, alpha, delta: ', theta)
        theta0 = self.scaling(theta) # v200, logc200, alpha, beta
        #print('v200, logc200, alpha, delta: ', theta0)
        theta_r4 = super().inverse_transformation(theta0) # rs, rhos, alpha, rNFW
        #print('rs, rhos, alpha, rNFW: ', theta_r4)
        theta_r2 = self.inv_scaling_rrho(theta_r4) 
        #print('rs, alpha, delta: ', theta_r2)

        return theta_r2  # rs, alpha, delta


    def direct_transformation(self, theta): # rs, alpha, delta
    
        theta0 = self.scaling_rrho(theta) # rs, rhos, alpha, rNFW
        theta1 = super().direct_transformation(theta0) # v200, logc200, alpha, beta
        theta2 = self.inv_scaling(theta1)

        return theta2 # v200, alpha, delta 



class FDM_fixeddelta(FDM):

    def __init__(self, **kwards):
        self.ndim = 3
        self.parameters = ['$v_{200}$', '$log_{10}m_{22}$', r'$\alpha$']
        self.delta = 1

    def initial(self, r_inf, v_inf):
        v200_init = 100 #v_inf  
        logm22_init = 0
        alpha_init = 3
        return [v200_init, logm22_init, alpha_init]


    def bounds(self):
        return [[10, 500], [-3,3], [1, 7]]


    def scaling(self, theta): # transfrom this 
        
        v200, logm22, alpha = theta[:self.ndim]
        m22 = 10 ** logm22
        delta = self.delta
        c200 = (0.0537 * m22 * delta * v200**2 * rho_crit**(1/3)) / (H0**2)

        if c200 < alpha:
            beta = 1 # the profile containe only FDM core in this case and do not depends on beta 
        else:
            def eq(b):
                theta = [v200, c200, alpha, b]
                return self.mu(c200, theta) - (0.350877 * m22**2 * rho_crit * v200**4) / (c200 * H0**4)

            #x = np.arange(0.001, 1000, 0.1) 
            #y = np.array([eq(_x) for _x in x])
            #plt.plot(x, y)
            #plt.show()

            #beta = scipy.optimize.broyden1(eq, [10], f_tol=1e-6)[0]  # !!! non-convergence error
            beta = scipy.optimize.toms748(eq, 1/local_inf, local_inf)
            #print('beta ', beta)

        return v200, c200, alpha, beta
  
    
    def scaling_rrho(self, theta): # rs, alpha, delta

        rs, logm22, alpha = theta[:self.ndim]
        m22 = 10 ** logm22
        delta = self.delta
        rhos = 1.9 * 0.01 / (m22**2 * rs**4)
        M200 = 5.42 * 10**9 / (m22**3 * delta**3 * rs**3) 
        r200 = 186.7 / (m22 * delta * rs * rho_crit**(1/3)) / 1000

        c200 = r200 / rs
 
        if c200 < alpha:
            beta = 1
        else:
            def eq(b):
                theta = [None, None, alpha, b]
                return 4 * np.pi * rhos * rs**3 * self.mu(c200, theta) - M200 / 10**9

            #x = np.arange(10**-3, 1000, 0.1)
            #y = np.array([eq(_x) for _x in x])
            #plt.plot(x, y)
            #plt.show()

            beta = scipy.optimize.broyden1(eq, [10], f_tol=1e-6)[0]

        rNFW = rs * beta
        return rs, rhos, alpha, rNFW


    def inv_scaling(self, theta): # v200, logc200, alpha, beta
        v200, c200, alpha, beta = theta[:4]
        #c200 = 10 ** logc200
        m22 = (1.68819 * np.sqrt(c200) * H0**2 * np.sqrt(self.mu(c200, theta))) / (np.sqrt(rho_crit) * v200**2)
        logm22 = np.log10(m22)
        delta = (18.6335 * c200 * H0**2) / (self.m22 *  rho_crit**(1/3) * v200**2)
        return v200, logm22, alpha 


    def inv_scaling_rrho(self, theta):   # the problem is here
        rs, rhos, alpha, rNFW = theta[:4] 
        m22 = 0.13784 / (np.sqrt(rhos) *rs**2)
        logm22 = np.log10(m22)

        # find r200
        def eq(r200):
            theta0 = [None, r200/rs, alpha, rNFW/rs]
            return rhos * rs**3 * self.mu(r200 / rs, theta0)  - 200 * rho_crit * r200**3 / 3

        #x = np.arange(10**-2, 1000, 10**-2)
        #y = np.array([eq(_x) for _x in x])
        #plt.plot(x, y)
        #plt.show()

        #r200 = scipy.optimize.broyden1(eq, [100], f_tol=1e-12)[0]
        r200 = scipy.optimize.broyden1(eq, [1000], f_tol=1e-12)[0]
#
        #delta =  0.186335 / (self.m22 * r200 * rho_crit**(1/3) * rs)
        #delta =  0.186335 / ((1/invm22) * r200 * rho_crit**(1/3) * rs)

        return rs, logm22, alpha



    def check_parameters(self, theta):
        
        v200, logm22, alpha = theta[:self.ndim]
        m22 = 10 ** logm22
        delta = self.delta
        c200 = (0.0537 * m22 * delta * v200**2 * rho_crit**(1/3)) / (H0**2)

        theta0 = [None, None, alpha, local_inf]
        if self.mu(c200, theta0) - (0.350877 * m22**2 * rho_crit * v200**4) / (c200 * H0**4) < 0:
            return False
        theta1 = [None, None, alpha, 1 / local_inf]
        if self.mu(c200, theta1) - (0.350877 * m22**2 * rho_crit * v200**4) / (c200 * H0**4) > 0:
            return False
        return True

         
    def velocity(self, r, theta):
        
        if self.check_parameters(theta):
            theta0 = self.scaling(theta[:self.ndim]) # v200, c200, alpha, beta
            return super().velocity(r, theta0)
        else:
            return - local_inf


    #def velocity(self, r, theta):
    #    try: 
    #        theta0 = self.scaling(theta[:self.ndim]) # v200, c200, alpha, beta
    #        return super().velocity(r, theta0)
    #    except Exception:
    #        return -1000
        

    def density(self, r, theta):

        theta0 = self.scaling(theta[:self.ndim]) # v200, logc200, alpha, beta
        return super().density(r, theta0)


    def inverse_transformation(self, theta): # v200, alpha, delta

        #print('v200, alpha, delta: ', theta)
        theta0 = self.scaling(theta) # v200, logc200, alpha, beta
        #print('v200, logc200, alpha, delta: ', theta0)
        theta_r4 = super().inverse_transformation(theta0) # rs, rhos, alpha, rNFW
        #print('rs, rhos, alpha, rNFW: ', theta_r4)
        theta_r2 = self.inv_scaling_rrho(theta_r4) 
        #print('rs, alpha, delta: ', theta_r2)

        return theta_r2  # rs, alpha, delta


    def direct_transformation(self, theta): # rs, alpha, delta
    
        theta0 = self.scaling_rrho(theta) # rs, rhos, alpha, rNFW
        theta1 = super().direct_transformation(theta0) # v200, logc200, alpha, beta
        theta2 = self.inv_scaling(theta1)

        return theta2 # v200, alpha, delta 
    

class MOND(Model):
    
    def __init__(self, **kwards):
        self.baryons = TotalLuminous(kwards['bulge']) 
        self.baryons.initial(kwards['D'], kwards['i']) 
        
        self.ndim = 1 + self.baryons.ndim
        kwards.setdefault('a0_prior', 'flat_log')
        self.parameters = [r'$\log_{10}a_0$'] + self.baryons.parameters

        if kwards['a0_prior'] == 'norm':
            self.parameters[0]= r'$a_0$'
            
        if kwards['a0_prior']!='norm' and kwards['a0_prior'] != 'flat_log':
            raise KeyError
            
        
        
            

        

        
    def initial(self, *args):
        if self.parameters[0] == r'$a_0$':
            return [1.2]  + self.baryons.initial(self.baryons.D, self.baryons.i)
        return [-11]  + self.baryons.initial(self.baryons.D, self.baryons.i)
    

    def bounds(self):
        if self.parameters[0] == r'$a_0$':
            return [[0.2,2.4]]  + self.baryons.bounds()
        return [[-16, -10]] + self.baryons.bounds()

    def priors(self):
        if self.parameters[0] == r'$a_0$':
            return ['norm'] + self.baryons.priors()
        elif self.parameters[0] == r'$\log_{10}a_0$':
            return ['uniform'] + self.baryons.priors()

        

    def sigma(self, *args):
        if self.parameters[0] == r'$a_0$':
            return [0.2]  + self.baryons.sigma(*args)
        return [0] + self.baryons.sigma(*args)
    
    def interpolation_function(self, a0, abar):
        return abar / (1 - np.exp(-np.sqrt(abar/a0))) 

    def velocity(self, r, theta, *lum_velocities):
        v_gas, v_disk, v_bulge = lum_velocities
        abar = self.baryons.velocity_square(v_gas, v_disk, v_bulge, theta) / (r * kpc_to_km)
        if np.any(abar < 0):
            return np.array([- local_inf for _ in r])
        
        if self.parameters[0] == r'$\log_{10}a_0$':
            log10_a0 = theta[0]
            a0 = 10**log10_a0

        if self.parameters[0] == r'$a_0$':
            a0 = theta[0]*10**-13

        atot = self.interpolation_function(a0, abar)   
        return  np.sqrt(atot * r * kpc_to_km)


#    def velocity(self, r, theta, *lum_velocities):
#        v_gas, v_disk, v_bulge = lum_velocities
#        abar = self.baryons.velocity_square(v_gas, v_disk, v_bulge, theta) / (r * kpc_to_km)
#        if np.any(abar < 0):
#            return np.array([- local_inf for _ in r])
#        log10_a0 = theta[0]
#        a0 = 10**log10_a0
#        atot = abar / (1 - np.exp(-np.sqrt(abar/a0)))     
#        return  np.sqrt(atot * r * kpc_to_km)

    def inverse_transformation(self, theta):  
        return theta

    def direct_transformation(self, theta):
        return theta 



class MOND_simple(MOND):

    def interpolation_function(self, a0, abar):
        return abar * (1/2 + np.sqrt(a0/abar + 1/4)) 


class MOND_standard(MOND):

    def interpolation_function(self, a0, abar):
        return abar * np.sqrt(1/2 + np.sqrt(a0**2/abar**2 + 1/4)) 

class MOND_fix(MOND):

    def __init__(self, **kwards):
        self.a0 = 1.2e-13
        self.baryons = TotalLuminous(kwards['bulge'])
        self.baryons.initial(kwards['D'], kwards['i'])

        self.ndim = self.baryons.ndim
        self.parameters = self.baryons.parameters

    def initial(self, *args):
        return self.baryons.initial(self.baryons.D, self.baryons.i)

    def bounds(self):
        return self.baryons.bounds()

    def priors(self):
        return self.baryons.priors()

    def sigma(self, *args):
        return self.baryons.sigma(*args)

    def velocity(self, r, theta, *lum_velocities):
        v_gas, v_disk, v_bulge = lum_velocities
        abar = self.baryons.velocity_square(v_gas, v_disk, v_bulge, theta) / (r * kpc_to_km)
        if np.any(abar < 0):
            return np.array([- local_inf for _ in r])
        atot = self.interpolation_function(self.a0, abar)
        return  np.sqrt(atot * r * kpc_to_km)


class MOND_simple_fix(MOND_fix):

    def interpolation_function(self, a0, abar):
        return abar * (1/2 + np.sqrt(a0/abar + 1/4))


class MOND_standard_fix(MOND_fix):

    def interpolation_function(self, a0, abar):
        return abar * np.sqrt(1/2 + np.sqrt(a0**2/abar**2 + 1/4))



    
    

# --------------------------------------------
#        ***  Baryonic matter models  ***
# --------------------------------------------


class SimpleLuminous(Model):

    def __init__(self, bulge):
        self.bulge = bulge
        if self.bulge:
            self.parameters = ['$\gamma_{disk}$', '$\gamma_{bulge}$']
            self.ndim = 2
        else:
            self.parameters = ['$\gamma_{disk}$']
            self.ndim = 1

    def initial(self, *args):
        if self.bulge:
            return [0.5, 0.7]
        return [0.5]

    def bounds(self):
        if self.bulge:
            return [[0, 1.2], [0, 1.5]]
        return [[0, 1.2]]

    def priors(self):
        if self.bulge:
            return ['lognorm', 'lognorm']
        return['lognorm']

    def sigma(self, *args):
        s = 0.1 * np.log(10)
        if self.bulge:
            return [s, s]
        return [s]

#     def velocity(self, v_gas, v_disk, v_bulge, theta):
#         if self.bulge:
#             g_disk, g_bulge = theta[-self.ndim:]
#             return (v_gas**2 + g_disk * v_disk**2 + g_bulge * v_bulge**2)**0.5
#         g_disk = theta[-1]
#         return (v_gas**2 + g_disk * v_disk**2)**0.5 
    
#    def velocity_square(self, v_gas, v_disk, v_bulge, theta):
#        if self.bulge:
#            g_disk, g_bulge = theta[-self.ndim:]
#            return v_gas**2 + g_disk * v_disk**2 + g_bulge * v_bulge**2
#        g_disk = theta[-1]
#        return v_gas**2 + g_disk * v_disk**2

    def velocity_square(self, v_gas, v_disk, v_bulge, theta):
        if self.bulge:
            g_disk, g_bulge = theta[-self.ndim:]
            return v_gas * abs(v_gas) + g_disk * v_disk * abs(v_disk) + g_bulge * v_bulge * abs(v_bulge)
        g_disk = theta[-1]
        return v_gas * abs(v_gas) + g_disk * v_disk * abs(v_disk)

     
    def rescale(self, r, v, err, theta):
        return r, v, err



# --------------------------------------------
#        ***  Baryonic matter models  ***
# --------------------------------------------
        
    

class TotalLuminous(SimpleLuminous):

    def __init__(self, bulge):
        super().__init__(bulge)
        self.ndim += 2 
        self.parameters = ['$D$', '$i$'] + self.parameters

    def initial(self, *args): 
        self.D, self.i = args
        return [self.D, self.i] + super().initial()  

    def bounds(self):
        return [[0, 1000], [0, 90]] + super().bounds()

    def priors(self):
        return ['norm', 'norm'] + super().priors()

    def sigma(self, *args):
        return list(args) + super().sigma() 

#     def velocity(self, v_gas, v_disk, v_bulge, theta):
#         if self.bulge:
#             D, i, g_disk, g_bulge = theta[-self.ndim:]
#             #return ((v_gas**2 + g_disk * v_disk **2 + g_bulge * v_bulge**2) * D / self.D) **0.5
#             return check_sqrt((v_gas * abs(v_gas) + g_disk * v_disk * abs(v_disk) + g_bulge * v_bulge * abs(v_bulge)) * D / self.D)
#         D, i, g_disk = theta[-self.ndim:]
#         #return ((v_gas**2 + g_disk * v_disk**2) * D / self.D)**0.5
#         return check_sqrt((v_gas * abs(v_gas) + g_disk * v_disk * abs(v_disk)) * D / self.D)

    
    def velocity_square(self, v_gas, v_disk, v_bulge, theta):
        if self.bulge:
            D, i, g_disk, g_bulge = theta[-self.ndim:]
            return (v_gas * abs(v_gas) + g_disk * v_disk * abs(v_disk) + g_bulge * v_bulge * abs(v_bulge)) * D / self.D
        D, i, g_disk = theta[-self.ndim:]
        return (v_gas * abs(v_gas) + g_disk * v_disk * abs(v_disk)) * D / self.D
    
    
    def rescale(self, r, v, err, theta):
        if self.bulge:
            D, i, g_disk, g_bulge = theta[-self.ndim:]
        else:
            D, i, g_disk = theta[-self.ndim:]

        v_obs = v * np.sin(np.deg2rad(self.i)) / np.sin(np.deg2rad(i))
        err_obs = err * np.sin(np.deg2rad(self.i)) / np.sin(np.deg2rad(i))
        r_obs = r * D / self.D

        return r_obs, v_obs, err_obs

###############################################################


class ExpThin(Model):


    def velocity_square(self, r, log_M, R, **kwards):
        M = 10 ** log_M
        x = r / (2 * R)
        return 2 * G_N * M / R * x**2 * (i0(x)*k0(x) - i1(x)*k1(x))

    def rescale(self, v, err, i0, i):
        
        v_obs = v * np.sin(np.deg2rad(i0)) / np.sin(np.deg2rad(i))
        err_obs = err * np.sin(np.deg2rad(i0)) / np.sin(np.deg2rad(i))

        return v_obs, err_obs

###################################################################################


class MP21(Model):

    def __init__(self):

        self.ndim  = 7 
        self.parameters = ['log_Mstar', 'Rstar', 'n', 'log_sigma0_gas', 'R1gas', 'R2gas', 'alpha'] 
        

    @property    
    def priors(self):
        return['norm', 'norm','norm','uniform','uniform','uniform', 'uniform']

    @property    
    def bounds(self):
        return [[6,10],[0,5], [0.36,10.], [4.,10.],[0.01,10.], [0.01,10.], [-500,1000.]]

    def initial(self, inits):
        return inits + [0,0,0,0]

    def sigma(self, sigmas):
        return sigmas + [0,0,0,0]

    def velocity_square(self, r, sigma0_star, Rstar, n, sigma0_gas, R1gas, R2gas, alpha, gamma_gas=1.33, gamma_star=1, **kwards):
        
        ser_d = dc.Sersic_disc.thin(sigmae=sigma0_star, Re=Rstar, n=n)
        
        vcirc_star=ser_d.vcirc(r)[:,1] # Vcric 2D array: col-0 R, col-1 Circular velocity on the plane [km/s]
        
        gas_distr = dc.Frat_disc.thin(sigma0 = sigma0_gas, Rd = R1gas, alpha=alpha, Rd2=R2gas)
        
        vcirc_gas = gas_distr.vcirc(r)[:,1]
        v2_bar = gamma_star * vcirc_star**2 + gamma_gas * vcirc_gas **2

        return v2_bar if not np.isnan(v2_bar).any() else  np.full_like(r, local_inf**2)
    
    def surf_density_gas(self, r, sigma0_gas, R1gas, R2gas, alpha, **kwards):
        # mix of a gaussian+exponential 
        gas_distr = dc.Frat_disc.thin(sigma0 = sigma0_gas, Rd = R1gas, alpha=alpha, Rd2=R2gas)

        return gas_distr.Sdens(r)   
        

    



class ZeroLuminous(Model):

    def __init__(self, bulge):
        self.bulge = bulge
        self.ndim  = 0 
        self.parameters = [] 
    
    def initial(self, *args):
        return []

    def bounds(self):
        return []

    def priors(self):
        return[]

    def sigma(self, *args):
        return []
    
    def velocity_square(self, v_gas, v_disk, v_bulge, theta):
        return np.array([0 for _ in v_gas])
     
    def rescale(self, r, v, err, theta):
        return r, v, err



class MONDLumionus(ZeroLuminous):
    
    def initial(self, *args):
        self.D, self.i = args
        return []
        #return [self.D, self.i]
    
    def rescale(self, r, v, err, theta):
        if self.bulge:
            D, i, g_disk, g_bulge = theta[-4:] 
        else:
            D, i, g_disk = theta[-3:]      

        v_obs = v * np.sin(np.deg2rad(self.i)) / np.sin(np.deg2rad(i))
        err_obs = err * np.sin(np.deg2rad(self.i)) / np.sin(np.deg2rad(i))
        r_obs = r * D / self.D

        return r_obs, v_obs, err_obs


 #       def rescale(self, r, v, err, theta):
 #       if self.bulge:
 #           D, i, g_disk, g_bulge = theta[-self.ndim:]
 #       else:
 #           D, i, g_disk = theta[-self.ndim:]

 #       v_obs = v * np.sin(np.deg2rad(self.i)) / np.sin(np.deg2rad(i))
 #       err_obs = err * np.sin(np.deg2rad(self.i)) / np.sin(np.deg2rad(i))
 #       r_obs = r * D / self.D

 #       return r_obs, v_obs, err_obs


class dSphLum(Model):

    def __init__(self):
        self.parameters = ['$\beta_a$', '$r_a$']

    def initial(self, *args):
        return [0, args[1]]

    def bounds(self):
        
        return [[-10, 1], [0, 100]]

    def priors(self):
        return ['flat', 'norm']
    
    def sigma(self):
        pass

 

# ----------------------------------------------------------------

#                ****   Profile tests    ****

# ----------------------------------------------------------------


def param_compare():
    
    #rs = 1
    #rhos = 0.01
    #n = 0.5
    #beta = 0.02

    #rs = 1
    #rhos = np.exp(-2.52)
    #alpha = 3
    #rNFW = 50

    rs = 1 #20
    alpha = 3
    #logm22 = -1
    delta = 0.5

    ##rs = 1
    ##m22 = 0.1
    ##alpha = 3
    ##delta = 1


    #model = NFW 
    model = FDM_fixedmass
    #model = coreNFW
    kwards = {'m22': 0.1}
    #kwards = {}

    print('Direct + Inverse')

    #theta = rs, rhos
    #theta = rs, rhos, n, beta
    #theta = rs, logm22, alpha, delta
    #theta = rs, rhos, alpha, rNFW
    theta = rs, alpha, delta

    theta0 = model(**kwards).direct_transformation(theta)
    theta1 = model(**kwards).inverse_transformation(theta0)

    print('r0, rho0 : ', theta)
    print('v200, c200: ', theta0)
    print('r0, rho0: ', theta1)

    print()
    print('Inverse + Direct')

    #v200 = 100
    #logc200 = 0.2
    #alpha = 3
    #beta = 20

    v200 = 100
    #logm22 = -1
    alpha = 3
    delta = 1

    ##v200 = 100
    ##invm22 = 10
    ##alpha = 3
    ##delta = 1

    #v200 = 10
    #c200 = 1
    #n = 0.5
    #beta = 0.02
    
    #theta = v200, logc200
    #theta = v200, logc200, n, beta
    #theta = v200, logm22, alpha, delta
    theta = v200, alpha, delta

    theta0 = model(**kwards).inverse_transformation(theta)
    theta1 = model(**kwards).direct_transformation(theta0)

    print('v200, c200 : ', theta)
    print('r0, rho0 : ', theta0)
    print('v200, c200 : ', theta1)


def error(func1, func2):
    if len(func1) != len(func2):
        return -9999
    return sum([abs(func1[i] - func2[i]) for i in range(len(func1))]) / len(func1)


def plot_2velocities(r, v1, v2, model_name):
   fig, ax = plt.subplots()
   ax.plot(r, v1, label = 'analytical')
   ax.plot(r, v2, label = 'numerical')
   ax.legend()
   ax.set_title(model_name, fontsize = 16)
   ax.set_xlabel('r', fontsize = 14)
   ax.set_ylabel('v', fontsize = 14)
   fig.savefig(model_name + '_velocity_error.png')
   plt.show()
   plt.close()


def mass_test():

    print(' *** Tests **** \n')
    print('Difference between numerical and analytical velocities')

    #v200 = 100
    #logc200 = 0.5
    #theta = v200, logc200
    #n = 0.5
    #beta = 20

    #v200 = 100
    #logc200 = 0.5
    #alpha = 4
    #beta = 15

    #theta = v200, logc200, alpha, beta

    v200 = 100
    #logm22 = 0
    alpha = 3
    delta = 1
    #theta = v200, logm22, alpha, delta
    theta = v200, alpha, delta

    #theta = v200, logc200, n, beta

    kwards = {'m22': 1}
    for model in [FDM_fixedmass]:
        r = np.arange(10**(-3), 10, 1)
        v_eq = model(**kwards).velocity(r, theta)
        v_num = [kGrav * ( quad( lambda x: model(**kwards).density(x, theta = theta)*x**2,  0, _r)[0] /_r) ** 0.5 for _r in r]
        print(model.__name__ , 'model :' , error(v_eq, v_num))
        plot_2velocities(r, v_eq, v_num, model.__name__)



if __name__ == "__main__":

    #i = 4 * 3.14 * 1.9 * FDM().I_sol(1)
    #print('I_sol', i)

    #param_compare()
    mass_test()
