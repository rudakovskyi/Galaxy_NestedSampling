

import math
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import scipy.optimize

# -------------------------------------------
#           ***  Dimentions  ***
# -------------------------------------------

# r -- kpc
# v -- km/s
# M -- M_Sun
# rho -- M_Sun / pc^3
# H0 -- km/s/kpc

H0 = 0.073 # H0 = 73 km/s/Mpc

rho_crit = 1.47741 * 10**(-7) # critical density of the Universe, M_Sun / pc^3 

# v(r) = kGrav * sqrt( (M(r)/4pi) /r )
# kGrav = sqrt[ 4 * pi * G_N * (M_Sun/kg) * (kpc/pc)^3 / (kpc/m)]  / (km/m)
# G_N = 6.67 * 10^-11 m^3 / kg / s^2

kGrav = 232.652 




# --------------------------------------------
#        ***  Dark matter models  ***
# --------------------------------------------



class Model:
    
    def __init__(self):
        self.ndim = 2
        self.parameters = ['$v_{200}$', '$c_{200}$']

    def initial(self, r_inf, v_inf):
        v200_init = v_inf
        c200_init = 1 #5
        return [v200_init, c200_init]  

    def bounds(self):
        return [[10, 500], [0, 1000]]
        #return [[10, 500], [0.01, 1000]]

    def priors(self):
        return ['uniform', 'uniform']

    def sigma(self):
        return [0, 0]
   

    def inverse_transformation(self, theta):

        v200, c200 = theta[:self.ndim]
        r0 = v200 / (10 * H0 * c200)
        rho0 = 200 * rho_crit * c200**3 / (3 * self.g(c200, theta))       
        return r0, rho0


    def direct_transformation(self, theta):

        r0, rho0 = theta[:self.ndim]
        theta0 = [None, None]
   
        def eq(x):
            return rho0 * self.g(x, theta0) - 200 * rho_crit * x**3 / 3

        #x = np.arange(10**-3, 50, 10**-3)
        #y = np.array([eq(_x) for _x in x])
        #z = np.array([0 for _x in x])
        #plt.plot(x, y)
        #plt.plot(x, z)
        #plt.show()

        c200 = scipy.optimize.broyden1(eq, [100], f_tol=1e-6)[0] 
        # the equation has several solutions(2), large initial value allows to find correct solution
        v200 = 10 * H0 * c200 * r0
        
        return v200, c200




class Burkert(Model):

    def g(self, x, theta = None):
        return ((np.log(1 + x**2))/4 + np.log(1 + x)/2 - np.arctan(x)/2) / x

    def velocity(self, r, theta):
        v200, c200 = theta[:self.ndim]
        x = r * 10 * H0 * c200 / v200
        return  v200 * (self.g(x) / self.g(c200)) ** 0.5

    def density(self, r, theta):
        r0, rho0 = self.inverse_transformation(theta)
        return ( rho0 / ( 1 + (r/r0)) ) / (1 + (r/r0)**2) 




class NFW(Model):

    def g(self, x, theta = None):
        return (np.log(1 + x) - x / (1 + x)) / x

    def velocity(self, r, theta):
        v200, c200 = theta[:self.ndim]
        x = r * 10 * H0 * c200 / v200 
        return v200 * (self.g(x) / self.g(c200)) ** 0.5

    def density(self, r, theta):
        r0, rho0 = self.inverse_transformation(theta)
        return rho0 / ((r/r0) * (1 + r/r0) ** 2)




class FDM_core(Model):

    def f_sol(self, x):
        return 1 / (1 + 0.091*x**2)**8
  
    def I_sol(self, x):
        a = 9.1 / 100
        return (np.sqrt(a) * x * ( -3465 + 48580 *a * x**2 + 92323 * a**2 * x**4 + 101376 * a**3 * x**6 +  65373 * a**4 * x**8 + 23100 * a**5 * x**10 + 3465 * a**6 * x**12) / (1 + a * x**2)**7 + \
            3465* np.arctan(np.sqrt(a) * x)) / (215040 * a**(3/2))

    def g(self, x, theta = None):
        return self.I_sol(x) / x

    def velocity(self, r, theta):
        v200, c200 = theta[:self.ndim]
        x = r * 10 * H0 * c200 / v200
        return v200 * (self.g(x, theta) / self.g(c200, theta)) ** 0.5

    def density(self, r, theta):
        r0, rho0 = self.inverse_transformation(theta)
        return rho0 / (1 + 0.091 * (r/r0)**2)**8



class FDM(Model):
    
    def __init__(self):
        self.ndim = 4
        self.parameters = ['$v_{200}$', '$c_{200}$', r'$\alpha$', r'$\beta$']
        # alpha = ra/rs     beta = rn/rs

    def initial(self, r_inf, v_inf):
        v200_init = v_inf  
        alpha_init = 3
        c200_init = 10
        beta_init = 5
        return [v200_init, c200_init, alpha_init, beta_init]

    def bounds(self):
        #return [[10, 500], [7, 1000], [1, 7], [1, 1000]]
        return [[10, 500], [0, 1000], [1, 7], [1, 1000]]

    def priors(self):
        return ['uniform' for _ in range(self.ndim)]

    def sigma(self):
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

    def g(self, x, theta):
        v200, c200, alpha, beta = theta[:4]
        K = self.f_sol(alpha) * beta**3 / ( self.f_tail(alpha / beta))
        return (self.I_sol(min(alpha, x)) + K * self.I_tail(alpha / beta, x / beta) ) / x 


    def velocity(self, r, theta):
        v200, c200, alpha, beta = theta[:4]
        x = r * 10 * H0 * c200 / v200
        return  np.array([v200 * (self.g(z, theta) / self.g(c200, theta)) ** 0.5 for z in x])


    def inverse_transformation(self, theta):
        v200, c200, alpha, beta = theta[:4]
        rs = v200 / (10 * H0 * c200)
        rhos = 200 * rho_crit * c200**3 / (3 * self.g(c200, theta))
        rNFW = beta * rs      
        return rs, rhos, alpha, rNFW


    def direct_transformation(self, theta):
        rs, rhos, alpha, rNFW = theta[:4]
        beta = rNFW / rs
        theta0 = [None, None, alpha, beta]
 
        def eq(x):
            return rhos * self.g(x, theta0) - 200 * rho_crit * x**3 / 3

        #x = np.arange(10**-3, 10, 10**-3)
        #y = np.array([eq(_x) for _x in x])
        #z = np.array([0 for _x in x])
        #plt.plot(x, y)
        #plt.plot(x, z)
        #plt.show()

        c200 = scipy.optimize.broyden1(eq, [100], f_tol=1e-6)[0]
        v200 = 10 * H0 * c200 * rs       
        return v200, c200, alpha, beta


    def density(self, r, theta):

        v200, c200, alpha, beta = theta[:4]
        rs, rhos, alpha, rNFW = self.inverse_transformation(theta)
        ra = rs * alpha
        if r < ra:
            return rhos/(1 + 0.091 * (r/rs)**2)**8
        else:
            rhoNFW =  (rhos / (1 + 0.091 * (ra/rs)**2)**8 ) * (ra/rNFW) * (1 + ra/rNFW)**2 
            return rhoNFW/ (r/rNFW) / (1 + r/rNFW)**2 




class FDM_scaled(FDM):

    def __init__(self):
        self.ndim = 3
        self.parameters = ['$v_{200}$', '$m_{22}$', r'$\alpha$']


    def initial(self, r_inf, v_inf):
        v200_init = v_inf  
        m22_init = 1
        alpha_init = 3
        return [v200_init, m22_init, alpha_init]


    def bounds(self):
        return [[10, 500], [0.01, 100], [1, 7]]


    def scaling(self, v200, m22, alpha):

        #c200 = v200**2 * m22 / (4.36 * 1000 * H0 * (10*H0*G_N)**(1/3))
        c200 = v200**2 * m22 * rho_crit**(1/3) / (4.63 * 1000 * H0**2 )

        print('c200:', c200)

        def eq(b):
            theta = [v200, c200, alpha, b]
            #return 200 * rho_crit / 3 * self.g(c200, theta) - 1.9 * H0**4 * c200 / ( m22**2 * v200**4)
            #return rho_crit / self.g(c200, theta) - 2.85 * c200 * H0**4 / (m22**2 * v200**4)
            #return 2.85 * self.g(c200, theta) - rho_crit * m22**2 * v200**4 / (c200 * H0**4)
            return 2.85 * self.g(c200, theta)


        x = np.arange(1, 10**3, 1)
        y = np.array([eq(_x) for _x in x])
        #z = np.array([0 for _x in x])
        plt.plot(x, y)
        #plt.plot(x, z)
        plt.show()

        beta = scipy.optimize.broyden1(eq, [100000], f_tol=1e-6)[0]
        print('beta', beta)
        return c200, beta


    def original_param(self, theta):

        v200, m22, alpha = theta[:self.ndim]
        c200, beta = self.scaling(v200, m22, alpha)
        theta0 = v200, c200, alpha, beta

        return theta0

    #def scalingsB(self, alpha = 3, beta = 100):

    #    c200 = v200**2 * m22 / (4.36 * 1000 * H0 * (10*H0*G)**(1/3))
   
    #    #def eq(b):
    #    #    theta = [None, c200, alpha, b]
    #    #    return 200 * rho_crit / 3*self.g(c200, theta) - 1.9 * H0**4 * c200 / ( m22**2 * v200**4)

    #    #beta = scipy.optimize.broyden1(eq, [100], f_tol=1e-6)[0]
    #    return v200, c200


    def velocity(self, r, theta):

        #v200, m22, alpha = theta[:self.ndim]
        #c200, beta = self.scaling(v200, m22, alpha)
        #thata0 = v200, c200, alpha, beta

        theta0 = self.original_param(theta)

        #x = r * 10 * H0 * c200 / v200
        #return  np.array([v200 * (self.g(z, theta0) / self.g(c200, theta0)) ** 0.5 for z in x])

        return super().velocity(r, theta0)


    def density(self, r, theta):

        #rs, m22, alpha = self.inverse_transformation(theta)
        
        #rhos = 1.9*0.01 / (m22**2 * rs**4)
        #beta = 1 # ?

        #theta0 = rs, rhos, alpha, beta
        #return super().density(r, theta0)

        #v200, m22, alpha = theta[:self.ndim]
        #c200, beta = self.scaling(theta)
        #theta0 = v200, c200, alpha, beta

        theta0 = self.original_param(theta)

        return super().density(r, theta0)

        #rs, rhos, alpha, rNFW = super().inverse_transformation(theta0)

        #assert(rhos == 1.9*0.01 / (m22**2 * rs**4))

        #ra = rs * alpha

        #if r < ra:
        #    return rhos/(1 + 0.091 * (r/rs)**2)**8
        #else:
        #    rhoNFW =  (rhos / (1 + 0.091 * (ra/rs)**2)**8 ) * (ra/rNFW) * (1 + ra/rNFW)**2 
        #    return rhoNFW/ (r/rNFW) / (1 + r/rNFW)**2 


    def inverse_transformation(self, theta):

        v200, m22, alpha = theta[:self.ndim]       
        #c200, beta = self.scaling(v200, m22, alpha)
        #theta0 = v200, c200, alpha, beta

        theta0 = self.original_param(theta)

        rs, rhos, alpha, rNFW = super().inverse_transformation(theta0)

        print('assert rhos :', rhos, 1.9*0.01 / (m22**2 * rs**4))
        assert(rhos == 1.9*0.01 / (m22**2 * rs**4))

        return rs, m22, alpha


    def direct_transformation(self, theta):

        rs, m22, alpha = theta[:self.ndim]
        #rhos = 1.9*0.01 / (m22**2 * rs**4)

        c200 = 46.3 / (H0 * rho_crit**(1/3) * rs**2 * m22)

        #beta = rNFW / rs
        #theta0 = [None, None, alpha, beta]
   
        #def eq(x):
        #    return rhos * self.g(x, theta0) - 200 * rho_crit * x**3 / 3

        #x = np.arange(10**-3, 10, 10**-3)
        #y = np.array([eq(_x) for _x in x])
        #z = np.array([0 for _x in x])
        #plt.plot(x, y)
        #plt.plot(x, z)
        #plt.show()

        #c200 = scipy.optimize.broyden1(eq, [100], f_tol=1e-6)[0]
        v200 = 10 * H0 * c200 * rs
        
        return v200, m22, alpha




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
        s = 10 ** 0.1
        if self.bulge:
            return [s, s]
        return [s]

    def velocity(self, v_gas, v_disk, v_bulge, theta):
        if self.bulge:
            g_disk, g_bulge = theta[-self.ndim:]
            return (v_gas**2 + g_disk * v_disk**2 + g_bulge * v_bulge**2)**0.5
        g_disk = theta[-1]
        return (v_gas**2 + g_disk * v_disk**2)**0.5 

    def rescale(self, r, v, err, theta):
        return r, v, err



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

    def velocity(self, v_gas, v_disk, v_bulge, theta):
        if self.bulge:
            D, i, g_disk, g_bulge = theta[-self.ndim:]
            return ((v_gas**2 + g_disk * v_disk **2 + g_bulge * v_bulge**2) * D / self.D) **0.5
        D, i, g_disk = theta[-self.ndim:]
        return ((v_gas**2 + g_disk * v_disk**2) * D / self.D)**0.5

    def rescale(self, r, v, err, theta):
        if self.bulge:
            D, i, g_disk, g_bulge = theta[-self.ndim:]
        else:
            D, i, g_disk = theta[-self.ndim:]

        v_obs = v * np.sin(np.deg2rad(self.i)) / np.sin(np.deg2rad(i))
        err_obs = err * np.sin(np.deg2rad(self.i)) / np.sin(np.deg2rad(i))
        r_obs = r * D / self.D

        return r_obs, v_obs, err_obs



# ----------------------------------------------------------------

#                ****   Profile tests    ****

# ----------------------------------------------------------------


def param_compare():
    
    rs = 1.5
    alpha = 3

    #rhos = np.exp(-2.52)
    #rNFW = 50
    m22 = 1

    model = FDM_scaled

    print('Direct + Inverse')

    theta = rs, m22, alpha
    #theta = rs, rhos, alpha, rNFW

    theta0 = model().direct_transformation(theta)
    theta1 = model().inverse_transformation(theta0)

    print('r0, rho0 : ', theta)
    print('v200, c200: ', theta0)
    print('r0, rho0: ', theta1)

    print()
    print('Inverse + Direct')

    v200 = 85.83
    alpha = 2
    #c200 = 7.49
    #beta = 20
    m22 = 1

    #theta = v200, c200, alpha, beta
    theta = v200, m22, alpha
    theta0 = model().inverse_transformation(theta)
    theta1 = model().direct_transformation(theta0)

    print('v200, c200 : ', theta)
    print('r0, rho0 : ', theta0)
    print('v200, c200 : ', theta1)


def error(func1, func2):
    if len(func1) != len(func2):
        return -9999
    return sum([abs(func1[i] - func2[i]) for i in range(len(func1))]) / len(func1)


def plot_2velocities(r, v1, v2, model_name):
   fig, ax = plt.subplots()
   ax.plot(r, v2, label = 'numerical')
   ax.plot(r, v1, label = 'analytical')
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

    #theta = [2, 0.01, 3, 10]
    #theta = [5, 0.2, 7, 150]

    v200 = 100
    #c200 = 1
    alpha = 3
    #beta = 20
    m22 = 1

    theta = [v200, m22, alpha]
    #theta = [v200, c200, alpha, beta]
    r = np.arange(10**-5, 10, 10**-3)

    for model in [FDM_scaled]:
        v_eq = model().velocity(r, theta)
        v_num = [kGrav * ( quad( lambda x: model().density(x, theta = theta)*x**2,  0, _r )[0] /_r) ** 0.5 for _r in r]

        print(model.__name__ , 'model :' , error(v_eq, v_num))
        plot_2velocities(r, v_eq, v_num, model.__name__)




if __name__ == "__main__":

    #mass_test()
    param_compare()
