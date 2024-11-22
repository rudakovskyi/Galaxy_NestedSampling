

import math
import time
import numpy as np
from scipy.stats import truncnorm
from scipy.integrate import quad
from scipy.special import gamma

import models_sparc as models
from models_sparc import local_inf
from models_sparc import G_N
####################################################
def is_not_zero(array):
    return not all(a == 0 for a in array)
####################################################
def gauss(x, mu, sigma):
    return - 0.5 *( (x - mu) **2 / sigma ** 2 + np.log(sigma**2 * 2 * np.pi) )
####################################################
def modified_gauss(y_pred, y, yerr_up, yerr_down):
    
    norm_const = -0.5 * np.log((np.pi/2)) -\
                   np.log(yerr_down+yerr_up)
    
    return - np.heaviside(y_pred - y, 0) * ((y - y_pred) / yerr_up)**2 +\
            - np.heaviside(y - y_pred, 0) * ((y - y_pred) / yerr_down)**2 - norm_const
####################################################
def check_sqrt(array):
    if np.any(array < 0):
        return [- local_inf for _ in array]
    else:
        return np.sqrt(array)
####################################################
plummer=lambda r,ra: (1+(r/ra)**2)**(-5./2.) #plammer profile with ra half-light radius
####################################################

####################################################
sec_to_rad = 1 / 206265
####################################################

class LikelihoodGenerator:
    
    def __init__(self):
        
        self.luminous_models = {'short' : models.SimpleLuminous, 'full' : models.TotalLuminous, 'exp_thin': models.ExpThin, 'MP21': models.MP21, 'zero' : models.ZeroLuminous, 'MOND_lum' : models.MONDLumionus}
        self.models = {'Burkert' : models.Burkert, 'NFW' : models.NFW, 'coreNFW' : models.coreNFW, 'DC14':models.DC14,'FDM_core' : models.FDM_core, 'FDM' : models.FDM, \
           'FDM_scaled' : models.FDM_scaled, 'FDM_fixedmass' : models.FDM_fixedmass, \
           'MOND' : models.MOND, 'MOND_simple' : models.MOND_simple, 'MOND_standard' : models.MOND_standard}
        self.distributions = {'norm': self.norm, 'lognorm': self.lognormal, 'uniform': self.uniform}
        self.ptforms = {'norm' : self.pf_norm, 'lognorm' : self.pf_lognormal, 'uniform' : self.pf_uniform}

        self.max_calc_time = 10800 # 3 hour in seconds

    def initialize_galaxy(self, galaxy):

        """ Takes as parameter galaxy number in SPARC catalogue or galaxy name. """ 

        with open('SPARC.txt', 'r') as sparc:

            if type(galaxy) == int:
                for i in range(galaxy):
                    sparc.readline()
                line = sparc.readline()
                split_line = line.split()
                self.galaxy_name = split_line[0]
                
            elif type(galaxy) == str:
                self.galaxy_name = galaxy
                line = sparc.readline()
                while not self.galaxy_name in line:
                    line = sparc.readline()
                split_line = line.split()
                
            self.additional_parameters = [float(split_line[2]), float(split_line[3]), float(split_line[5]), float(split_line[6])]

        with open('RC/' + self.galaxy_name + '_rotmod.dat', 'r') as galaxy_file:

            rc_data = galaxy_file.readlines()[3:] 

            self.data = [] 
            for i in range(6):
                self.data.append(np.array([]))
 
            for rc_line in rc_data:
                rc_split_line = rc_line.split()
                for i in range(6):
                    self.data[i] = np.append(self.data[i], float(rc_split_line[i]))

            self.data = tuple(self.data)
            self.bulge = is_not_zero(self.data[5])
            MLums = np.array(self.data[3:6])[:,-1]**2*self.data[0][-1]/G_N
            if not self.bulge:
                MLums = MLums[:2]
            self.MLums = MLums


    def set_model(self, model, lum_model, **kwards):

        """ Set models and related parameters. """

    
        self.model = self.models[model](**kwards)
        self.lum_model = self.luminous_models[lum_model](self.bulge)

        self.ndim = self.model.ndim + self.lum_model.ndim
        self.parameters = self.model.parameters + self.lum_model.parameters

        self.init = self.model.initial(self.data[0][-1],  self.data[1][-1]) + \
            self.lum_model.initial(self.additional_parameters[0], self.additional_parameters[2])

        self.priors = self.model.priors() + self.lum_model.priors()
        self.diapason = self.model.bounds() + self.lum_model.bounds()
        self.sigma = self.model.sigma(self.additional_parameters[1], self.additional_parameters[3]) + self.lum_model.sigma(self.additional_parameters[1], self.additional_parameters[3])

        self.calc_start = time.time()

    # Priors
  
    #def norm(self, n, x):
    #    mu = self.init[n]
    #    sigma = self.sigma[n]
    #    return - 0.5 *( (x - mu) **2 / sigma ** 2 + np.log(sigma**2 * 2 * np.pi) )

    def norm(self, n, x):
        mu = self.init[n]
        sigma = self.sigma[n]

        def f(x):
            return - 0.5 *( (x - mu) **2 / sigma ** 2 + np.log(sigma**2 * 2 * np.pi) )

        norm = quad(lambda x: np.exp(f(x)), self.diapason[n][0], self.diapason[n][1])[0]
        return f(x) -np.log(norm)


    #def lognormal(self, n, x):
    #    log_mu = np.log(self.init[n])
    #    sigma = self.sigma[n]
    #    return -0.5 *( ( np.log(x) - log_mu) **2 / sigma**2 + np.log(x**2 * sigma**2 * 2 * np.pi) )

    def lognormal(self, n, x):
        log_mu = np.log(self.init[n])
        sigma = self.sigma[n]

        def f(x):
            return -0.5 *( ( np.log(x) - log_mu) **2 / sigma**2 + np.log(x**2 * sigma**2 * 2 * np.pi) )

        norm = quad(lambda x: np.exp(f(x)), self.diapason[n][0], self.diapason[n][1])[0]
        return f(x) - np.log(norm)


    def uniform(self, n, x = 0):
        return - np.log(self.diapason[n][1] - self.diapason[n][0])



    # Probabilities

    def log_prior(self, theta):
        for k in range(self.ndim):
            if theta[k] < self.diapason[k][0] or theta[k] > self.diapason[k][1]:
                return -np.inf
        return sum([self.distributions[self.priors[i]](i, theta[i]) for i in range(self.ndim)])

    def total_velocity(self, r, v_gas, v_disk, v_bulge, theta):
        return check_sqrt(self.model.velocity(r, theta, v_gas, v_disk, v_bulge)**2 + self.lum_model.velocity_square(v_gas, v_disk, v_bulge, theta))


    def log_likelihood(self, theta):

        if time.time() > self.calc_start + self.max_calc_time:
            raise TimeoutError

        r, v, err, gas, disk, bulge = self.data
        r_obs, v_obs, err_obs = self.lum_model.rescale(r, v, err, theta)

        model_velocity = self.total_velocity(r_obs, gas, disk, bulge, theta)
        for _mv in model_velocity:
            if not np.isfinite(_mv):
                return - np.inf
        return np.sum(gauss(v_obs, model_velocity, err_obs))

        #return np.sum(gauss(v_obs, self.total_velocity(r_obs, gas, disk, bulge, theta), err_obs))


    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf    
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

        #log_prob = lp + self.log_likelihood(theta)
        #return log_prob 



    # Prior transform

    def bound_trans(self, a):
        if a <= 0:
            return -np.inf
        else:
            return np.log(a)

    def pf_uniform(self, n, u):
        return (self.diapason[n][1]-self.diapason[n][0])*u+self.diapason[n][0]

    def pf_norm(self, n, u):
        m, s = self.init[n], self.sigma[n]  # mean and standard deviation
        low, high = self.diapason[n][0], self.diapason[n][1]  # lower and upper bounds
        low_n, high_n = (low - m) / s, (high - m) / s  # standardize
        return  truncnorm.ppf(u, low_n, high_n, loc=m, scale=s)

    def pf_lognormal(self, n, u):
        m, s = np.log(self.init[n]), self.sigma[n]  # mean and standard deviation
        low, high = self.bound_trans(self.diapason[n][0]), self.bound_trans(self.diapason[n][1])
        low_n, high_n = (low - m) / s, (high - m) / s  # standardize
        x = truncnorm.ppf(u, low_n, high_n, loc=m, scale=s)
        return np.exp(x)

    def ptform(self, u):
        return [self.ptforms[self.priors[i]](i, u[i]) for i in range(self.ndim)]


class LikelihoodGeneratorDSph(LikelihoodGenerator):

    def initialize_galaxy(self, galaxy):
        
        self.galaxy_name = galaxy
        data_path = f'dSphs/{galaxy}.dat'
        dat=np.loadtxt(data_path, usecols=(0,1,4)).T
        self.data = dat
 
    def set_model(self, model, **kwards):

        """ Set models and related parameters. """
        
        self.model = self.models[model](**kwards)
    #       self.lum_model = self.luminous_models[lum_model](self.bulge)
        kwards.setdefault('lum_dim', 2)
        kwards.setdefault('priors', ['uniform', 'norm'])
        kwards.setdefault('parameters', [r'$\beta_a$', r'$r_h$'])
        kwards.setdefault('bounds', [[-9.,1.0],[0,3]])
        kwards.setdefault('sigma', [0, 0.2])
        kwards.setdefault('initial', [0, 0.2])
    #       self.ndim = self.model.ndim + self.lum_model.ndim
        self.ndim = self.model.ndim + kwards['lum_dim']
        self.parameters = self.model.parameters + kwards['parameters']

        self.init = self.model.initial(self.data[0][-1],  self.data[1][-1]) + \
            kwards['initial']
        self.priors = self.model.priors() + kwards['priors']
        self.diapason = self.model.bounds() + kwards['bounds']
        self.sigma = self.model.sigma() + kwards['sigma']

        self.calc_start = time.time()


    def log_likelihood(self, theta):

        # if time.time() > self.calc_start + self.max_calc_time:
        #     raise TimeoutError

        r_obs, sigma_obs, err_obs = self.data
        

        slos = self.model.sigma_los(theta)
        model_sigma = slos(r_obs)
        # for _mv in model_velocity:
        #     if not np.isfinite(_mv):
        #         return - np.inf
        return np.sum(gauss(sigma_obs, model_sigma, err_obs))
    
class LikelihoodGeneratorLT(LikelihoodGenerator):


    def initialize_galaxy(self, galaxy):
        
        self.galaxy_name = galaxy
        data_path = f'finalrot/{self.galaxy_name}_onlinetab.txt'
        dat = np.genfromtxt(data_path, skip_header=4, skip_footer=-7, usecols=(1,6,7)).T

        self.data = dat

    def set_model(self, model, **kwards):

        """ Set models and related parameters. """
        self.model = self.models[model](**kwards)
    #       self.lum_model = self.luminous_models[lum_model](self.bulge)
        kwards.setdefault('lum_model', 'exp_thin')

        kwards.setdefault('lum_dim', 2)
        kwards.setdefault('priors', ['lognorm', 'norm'])
        kwards.setdefault('lum_parameters', ['log_Mstar', 'i'])
        kwards.setdefault('bounds', [[4.,10],[0,90]])
        kwards.setdefault('sigma', [0.2, 2])
        kwards.setdefault('initial', [7, 90])
        kwards.setdefault('Rgas', 1)
        kwards.setdefault('Rstar', 1)
        kwards.setdefault('log_Mgas', 8)

        self.lum_model = self.luminous_models[kwards['lum_model']]()

        self.fixed_pars_names = ['Rstar', 'log_Mgas','Rgas']
        self.fixed_pars = {par_name:kwards[par_name] for par_name in self.fixed_pars_names if par_name in kwards}


    #       self.ndim = self.model.ndim + self.lum_model.ndim
        self.ndim = self.model.ndim + kwards['lum_dim']
        self.lum_parameters = kwards['lum_parameters']
        self.parameters = self.model.parameters + self.lum_parameters
        
        self.init = self.model.initial(self.data[0][-1],  self.data[1][-1]) + \
            kwards['initial']
        self.priors = self.model.priors() + kwards['priors']
        self.diapason = self.model.bounds() + kwards['bounds']
        self.sigma = self.model.sigma() + kwards['sigma']

        self.calc_start = time.time()
    
    def total_velocity(self, r, theta):
        
        
        params = dict(zip(self.lum_parameters, theta[self.model.ndim:]))

        params.update(self.fixed_pars)
        log_Mstar =  params['log_Mstar']
        Rstar = params['Rstar']

        log_Mgas =  params['log_Mgas']
        Rgas = params['Rgas']

        
        v_dm = self.model.velocity(r, theta)
        v2_dm = v_dm ** 2
        v2_gas = self.lum_model.velocity_square(r=r, log_M=log_Mgas, R=Rgas)
        v2_star = self.lum_model.velocity_square(r=r, log_M=log_Mstar, R=Rstar)
        return np.sqrt(v2_dm + v2_gas + v2_star)

    def log_likelihood(self, theta):

        # if time.time() > self.calc_start + self.max_calc_time:
        #     raise TimeoutError
        i = theta[-1]
        i0 = self.init[-1]
        r, v, err = self.data
        v_obs, err_obs = self.lum_model.rescale(v, err, i0, i)
        

        v_pred = self.total_velocity(r, theta)
        # for _mv in model_velocity:
        #     if not np.isfinite(_mv):
        #         return - np.inf
        return np.sum(gauss(v_obs, v_pred, err_obs))
    
############################################

class LikelihoodGeneratorMP21(LikelihoodGenerator):


    def initialize_galaxy(self, galaxy):
        
        self.galaxy_name = galaxy.lower()
        data_path = f'final_sample/{self.galaxy_name}tab.dat'
        dat = np.genfromtxt(data_path, skip_header=1, usecols=(0,5,6,7,8)).T

        self.data = dat
        with open('MP21_stellar_pars.txt', 'r') as mp21:
    
            line = mp21.readline()
            while not self.galaxy_name in line:
                line = mp21.readline()
            split_line = line.split()
            D = np.float64(split_line[1]) # kpc
            self.Rstar_mean = sec_to_rad * np.float64(split_line[2]) * D #kpc
            self.Rstar_err = sec_to_rad * np.float64(split_line[3]) * D #kpc
            self.log10_Mstar_mean = np.float64(split_line[4])
            self.log10_Mstar_err = np.float64(split_line[5])
            self.n_mean = np.float64(split_line[6])
            self.n_err = np.float64(split_line[7])
    
    
    def set_model(self, model, **kwards):

        """ Set models and related parameters. """

        kwards.setdefault('lum_model', 'MP21')
        kwards.setdefault('fixed_pars', {})

        self.model = self.models[model](**kwards)


        self.fixed_pars_names = list(kwards['fixed_pars'].keys())

        self.lum_model = self.luminous_models[kwards['lum_model']]()

        self.fixed_pars = kwards['fixed_pars']


        _param_ind = [ind for ind, param_name in enumerate(self.lum_model.parameters) if param_name not in self.fixed_pars_names]

        self.lum_parameters = [self.lum_model.parameters[i] for i in _param_ind]

        self.ndim = self.model.ndim + len(self.lum_parameters)

        self.parameters = self.model.parameters + self.lum_parameters
        

        
        self.init = self.model.initial(self.data[0][-1],  self.data[1][-1]) + \
            [self.lum_model.initial([self.log10_Mstar_mean, self.Rstar_mean, self.n_mean])[i] for i in _param_ind]
        

        self.priors = self.model.priors() + [self.lum_model.priors[i] for i in _param_ind]
        self.diapason = self.model.bounds() + [self.lum_model.bounds[i] for i in _param_ind]
        self.sigma = self.model.sigma() + [self.lum_model.sigma([self.log10_Mstar_err, self.Rstar_err, self.n_err])[i] for i in _param_ind]

        self.calc_start = time.time()


    def total_velocity(self, r, theta):
        
        
        params = dict(zip(self.lum_parameters, theta[self.model.ndim:]))

        params.update(self.fixed_pars)
        log_Mstar =  params['log_Mstar']
        Rstar = params['Rstar']
        n = params['n']

        Mstar = 10**log_Mstar


        sigma0_star = n / (2 * np.pi) * Mstar / gamma(2/n) / Rstar ** 2
       
        log_sigma0_gas, R1gas, R2gas, alpha = params['log_sigma0_gas'], params['R1gas'], params['R2gas'], params['alpha']
        
        v_dm = self.model.velocity(r, theta)
 
        v2_dm = v_dm ** 2
        
        
        sigma0_gas = 10 ** log_sigma0_gas 
        
        v2_baryons = self.lum_model.velocity_square(r, sigma0_star, Rstar, n, sigma0_gas, R1gas, R2gas, alpha)

        return np.sqrt(v2_dm + v2_baryons)

    def sigma_gas(self, r, theta):


        params = dict(zip(self.lum_parameters, theta[self.model.ndim:]))
        params.update(self.fixed_pars)
        log_sigma0_gas, R1gas, R2gas, alpha = params['log_sigma0_gas'], params['R1gas'], params['R2gas'], params['alpha']
        sigma0_gas = 10 ** log_sigma0_gas 
        return self.lum_model.surf_density_gas(r, sigma0_gas, R1gas, R2gas, alpha)[:,1]
    

    def log_likelihood(self, theta):
        
        if time.time() > self.calc_start + self.max_calc_time:
            raise TimeoutError
        # params = dict(zip(self.lum_parameters, theta[self.model.ndim:]))
        # params.update(self.fixed_pars)
        r, sigma_obs_gas, sigma_err, v, v_err = self.data

        v_pred = self.total_velocity(r, theta)

        sigma_pred = self.sigma_gas(r, theta) / 1e6 #  from Msun/kpc^2 -- > Msun/pc^2

        return float(np.sum(gauss(v, v_pred, v_err)) + np.sum(gauss(sigma_obs_gas, sigma_pred, sigma_err)))

# -------------------------------------------------

#               ****    Tests   ****

# -------------------------------------------------


def normalization_test():
    
    # priors

    LG = LikelihoodGenerator()
    priors = [LG.uniform, LG.norm]

    # uniform
    lim = 10**9
    LG.diapason = [[0, lim]]

    f = LG.uniform
    norm = quad(lambda x: np.exp(f(0, x)), 0, lim)[0]
    print(f.__name__, norm)

    # normal 
    # meanwhile of  bounds, due to large parameter range compared to sigma, normalization is very close to 1
    lim = 10**3

    LG.init = [70]
    LG.sigma = [5]
    LG.diapason = [[0, lim]]
    
    f = LG.norm
    norm = quad(lambda x: np.exp(f(0, x)), 0, lim)[0]
    print(f.__name__, norm)

    #lognormal
    # here is problem due to narrow bounds
    lim = 1.2

    LG.init = [0.5]
    LG.sigma = [10 ** 0.1]
    LG.diapason = [[0, lim]]
    
    f = LG.lognormal
    norm = quad(lambda x: np.exp(f(0, x)), 0, lim)[0]
    print(f.__name__, norm)


    # likelihood (gauss)
    lim = 10**3
    mu = 50
    sigma = 5
    norm = quad(lambda x: np.exp(gauss(x, mu, sigma)), -lim, lim)
    print('gauss: ', norm)

if __name__ == "__main__":  
    normalization_test()
