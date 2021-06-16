

import math
import numpy as np
from scipy.stats import truncnorm
from scipy.integrate import quad

import models_vc200 as models



def is_not_zero(array):
    return not all(a == 0 for a in array)

def gauss(x, mu, sigma):
    return - 0.5 *( (x - mu) **2 / sigma ** 2 + np.log(sigma**2 * 2 * np.pi) )



class LikelihoodGenerator:
    
    def __init__(self):
        
        self.luminous_models = {'short' : models.SimpleLuminous, 'full' : models.TotalLuminous}
        self.models = {'Burkert' : models.Burkert, 'NFW' : models.NFW, 'FDM_core' : models.FDM_core, 'FDM' : models.FDM, 'FDM_scaled' : models.FDM_scaled}
        self.distributions = {'norm': self.norm, 'lognorm': self.lognormal, 'uniform': self.uniform}
        self.ptforms = {'norm' : self.pf_norm, 'lognorm' : self.pf_lognormal, 'uniform' : self.pf_uniform}


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
            self.buldge = is_not_zero(self.data[5])


    def set_model(self, model, lum_model):

        """ Set models and related parameters. """

        self.model = self.models[model]()
        self.lum_model = self.luminous_models[lum_model](self.buldge)

        self.ndim = self.model.ndim + self.lum_model.ndim
        self.parameters = self.model.parameters + self.lum_model.parameters

        self.init = self.model.initial(self.data[0][-1],  self.data[1][-1]) + \
            self.lum_model.initial(self.additional_parameters[0], self.additional_parameters[2])

        self.priors = self.model.priors() + self.lum_model.priors()
        self.diapason = self.model.bounds() + self.lum_model.bounds()
        self.sigma = self.model.sigma() + self.lum_model.sigma(self.additional_parameters[1], self.additional_parameters[3])

        

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

    def total_velocity(self, r, v_gas, v_disk, v_buldge, theta):
        return np.sqrt(self.model.velocity(r, theta)**2 + self.lum_model.velocity(v_gas, v_disk, v_buldge, theta)**2)


    def log_likelihood(self, theta):
        r, v, err, gas, disk, buldge = self.data
        r_obs, v_obs, err_obs = self.lum_model.rescale(r, v, err, theta)
        return np.sum(gauss(v_obs, self.total_velocity(r_obs, gas, disk, buldge, theta), err_obs))


    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf             
        log_prob = lp + self.log_likelihood(theta)
        return log_prob 



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