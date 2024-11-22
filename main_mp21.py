

import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import sys
from io import StringIO

import os
import pickle
import dynesty
from dynesty import plotting as dyplot
from scipy.stats import gaussian_kde
from scipy.optimize import fsolve, minimize, broyden1
import galpynamics.dynamic_component as dc

from likelihoodgenerator import LikelihoodGeneratorMP21

import warnings
warnings.filterwarnings("ignore")

def create_folders(models):

    for model in models:
        try:
            os.mkdir(model)
        except FileExistsError:
            pass

        try:
            os.mkdir('{}/corner_plot'.format(model))
        except FileExistsError:
            pass
        
        

save = True
plot = True


galaxy_list = ['LVHIS009','LVHIS012','LVHIS020','LVHIS026','LVHIS029','LVHIS060','LVHIS065','LVHIS072','LVHIS078','LVHIS080']

models = [ 'Burkert', 'FDM_scaled', 'NFW']

if __name__ == "__main__":

    for model in models[2:]:
        create_folders([model])
        for galaxy in galaxy_list:

            file_name = galaxy


            LGenerator = LikelihoodGeneratorMP21()
            LGenerator.initialize_galaxy(galaxy)

            LGenerator.set_model(model, lum_model='MP21')
            processes = 24
            nlive = 500
            with Pool(processes=processes) as pool:
                try:
#                    dsampler = dynesty.NestedSampler(LGenerator.log_likelihood, LGenerator.ptform, ndim = LGenerator.ndim, nlive = nlive, pool = pool, queue_size = processes)
                    dsampler = dynesty.DynamicNestedSampler(LGenerator.log_likelihood, LGenerator.ptform, ndim = LGenerator.ndim, nlive = nlive, pool = pool, queue_size = processes)

                    dsampler.run_nested(maxiter_init=30000, maxiter_batch=10000, maxbatch=10, print_progress = True)
                    
                    
#                except TimeoutError:
                except Exception:
                    file_name += '_fail'

            dresults = dsampler.results
            logz = dresults.get(['logz'][-1])[-1]
            logzerr = dresults.get(['logzerr'][-1])[-1]

            with open(model + '/results.txt', 'a') as fout:
                fout.write('{} \t {:.3f} \t {:.3f} \n'.format(galaxy, logz, logzerr))


            save = True
            plot = True

            if save:

                fout='{}/dresults_{}.pkl'.format(model, file_name)

                with open(fout, 'wb') as fn:
                    pickle.dump(dresults, fn)

            #                 fout_dsampler='{}/dsampler_{}_{}.pkl'.format(model, file_name, lum_model)

            #                 with open(fout_dsampler, 'wb') as fn:
            #                     pickle.dump(dsampler, fn)
            if plot:          
                fig, ax = dyplot.cornerplot(dresults, labels=LGenerator.parameters, color='blue', show_titles=True, smooth = 0.05)  
                fig.savefig('{}/corner_plot/{}.png'.format(model, file_name))
                plt.close() 
