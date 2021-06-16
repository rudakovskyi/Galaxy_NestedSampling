
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

import pickle
import dynesty
from dynesty import plotting as dyplot

from likelihoodgenerator import LikelihoodGenerator




if __name__ == "__main__":

    #i = 'IC2574'
    #bulge_galaxies = ['IC4202', 'NGC0891']
    
    for i in range(10, 11):
    
        LGenerator = LikelihoodGenerator()
        LGenerator.initialize_galaxy(i) # read one gelaxy data from the file 
        galaxy_name = LGenerator.galaxy_name

        lum_model = 'full'
        #for model in ['Burkert', 'NFW', 'FDM']:
        for model in ['Burkert']:

            print(galaxy_name, model)

            LGenerator.set_model(model, lum_model)
    
            nlive = 300

            processes = 1
            pool = None
            if True:
            #with Pool(processes=processes) as pool:
                dsampler = dynesty.DynamicNestedSampler(LGenerator.log_likelihood, LGenerator.ptform, ndim = LGenerator.ndim, nlive = nlive, pool = pool, queue_size = processes)
                dsampler.run_nested(print_progress=True)
            dresults = dsampler.results
     
            save = True
            plot = True

            if save:
                fout='{}/dresults_{}_{}.pkl'.format(model, galaxy_name, lum_model)
                with open(fout, 'wb') as fn:
                    pickle.dump(dresults, fn)

                if plot:          
                    fig, ax = dyplot.cornerplot(dresults, labels=LGenerator.parameters)  
                    fig.savefig('{}/corner_plot/{}_NS_{}.png'.format(model, galaxy_name, lum_model))
                    plt.close()
    
            logz = dresults.get(['logz'][-1])[-1]
            logzerr = dresults.get(['logzerr'][-1])[-1]

            with open(model + '/results.txt', 'a') as fout:
                fout.write('{} \t {:.3f} \t {:.3f} \n'.format(galaxy_name, logz, logzerr))


