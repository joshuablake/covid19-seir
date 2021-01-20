from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
from joblib import Parallel, delayed
import multiprocessing
import time as timer
import numpy as np
import scipy.stats as stats
import pints

class PintsLogPDF(pints.LogPDF):
    def __init__(self, regional_logPs_list, global_size=12, regional_size=15):
        self._regional_logPs_list = regional_logPs_list  
        self._regions = len(self._regional_logPs_list)   
        self._global_size = global_size
        self._regional_size = regional_size

        self.num_cores = multiprocessing.cpu_count() - 1
        print('Using total threads: ', self.num_cores)

    def global_priors(self, global_parameters):
        #assert len(global_parameters) == self.n_params_global
        dI = global_parameters[0]
        ksens, kspec, eta = global_parameters[1:4]
        p_dead = global_parameters[4:11]  
        beta_rw_sd =  global_parameters[11]             

        # infectious_period 
        logPrior_dI = stats.gamma.logpdf(dI, a = 1.43, scale = 1/0.549)# add 1.43,
        # sero sensitivity
        logPrior_ksens = stats.beta.logpdf(ksens, 23.5, 9.5)# 23.5, 9.5
        # sero specificity
        logPrior_kspec = stats.beta.logpdf(kspec, 569.5, 5.5) #569.5, 5.5
        # negbin_overdispersion
        logPrior_eta = stats.gamma.logpdf(eta, a = 1.0, scale = 1/0.2)# rate parameterisation 

        # prop_of_death (IFR)
        p_dead_loc = np.array([1., 1., 1., 1., 1., 1., 9.5])
        p_dead_scale = np.array([62110.801242236, 23363.4859813084, \
            5290.00529100529, 1107.64745011086, 120.951219512195, \
            31.1543408360129, 112.])
        
        logPrior_p_dead = np.sum(stats.beta.logpdf(p_dead, p_dead_loc, p_dead_scale))        
        # random-walk sd
        logPrior_beta_rw_sd = stats.gamma.logpdf(beta_rw_sd, a = 1.0, scale = 1/100)#WIP was 1/100
        #logPrior_beta_rw_sd = stats.invgamma.logpdf(beta_rw_sd,a=5.0,scale=5.0)                             

        logGlobPrior = logPrior_dI + logPrior_ksens + logPrior_kspec + \
            logPrior_eta + logPrior_p_dead + logPrior_beta_rw_sd    
        return logGlobPrior

    def ParallelLogP(self, thr, glob,local):
        local_glob = np.hstack((glob,local[thr,:]))
        return self._regional_logPs_list[thr](local_glob,joint_reg=True)

    def __call__(self, x):

        with Parallel(n_jobs=self.num_cores) as parallel:
            allreg_x = x[self._global_size:].reshape((self._regions,\
                            self._regional_size))
            current_log_targets = parallel(
                delayed(self.ParallelLogP)(thread,x[:self._global_size],
                allreg_x) for thread in range(self._regions))
        current_log_target = np.sum(current_log_targets) + \
            self.global_priors(x[:self._global_size])
        return current_log_target        

    def n_parameters(self):
        return self._regional_size*self._regions + self._global_size