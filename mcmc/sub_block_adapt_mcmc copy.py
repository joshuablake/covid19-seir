from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import scipy.stats as stats
import time
from joblib import Parallel, delayed
import multiprocessing
import time as timer

class SubBlockRegionalAMGS(object):
    def __init__(self, target, x0, mean_est, iterations=150000):

        self._target = target
        self._iterations = iterations
        self._x0 = x0
        self._regions = int(7)
        self._sub_blocks = int(2)
        self._global_size = int(12)
        self._regional_size = int(15)

        # This iterator churns out start index of each regional block of parameters
        self.loop_iterator = \
            range(self._global_size, ((self._regional_size*self._regions) + \
            self._global_size),self._regional_size)
        self.region_starts = np.array([i for i in self.loop_iterator])
        self._verbose = True

        # One scale for each region
        self.regionalscale = np.zeros(self._regions)
        self.globalscale = 0.0

        # One mean  and cov block with same global 
        # but different regional parameters
        self._glob_mean_est = mean_est[:self._global_size]
        self._reg_mean_est = []
        for l in self.loop_iterator:
            self._reg_mean_est.append(mean_est[l:l+self._regional_size]) 

        self._glob_cov_est = np.abs(self._glob_mean_est)
        self._glob_cov_est[self._glob_cov_est == 0] = 1
        self._glob_cov_est = np.diag(0.01 * self._glob_cov_est)     

        self._reg_cov_est = []
        for num_cov in range(self._regions):
            reg_cov_est = np.abs(self._reg_mean_est[num_cov]) 

            reg_cov_est[reg_cov_est == 0] = 1
            reg_cov_est = np.diag(0.01 * reg_cov_est)
            self._reg_cov_est.append(reg_cov_est)
            
        self._j = 12

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
        logPrior_beta_rw_sd = stats.gamma.logpdf(beta_rw_sd, a = 1.0, scale = 1.0)#WIP was 1/100
        #logPrior_beta_rw_sd = stats.invgamma.logpdf(beta_rw_sd,a=5.0,scale=5.0)                             

        logGlobPrior = logPrior_dI + logPrior_ksens + logPrior_kspec + \
            logPrior_eta + logPrior_p_dead + logPrior_beta_rw_sd    
        return logGlobPrior
        
    def _run_independent(self):
        # Report the current settings
        if self._verbose:
            print('Running subblock (all regions) AMGS')
            print('Total number of iterations: ' + str(self._iterations))
    
        # Problem dimension
        d = (self._regional_size*self._regions) + self._global_size

        # Initial starting parameters
        current = self._x0

        # Chain of stored samples
        chain = np.zeros((self._iterations, d))
        chain_log_target = np.zeros(self._iterations)

        # Initial acceptance rate 
        acceptance_reg = 0
        acceptance_glob = 0

        # Setup Parallel processing
        def ParallelLogP(thr,glob,local):
            local_glob = np.hstack((glob,local[thr,:]))
            return self._target[thr](local_glob,joint_reg=True)
        num_cores = multiprocessing.cpu_count()
        print('Using total threads: ', num_cores)

        for i in range(self._iterations):
            for s in range(self._sub_blocks):                
                # Copy the current whole parameter vector
                proposed = current.copy()
                if s == 0:
                    # Carry out MH step for all regions in parallel
                    # gather all regional parameters, logPs, and flags
                    proposed_logp # build total logP
                    proposed # build full vector
                elif s == 1:
                    # Carry out MH step for Global 
                    # gather global parameters, logPs, and flags  
                    proposed_logp # build total logP
                    proposed # build full vector                                     

                # Free the current's copy
                current = proposed
                current_log_target = propsed_log_target
                proposed = None     
            

            # Start the adaptations
            if i > 199:
                start_adaptation

                        
            # Store the current
            chain[i,:] = current 
            chain_log_target[i] = current_log_target
            # Print-outs
            if self._verbose and ((i % 2000 == 0) or (i==self._iterations-1)):
                print_outs

        return chain
    
    def run(self):
        return self._run_independent()


