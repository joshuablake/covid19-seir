from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import scipy.stats as stats
import time
from joblib import Parallel, delayed
import multiprocessing
import time as timer


class subBlockRegionalAMGS(object):
    def __init__(self, target, x0, mean_est, iterations=150000):

        self._target = target
        self._iterations = iterations
        self._x0 = x0
        self._sub_blocks = int(7)
        self._global_size = int(12)
        self._regional_size = int(22)
        # The iterator churns start index of each regional block of parameters
        self.loop_iterator = \
            range(self._global_size, ((self._regional_size*self._sub_blocks) + \
            self._global_size),self._regional_size)
        self.region_starts = np.array([i for i in self.loop_iterator])
        self._verbose = True

        # One scale for each region
        self.globalscale = np.zeros(self._sub_blocks)

        # One mean  and cov block with same global 
        # but different regional parameters
        self._mean_est = []
        for l in self.loop_iterator:
            self._mean_est.append(np.hstack((mean_est[:self._global_size],\
            mean_est[l:l+self._regional_size]))) 

        self._cov_est = []
        for num_cov in range(self._sub_blocks):
            cov_est = np.abs(self._mean_est[num_cov]) 

            cov_est[cov_est == 0] = 1
            cov_est = np.diag(0.01 * cov_est)
            self._cov_est.append(cov_est)

    def _run_independent(self):
        # Report the current settings
        if self._verbose:
            print('Running subblock (all regions) AMGS')
            print('Total number of iterations: ' + str(self._iterations))

    
        # Problem dimension
        d = (self._regional_size*self._sub_blocks) + self._global_size

        # Initial starting parameters
        current = self._x0

        # Chain of stored samples

        chain = np.zeros((self._iterations, d))

        # Initial acceptance rate 
        acceptance = 0
        def ParallelLogP(thr,glob,local):
            local_glob = np.hstack((glob,local[thr,:]))
            return self._target[thr](local_glob,joint_reg=True)
        num_cores = multiprocessing.cpu_count()
        print(num_cores)
        for i in range(self._iterations):
            j = int(np.random.choice(self.region_starts,1))
            pos_j = list(self.region_starts).index(j)
            # Copy the current whole parameter vector
            proposed = current.copy()
            # Update the global plus one region's parameters
            oneoldreg_oldglob_current = np.hstack((current[:self._global_size],current[j:(j+self._regional_size)]))
            onenewreg_newglob_proposed = \
                    np.random.multivariate_normal(oneoldreg_oldglob_current, \
                        np.exp(self.globalscale[pos_j]) * \
                            self._cov_est[pos_j])
            proposed[:self._global_size] = onenewreg_newglob_proposed[:self._global_size] # Update global
            proposed[j:j+self._regional_size] = onenewreg_newglob_proposed[self._global_size:]  # and regional parameters

            # Evaluate the logP if first iteration of the chain

            
            if i==0:
                alloldreg_current = current[self._global_size:].reshape((self._sub_blocks,self._regional_size))
                current_log_targets = Parallel(n_jobs=num_cores)(
                    delayed(ParallelLogP)(thread,current[:self._global_size],
                    alloldreg_current) for thread in range(self._sub_blocks))
                current_log_target = np.sum(current_log_targets)


            # Evaluate the logP of the proposed
            allnewreg_proposed = proposed[self._global_size:].reshape((self._sub_blocks,self._regional_size))
            proposed_log_targets = Parallel(n_jobs=num_cores)(
                delayed(ParallelLogP)(thread,proposed[:self._global_size],
                allnewreg_proposed) for thread in range(self._sub_blocks))
            proposed_log_target = np.sum(proposed_log_targets)            
            # Routine MH step
            log_ratio = proposed_log_target - current_log_target
            log_ratio = min(np.log(1), log_ratio)

            accepted = 0
            if np.isfinite(proposed_log_target):
                if log_ratio > np.log(np.random.rand(1)):
                    accepted = 1
                    current = proposed
                    current_log_target = proposed_log_target  

            # Free the current's copy
            proposed = None     
            
            # Update acceptance rates

            acceptance = (i * acceptance + float(accepted)) / (i + 1)        
            # Adapt proposal scale
            if i > 199:
                learning_rate = (i - 199 + 1.0) ** -0.6
                R = 0
                for l in self.loop_iterator:

                    self.globalscale[R] += learning_rate * (accepted - 0.234) 
                    current_block = np.hstack((current[:self._global_size],current[l:l+self._regional_size]))
                    difference = current_block - self._mean_est[R]
                    self._mean_est[R] = (1-learning_rate) * self._mean_est[R] + learning_rate*current_block
                    self._cov_est[R] = (1-learning_rate) * self._cov_est[R] + \
                        learning_rate*(np.outer(difference, difference))
                    R += 1
                         
            # Store the current
            chain[i,:] = current 

            # Print-outs
            if self._verbose and ((i % 5000 == 0) or (i==self._iterations-1)):
                print('Iteration ' + str(i) + ' of ' + str(self._iterations))
                print('  Acceptance rate: ' + str(acceptance))

        return chain
    
    def run(self):

        return self._run_independent()

