from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import scipy.stats as stats
import time
import pickle


class subBlockAMGS2(object):
    def __init__(self, target, x0, mean_est, iterations=150000):

        self._target = target
        self._iterations = iterations
        self._x0 = x0
        self._sub_blocks = 3
        self._max_blocksize = 9

        self._verbose = True
        self.globalscale = np.zeros(self._sub_blocks)

        self._mean_est = [ mean_est[:9], mean_est[9:20], mean_est[20:] ]


        self._cov_est = []
        for num_cov in range(self._sub_blocks):

            if num_cov == 2:
                cov_est = np.abs(self._mean_est[num_cov])
                cov_est = cov_est[:9]
            else:
               cov_est = np.abs(self._mean_est[num_cov]) 

            cov_est[cov_est == 0] = 1
            cov_est = np.diag(0.01 * cov_est)
            # Use to create diagonal matrix
            self._cov_est.append(cov_est)

    def _run_independent(self):
        # Report the current settings
        if self._verbose:
            print('Running subblock AMGS')
            print('Total number of iterations: ' + str(self._iterations))

    
        # Problem dimension
        d = self._target.n_params

        # Initial starting parameters
        current = self._x0

        # Chain of stored samples

        chain = np.zeros((self._iterations,d))

        # Initial acceptance rate 
        acceptance_ode = 0
        acceptance_gp = 0
        acceptance_bg = 0

        for i in range(self._iterations):
            chosen = np.ones(9)
            for j in range(self._sub_blocks):

                # Copy the current whole parameter vector
                proposed = current.copy()

                # Sequentially update (locally) the copied version, which is the proposed
                if j==0:
                    proposed[:9] = np.random.multivariate_normal(current[:9], np.exp(self.globalscale[j]) * self._cov_est[j])
                elif j==1:
                    proposed[9:20] = np.random.multivariate_normal(current[9:20], np.exp(self.globalscale[j]) * self._cov_est[j])
                elif j==2:
                    chosen = np.random.choice(np.arange(20,49,1),9)
                    proposed[chosen] = np.random.multivariate_normal(current[chosen], np.exp(self.globalscale[j]) * self._cov_est[j])

                # Evaluate the logP if first iteration of the chain
                if i==0 and j==0:
                    current_log_target = self._target(current)

                # Evaluate the logP of the proposed
                t0 = time.time()
                proposed_log_target = self._target(proposed)
                t1 = time.time()
                total = t1-t0
                print('Time for pdf is: ')
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
                if j==0:
                    acceptance_gp = (i * acceptance_gp + float(accepted)) / (i + 1)      
                elif j==1:
                    acceptance_ode = (i * acceptance_ode + float(accepted)) / (i + 1)    
                elif j==2:
                    acceptance_bg = (i * acceptance_bg + float(accepted)) / (i + 1)    

            # Adapt proposal scale
            if i > 199:
                learning_rate = (i - 199 + 1.0) ** -0.6
                for num_block in range(self._sub_blocks):

                    self.globalscale[num_block] += learning_rate * (accepted - 0.234) 

                    if num_block == 0:
                        current_block = current[:9]
                        difference = current_block - self._mean_est[num_block]
                    elif num_block == 1:
                        current_block = current[9:20]
                        difference = current_block - self._mean_est[num_block]
                    elif num_block == 2:
                        current_block = current[chosen]
                        difference = current_block - self._mean_est[num_block][chosen-20]

                    if num_block == 2:
                        self._mean_est[num_block][chosen-20] = (1-learning_rate) * self._mean_est[num_block][chosen-20] + learning_rate*current_block
                    else:
                        self._mean_est[num_block] = (1-learning_rate) * self._mean_est[num_block] + learning_rate*current_block
                    
                    self._cov_est[num_block] = (1-learning_rate) * self._cov_est[num_block] + learning_rate*(np.outer(difference, difference))

                          
            # Store the current
            chain[i,:] = current 

            # Print-outs
            if self._verbose and ((i % 20000 == 0) or (i==self._iterations-1)):
                print('Iteration ' + str(i) + ' of ' + str(self._iterations))
                print('  Acceptance rate gp: ' + str(acceptance_gp))
                print('  Acceptance rate ode: ' + str(acceptance_ode))
                print('  Acceptance rate bg: ' + str(acceptance_bg))

        return chain
    
    def run(self):

        return self._run_independent()

