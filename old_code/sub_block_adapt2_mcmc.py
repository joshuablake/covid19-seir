from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import scipy.stats as stats
import time
import pickle


class subBlockAMGS3(object):
    def __init__(self, target, x0, mean_est, iterations=150000):

        self._target = target
        self._iterations = iterations
        self._x0 = x0
        self._sub_blocks = 2
        self._max_blocksize = 20

        self._verbose = True
        self.globalscale = np.zeros(self._sub_blocks)

        self._mean_est = [ mean_est[:47], mean_est[47:] ]


        self._cov_est = []
        for num_cov in range(self._sub_blocks):

            if num_cov == 1:
                cov_est = np.abs(self._mean_est[num_cov])
                cov_est = cov_est[:20]
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
        acceptance_block1 = 0
        acceptance_block2 = 0
        #cceptance_bg = 0

        for i in range(self._iterations):
            chosen = np.ones(9)
            for j in range(self._sub_blocks):

                # Copy the current whole parameter vector
                proposed = current.copy()

                # Sequentially update (locally) the copied version, which is the proposed
                if j==0:
                    proposed[:47] = np.random.multivariate_normal(current[:47], np.exp(self.globalscale[j]) * self._cov_est[j])
    
                elif j==1:
                    chosen = np.random.choice(np.arange(47,166,1),20)
                    proposed[chosen] = np.random.multivariate_normal(current[chosen], np.exp(self.globalscale[j]) * self._cov_est[j])

                # Evaluate the logP if first iteration of the chain
                if i==0 and j==0:
                    current_log_target = self._target(current)

                # Evaluate the logP of the proposed
                
                proposed_log_target = self._target(proposed)
                
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
                    acceptance_block1 = (i * acceptance_block1 + float(accepted)) / (i + 1)      
                elif j==1:
                    acceptance_block2 = (i * acceptance_block2 + float(accepted)) / (i + 1)    
 

            # Adapt proposal scale
            if i > 199:
                learning_rate = (i - 199 + 1.0) ** -0.6
                for num_block in range(self._sub_blocks):

                    self.globalscale[num_block] += learning_rate * (accepted - 0.234) 

                    if num_block == 0:
                        current_block = current[:47]
                        difference = current_block - self._mean_est[num_block]
                    elif num_block == 1:
                        current_block = current[chosen]
                        difference = current_block - self._mean_est[num_block][chosen-47]

                    if num_block == 1:
                        self._mean_est[num_block][chosen-47] = (1-learning_rate) * self._mean_est[num_block][chosen-47] + learning_rate*current_block
                    else:
                        self._mean_est[num_block] = (1-learning_rate) * self._mean_est[num_block] + learning_rate*current_block
                    
                    self._cov_est[num_block] = (1-learning_rate) * self._cov_est[num_block] + learning_rate*(np.outer(difference, difference))

                          
            # Store the current
            chain[i,:] = current 

            # Print-outs
            if self._verbose and ((i % 500 == 0) or (i==self._iterations-1)):
                print('Iteration ' + str(i) + ' of ' + str(self._iterations))
                print('  Acceptance rate block1: ' + str(acceptance_block1))
                print('  Acceptance rate block2: ' + str(acceptance_block2))


        return chain
    
    def run(self):

        return self._run_independent()

