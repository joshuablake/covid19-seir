from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
from math import isfinite
import numpy as np
import scipy.stats as stats
import time

class SingleBlockAMGS(object):
    def __init__(self, target, x0, mean_est=None, iterations=150000):
        if mean_est is None:
            mean_est = x0
        assert len(x0) == target.n_params
        assert len(mean_est) == target.n_params
        self._target = target
        self._iterations = iterations
        self._x0 = np.array(x0)

        self._verbose = True

        self.scale = 0.0
        self._mean_est = np.array(mean_est)
        self._cov_est = np.abs(self._mean_est)
        self._cov_est[self._cov_est == 0] = 1
        self._cov_est = np.diag(0.01 * self._cov_est)
        
    def run(self):
        # Report the current settings
        if self._verbose:
            print('Running AMGS')
            print('Total number of iterations: ' + str(self._iterations))

        # Initial starting parameters
        current = self._x0

        # Chain of stored samples
        chain = np.zeros((self._iterations, self._target.n_params))
        chain_log_target = np.zeros(self._iterations)

        # Initial acceptance rate
        acceptance = 0


        for i in range(self._iterations):             
            # Copy the current whole parameter vector
            proposed = current.copy()
            # Update  parameters              
            try:
                proposed = np.random.multivariate_normal(current,
                        np.exp(self.scale) * self._cov_est)      
            except np.linalg.linalg.LinAlgError as e:
                e.chain = chain   
                raise e

            # Evaluate the logP if first iteration of the chain            
            if i==0:
                current_log_target = self._target(current)
                if not np.isfinite(current_log_target):
                    raise ValueError("Posterior density not finite at starting value")

            # Evaluate the logP of the proposed
            proposed_log_target = self._target(proposed)

            # Routine MH step
            log_ratio = proposed_log_target - current_log_target
            log_ratio = min(np.log(1), log_ratio)

            accepted = 0
            if np.isfinite(proposed_log_target) and log_ratio > np.log(np.random.rand(1)):
                accepted = 1
                current = proposed
                current_log_target = proposed_log_target  

            # Free the current's copy
            proposed = None     
        
            # Update acceptance rates
            acceptance = (i * acceptance + float(accepted)) / (i + 1)

            # Start the adaptations
            if i > 199:
                learning_rate = (i - 199 + 1.0) ** -0.6

                # Adapt proposal scales
                self.scale += learning_rate * (accepted - 0.234) 

                # Etch out the changes for global and regional components
                difference = current - self._mean_est

                # Adapt proposal means
                self._mean_est = (1-learning_rate) * self._mean_est + learning_rate*current

                # Adapt proposal covariances
                self._cov_est = (1-learning_rate) * self._cov_est + \
                    learning_rate*(np.outer(difference, difference))
                        
            # Store the current
            chain[i,:] = current 
            chain_log_target[i] = current_log_target
            # Print-outs
            if self._verbose and ((i % 2000 == 0) or (i==self._iterations-1)):
                print('Iteration ' + str(i) + ' of ' + str(self._iterations))
                print('Acceptance rate: ' + str(acceptance))
                #print('Regional  beta: ' + str(np.exp(self.globalscale)))
                if i>100:
                    print('Log target: ', np.mean(chain_log_target[i-100:i]))

        return chain
    