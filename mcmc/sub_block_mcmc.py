from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import scipy.stats as stats
import pickle


class subBlockMCMC(object):
    def __init__(self, target, x0, scale, iterations=150000):

        self._target = target
        self._iterations = iterations
        self._x0 = x0
        self._sub_blocks = 9
        self._max_blocksize = 9
        self._scale = scale

        self._verbose = True

    def _run_independent(self):
        # Report the current settings
        if self._verbose:
            print('Running adaptive Independent MCMC')
            print('Total number of iterations: ' + str(self._iterations))

    
        # Problem dimension
        d = self._target.n_params

        # Initial starting parameters
        current = self._x0

        # Chain of stored samples

        chain = np.zeros((self._iterations,d))

        # Initial acceptance rate 
        acceptance_eta = 0
        acceptance_dI = 0
        acceptance_theta = 0
        acceptance_psir = 0
        acceptance_nu = 0
        acceptance_m = 0
        acceptance_gp = 0
        acceptance_cc = 0
        acceptance_bg = 0
        acc_rate = 0

        for i in range(self._iterations):
            chosen = np.ones(9)
            for j in range(self._sub_blocks):

                # Copy the current whole parameter vector
                proposed = current.copy()

                # Sequentially update (locally) the copied version, which is the proposed
                if j==0:
                    proposed[:8] = stats.norm(current[:8], self._scale[j]).rvs()
                elif j==1:
                    proposed[8] = stats.norm(current[8], self._scale[j]).rvs()
                elif j==2:
                    proposed[9:14] = stats.norm(current[9:14], self._scale[j]).rvs()
                elif j==3:
                    proposed[14] = stats.norm(current[14], self._scale[j]).rvs()
                elif j==4:
                    proposed[15] = stats.norm(current[15], self._scale[j]).rvs()
                elif j==5:
                    proposed[16] = stats.norm(current[16], self._scale[j]).rvs()
                elif j==6:
                    proposed[17] = stats.norm(current[17], self._scale[j]).rvs()   
                elif j==7:
                    proposed[18:20] = stats.norm(current[18:20], self._scale[j]).rvs()   
                elif j==8:
                    chosen = np.random.choice(np.arange(20,49,1),9)
                    proposed[chosen] = stats.norm(current[chosen], self._scale[j][chosen-20]).rvs()   

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
                    acceptance_gp = acc_rate = (i * acceptance_gp + float(accepted)) / (i + 1)      
                elif j==1:
                    acceptance_cc = acc_rate = (i * acceptance_cc + float(accepted)) / (i + 1)    
                elif j==2:
                    acceptance_m = acc_rate = (i * acceptance_m + float(accepted)) / (i + 1)    
                elif j==3:
                    acceptance_dI = acc_rate = (i * acceptance_dI + float(accepted)) / (i + 1)    
                elif j==4:
                    acceptance_theta = acc_rate = (i * acceptance_theta + float(accepted)) / (i + 1)    
                elif j==5:
                    acceptance_psir = acc_rate = (i * acceptance_psir + float(accepted)) / (i + 1)    
                elif j==6:
                    acceptance_nu = acc_rate = (i * acceptance_nu + float(accepted)) / (i + 1)    
                elif j==7:
                    acceptance_eta = acc_rate = (i * acceptance_eta + float(accepted)) / (i + 1)    
                elif j==8:
                    acceptance_bg = acc_rate = (i * acceptance_bg + float(accepted)) / (i + 1) 

                # Adapt proposal scale
                if i < 150000 and i % 500 == 0:

                    if acc_rate <= 0.2:
                        self._scale[j] *= 0.9
                    elif acc_rate >= 0.3:
                        self._scale[j] *= 1.1  
                          
            # Store the current
            chain[i,:] = current 

            # checkpoint
            if i % 50000 == 0:
                chain_filename = 'subblock_chain.p'
                pickle.dump(chain, open(chain_filename, 'wb'))
            # Print-outs
            if self._verbose and ((i % 10000 == 0) or (i==self._iterations-1)):
                print('Iteration ' + str(i) + ' of ' + str(self._iterations))
                print('  Acceptance rate gp: ' + str(acceptance_gp))
                print('  Acceptance rate cc: ' + str(acceptance_cc))
                print('  Acceptance rate m: ' + str(acceptance_m))
                print('  Acceptance rate dI: ' + str(acceptance_dI))
                print('  Acceptance rate theta: ' + str(acceptance_theta))
                print('  Acceptance rate psir: ' + str(acceptance_psir))
                print('  Acceptance rate nu: ' + str(acceptance_nu))
                print('  Acceptance rate eta: ' + str(acceptance_eta))
                print('  Acceptance rate bg: ' + str(acceptance_bg))

        return chain, self._scale
    
    def run(self):

        return self._run_independent()

