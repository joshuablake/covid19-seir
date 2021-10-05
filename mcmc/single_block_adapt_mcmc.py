from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
from math import isfinite
import numpy as np
import scipy.stats as stats
import time

class AMGSBlock(object):
    def __init__(self, target, mean_est, record_diagnostics, iterations):
        assert len(mean_est) == target.n_params
        self._target = target

        self._verbose = True

        self.scale = 0.0
        self._mean_est = np.array(mean_est)
        self._cov_est = np.abs(self._mean_est)
        self._cov_est[self._cov_est == 0] = 1
        self._cov_est = np.diag(0.01 * self._cov_est)
        self._record_diagnostics = record_diagnostics
        # Initial acceptance rate
        self.acceptance = 0

        if record_diagnostics:
            # Chains of diagnostics
            self.mean_chain = np.empty((iterations - 200, self._target.n_params))
            self.scale_chain = np.empty(iterations - 200)
            self.cov_chain = np.empty_like(self.scale_chain)
            self.cov_matrix_chain = np.empty((iterations-200, self._target.n_params, self._target.n_params))

    def do_update(self, iteration_num, current, current_log_target):
        # Update  parameters              
        proposed = np.random.multivariate_normal(current,
                    np.exp(self.scale) * self._cov_est)
        # Evaluate the logP of the proposed
        proposed_log_target = self._target(proposed)

        # Routine MH step
        log_ratio = proposed_log_target - current_log_target
        log_ratio = min(np.log(1), log_ratio)

        accepted = np.isfinite(proposed_log_target) and log_ratio > np.log(np.random.rand(1))

        # Update acceptance rates
        self.acceptance = (iteration_num * self.acceptance + float(accepted)) / (iteration_num + 1)

        # Start the adaptations
        if iteration_num > 199:
            learning_rate = (iteration_num - 199 + 1.0) ** -0.6

            # Adapt proposal scales
            self.scale += learning_rate * (accepted - 0.234) 

            # Etch out the changes for global and regional components
            difference = current - self._mean_est

            # Adapt proposal means
            self._mean_est = (1-learning_rate) * self._mean_est + learning_rate*current

            # Adapt proposal covariances
            self._cov_est = (1-learning_rate) * self._cov_est + \
                learning_rate*(np.outer(difference, difference))

            if self._record_diagnostics:
                self.mean_chain[iteration_num-200] = self._mean_est
                self.cov_chain[iteration_num-200] = np.linalg.cond(self._cov_est)
                self.scale_chain[iteration_num-200] = self.scale
                self.cov_matrix_chain[iteration_num-200] = self._cov_est

        if accepted:
            return proposed, proposed_log_target
        return current, current_log_target

class UpdateRWSD(object):
    def __init__(self, rw_param):
        self._rw_param = rw_param
        self.acceptance = 0

    def do_update(self, iteration_num, current, current_log_target):
        current_rw = current[-len(self._rw_param):]
        sd_param = self._rw_param.sd_param
        sd_len = self._rw_param.sd_len
        current_sd = current_rw[:sd_len]
        proposal_sd = sd_param.draw_from_prior()
        proposal_rw = current_rw.copy()
        proposal_rw[:sd_len] = proposal_sd
        log_target_diff = self._rw_param.log_prior_jacobian(proposal_rw) - self._rw_param.log_prior_jacobian(current_rw)
        log_kernel = sd_param.log_prior_jacobian(current_sd) - sd_param.log_prior_jacobian(proposal_sd)
        log_accept_ratio = log_target_diff + log_kernel
        accepted = log_accept_ratio > np.log(np.random.rand(1))
        if accepted:
            self.acceptance = (iteration_num * self.acceptance + 1) / (iteration_num + 1)
            current[-len(self._rw_param):] = proposal_rw
            return current, current_log_target + log_target_diff
        else:
            self.acceptance = (iteration_num * self.acceptance) / (iteration_num + 1)
            return current, current_log_target


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

    def _get_return_value(self, amgs_block):
        if self._record_diagnostics:
            return self.chain, self.chain_log_target, amgs_block.mean_chain, amgs_block.cov_chain, amgs_block.scale_chain, amgs_block.cov_matrix_chain

        if self._return_log_target:
            return self.chain, self.chain_log_target

        return self.chain
        
    def run(self, return_log_target=False, record_diagnostics=False):
        if record_diagnostics:
            return_log_target = True
        self._return_log_target = return_log_target
        self._record_diagnostics = record_diagnostics

        amgs_block = self._loop_init()

        for i in range(self._iterations):           
            self.safe_amgs_update(amgs_block, i)
                        
            # Store the current
            self.chain[i,:] = self.current 
            self.chain_log_target[i] = self.current_log_target
            self._print_progress(amgs_block, i)

        return self._get_return_value(amgs_block)

    def safe_amgs_update(self, amgs_block, i):
        try:
            self.current, self.current_log_target = amgs_block.do_update(i, self.current, self.current_log_target)      
        except np.linalg.linalg.LinAlgError as e:
            e.chain = self.chain
            e.final_iteration = i - 1
            if self._record_diagnostics:
                e.diag = self.chain_log_target, amgs_block.mean_chain, amgs_block.cov_chain, amgs_block.scale_chain, amgs_block.cov_matrix_chain
            raise e

    def _print_progress(self, amgs_block, i):
        # Print-outs
        if self._verbose and ((i % 2000 == 0) or (i==self._iterations-1)):
            print('Iteration ' + str(i) + ' of ' + str(self._iterations))
            print('AMGS acceptance rate: ' + str(amgs_block.acceptance))
                #print('Regional  beta: ' + str(np.exp(self.globalscale)))
            if i>100:
                print('Log target: ', np.mean(self.chain_log_target[i-100:i]))

    def _loop_init(self):
        amgs_block = AMGSBlock(self._target, self._mean_est, self._record_diagnostics, self._iterations)
        
        # Report the current settings
        if self._verbose:
            print('Running AMGS')
            print('Total number of iterations: ' + str(self._iterations))

        # Initial starting parameters
        self.current = self._x0

        # Chain of stored samples
        self.chain = np.empty((self._iterations, self._target.n_params))
        self.chain_log_target = np.empty(self._iterations)

        # Initial log-posterior
        self.current_log_target = self._target(self.current)
        if not np.isfinite(self.current_log_target):
            raise ValueError("Posterior density not finite at starting value")
        return amgs_block
    
class AMGSWithRWSDUpdate(SingleBlockAMGS):
    def __init__(self, target, x0, mean_est=None, iterations=150000):
        assert list(target._model_params.keys())[-1] == "beta"
        self._len_rw_params = len(target._model_params["beta"])
        super().__init__(target, x0, mean_est=mean_est, iterations=iterations)

    def run(self, return_log_target=False, record_diagnostics=False):
        if record_diagnostics:
            return_log_target = True
        self._return_log_target = return_log_target
        self._record_diagnostics = record_diagnostics

        amgs_block = self._loop_init()
        sd_update = UpdateRWSD(self._target._model_params["beta"])

        for i in range(self._iterations):           
            self.safe_amgs_update(amgs_block, i)
            self.current, self.current_log_target = sd_update.do_update(i, self.current, self.current_log_target)   

            # Store the current
            self.chain[i,:] = self.current 
            self.chain_log_target[i] = self.current_log_target
            self._print_progress(amgs_block, sd_update, i)

        return self._get_return_value(amgs_block)

    def _print_progress(self, amgs_block, sd_update, i):
        super()._print_progress(amgs_block, i)
        # Print-outs
        if self._verbose and ((i % 2000 == 0) or (i==self._iterations-1)):
            print(f"SD acceptance rate {sd_update.acceptance}")