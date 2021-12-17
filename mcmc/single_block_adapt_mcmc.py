from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
from math import isfinite
import numpy as np
import scipy.stats as stats
import time

class AMGSBlock(object):
    def __init__(self, target, mean_est, record_diagnostics, iterations, n_params=None, stop_adapting=None):
        if stop_adapting is None:
            self._stop_adapting = iterations
        else:
            self._stop_adapting = stop_adapting
        
        if n_params is None:
            n_params = target.n_params
        assert len(mean_est) == n_params
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
            self.mean_chain = np.empty((iterations - 200, n_params))
            self.scale_chain = np.empty(iterations - 200)
            self.cov_chain = np.empty_like(self.scale_chain)
            self.cov_matrix_chain = np.empty((iterations-200, n_params, n_params))

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
        if iteration_num > 199 and iteration_num < self._stop_adapting:
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

class UpdateRWSDGaussianProposal(object):
    def __init__(self, rw_param, proposal_std=0.8):
        self._rw_param = rw_param
        self._proposal_std = proposal_std
        self.acceptance = 0

    def do_update(self, iteration_num, current, current_log_target):
        current_rw = current[-len(self._rw_param):]
        sd_len = self._rw_param.sd_len
        current_sd = current_rw[:sd_len]
        proposal_sd = np.random.normal(current_sd, self._proposal_std)
        proposal_rw = current_rw.copy()
        proposal_rw[:sd_len] = proposal_sd
        log_target_diff = self._rw_param.log_prior_jacobian(proposal_rw) - self._rw_param.log_prior_jacobian(current_rw)
        accepted = log_target_diff > np.log(np.random.rand(1))
        if accepted:
            self.acceptance = (iteration_num * self.acceptance + 1) / (iteration_num + 1)
            current[-len(self._rw_param):] = proposal_rw
            return current, current_log_target + log_target_diff
        else:
            self.acceptance = (iteration_num * self.acceptance) / (iteration_num + 1)
            return current, current_log_target


class SingleBlockAMGS(object):
    def __init__(self, target, x0, mean_est=None, iterations=150000, n_params=None, stop_adapting=None):
        if mean_est is None:
            mean_est = x0
        if n_params is None:
            n_params = target.n_params
        assert len(mean_est) == n_params
        assert len(x0) == n_params
        self._target = target
        self._iterations = iterations
        self._x0 = np.array(x0)

        self._verbose = True

        self.scale = 0.0
        self._mean_est = np.array(mean_est)

        self._stop_adapting = stop_adapting

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

    def _loop_init(self, amgs_target=None):
        if amgs_target is None:
            amgs_target = self._target
        amgs_block = AMGSBlock(amgs_target, self._mean_est, self._record_diagnostics, self._iterations, stop_adapting=self._stop_adapting)
        
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
    
# WARNING: this is invalid if using a non-centred parameterisation
class AMGSWithRWSDUpdate(SingleBlockAMGS):
    def __init__(self, target, x0, mean_est=None, iterations=150000, stop_adaption=None):
        assert list(target._model_params.keys())[-1] == "beta"
        self._len_rw_params = len(target._model_params["beta"])
        super().__init__(target, x0, mean_est=mean_est, iterations=iterations, stop_adaption=stop_adaption)

    def run(self, sd_update_object=UpdateRWSD, return_log_target=False, record_diagnostics=False):
        if record_diagnostics:
            return_log_target = True
        self._return_log_target = return_log_target
        self._record_diagnostics = record_diagnostics

        amgs_block = self._loop_init()
        sd_update = sd_update_object(self._target._model_params["beta"])

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

# Wrap a target but hold the value of one parameter fixed
class _TargetOmitValue(object):
    def __init__(self, underlying_target, position_to_fix, fixed_value_start, fixed_param):
        self._target = underlying_target
        self.n_params = self._target.n_params - 1
        self.position_to_fix = position_to_fix + 1
        self._fixed_param = fixed_param
        self._fixed_value = self._fixed_param.sample_to_value(fixed_value_start)

    def __call__(self, value):
        value = np.insert(value, self.position_to_fix, self._fixed_value)
        return self._target(value)

    def set_fixed_value(self, value):
        self._fixed_value = self._fixed_param.sample_to_value(value)
    
# WARNING: this is invalid if using a non-centred parameterisation
# Update the standard deviation outside of the AMGS block only
class AMGSWithIndepedendentRWSDUpdate(SingleBlockAMGS):
    def __init__(self, target, x0, iterations=150000, stop_adaption=None):
        assert list(target._model_params.keys())[-1] == "beta"
        self.sd_position = -len(target._model_params["beta"])
        self.sd_start = x0[self.sd_position]
        x0_exclude_sd = np.delete(x0, self.sd_position)
        sd_param = target._model_params["beta"].sd_param
        self.amgs_target = _TargetOmitValue(target, self.sd_position, self.sd_start, sd_param)
        super().__init__(self.amgs_target, x0_exclude_sd, iterations=iterations, stop_adapting=stop_adaption)

    def run(self, sd_update_object=UpdateRWSDGaussianProposal, return_log_target=False, record_diagnostics=False):
        if record_diagnostics:
            return_log_target = True
        self._return_log_target = return_log_target
        self._record_diagnostics = record_diagnostics

        amgs_block = self._loop_init()
        sd_update = sd_update_object(self._target._target._model_params["beta"])
        sd_current = self.sd_start
        true_chain_shape = list(self.chain.shape)
        true_chain_shape[1] += 1
        self.chain = np.empty(true_chain_shape)

        for i in range(self._iterations):
            self.amgs_target.set_fixed_value(sd_current)   
            self.safe_amgs_update(amgs_block, i)
            all_params = np.insert(self.current, self.sd_position+1, sd_current)
            all_params, self.current_log_target = sd_update.do_update(i, all_params, self.current_log_target)   
            sd_current = all_params[self.sd_position]

            # Store the current
            self.chain[i,:] = all_params
            self.chain_log_target[i] = self.current_log_target
            self._print_progress(amgs_block, sd_update, i)

        return self._get_return_value(amgs_block)

    def _print_progress(self, amgs_block, sd_update, i):
        super()._print_progress(amgs_block, i)
        # Print-outs
        if self._verbose and ((i % 2000 == 0) or (i==self._iterations-1)):
            print(f"SD acceptance rate {sd_update.acceptance}")