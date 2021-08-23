from . import params
from _pytest.mark import param
import numpy as np
from numpy.core.numeric import Inf
import pandas as pd
import scipy
import scipy.stats as stats
import scipy.special as special

MODEL_PARAMS = {
    "psir": params.Param(stats.gamma(a = 31.36, scale = 1/224)),
    "i0": params.ExpAfterPrior(stats.norm(-5, 2)),
    "pi": params.Param(stats.beta(23.0, 3.44)),
    "dL": params.FixedParam(2),
    "dI": params.FixedParam(4),
}

RW_SD_PRIOR = stats.expon(scale=1)

class LogPosteriorOneRegion(object):
    def __init__(self, forward_model, positive_num, test_num, testing = False, fixed_params = {}):
        assert len(positive_num) == len(test_num)
        self._forward_model = forward_model
        self._positives = positive_num
        self._tests = test_num
        self._model_params = MODEL_PARAMS.copy()
        self._model_params.update({name: params.FixedParam(value) for name, value in fixed_params.items()})
        self._model_params["beta"] = params.LogGaussianRW(RW_SD_PRIOR, self._forward_model.num_betas)
        self.n_params = sum(len(p) for p in self._model_params.values())

    def _log_lik(self, results):
        return sum(stats.binom.logpmf(self._positives, self._tests, results[1]))

    def __call__(self, parameters):
        assert len(parameters) == self.n_params

        log_prior = 0
        idx = 0
        param_values = {}
        for name, config in self._model_params.items():
            n_values = len(config)
            end_idx = idx + n_values
            values = parameters[idx:end_idx]
            log_prior += config.log_prior_jacobian(values)
            param_values[name] = config.sample_to_value(values)
            idx = end_idx

        try:
            result = self._forward_model.simulate(**param_values)
        except ValueError:
            return -Inf

        # Now calculate the individual data stream log Likelihoods
        log_lik = self._log_lik(result)

        return log_lik + log_prior

    def get_param_names(self):
        names = []
        for name, config in self._model_params.items():
            n_values = len(config)
            if n_values == 0:
                continue
            if n_values == 1:
                names.append(name)
            else:
                names.extend("{}[{}]".format(name, i) for i in range(n_values))
        return names

class LogPosteriorStratifiedRegion(object):
    def __init__(self, forward_model, positive_num, test_num, testing = False, fixed_params = {}):
        assert positive_num.shape == test_num.shape == (forward_model.max_time, forward_model.num_strata)
        assert (positive_num >= 0).all()
        assert (test_num >= 0).all()
        assert (positive_num <= test_num).all()
        self._results_index = (test_num > 0)
        self._forward_model = forward_model
        self._positives = positive_num[self._results_index]
        self._tests = test_num[self._results_index]
        self._fixed_params = fixed_params
        model_params_to_estimate = [p for p in PARAMS if p not in fixed_params.keys()]
        beta_params = ["beta_rw_sd"] + ["beta[{}]".format(i) for i in range(forward_model._num_betas)]
        self._estimated_params = tuple(model_params_to_estimate + beta_params)
        self.n_params = len(self._estimated_params)
        self._transform = False

    def _log_lik(self, results):
        return sum(stats.binom.logpmf(self._positives, self._tests, results[1]))

    def __call__(self, parameters, joint_reg=False):
        assert len(parameters) == self.n_params
        named_params = {k: v for k, v in zip(self._estimated_params, parameters)}
        named_params.update(self._fixed_params)
        betas = parameters[-self._forward_model._num_betas:]

        ## Now the priors ####
        log_prior = 0
        for param, prior in PRIORS.items():
            if param not in self._fixed_params:
                log_prior += prior.logpdf(named_params[param])

        
        # Call the simulator to get the expected nnis and prev
        named_params["i0"] = np.exp(named_params["i0"])
        params_for_sim = {k: named_params[k] for k in PARAMS}
        params_for_sim["beta"] = np.exp(np.cumsum(betas))
        try:
            result = self._forward_model.simulate(**params_for_sim)
        except ValueError:
            return -Inf

        # Now calculate the individual data stream log Likelihoods
        log_lik = self._log_lik(result)
                
        # Now calculate the random-walk likelihood
        log_rw_lik = np.sum(stats.norm.logpdf(betas, 0.0, named_params["beta_rw_sd"]))

        return log_lik + log_rw_lik + log_prior
            
        


