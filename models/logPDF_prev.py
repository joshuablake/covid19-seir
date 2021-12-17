from . import params
import numpy as np
from numpy.core.numeric import Inf
import pandas as pd
import scipy.stats as stats

MODEL_PARAMS = {
    "psir": params.Param(stats.gamma(a = 31.36, scale = 1/224)),
    "i0": params.ExpAfterPrior(stats.norm(-5, 2)),
    "pi": params.Param(stats.beta(23.0, 3.44)),
    "dL": params.FixedParam(2),
    "dI": params.FixedParam(4),
}

RW_SD_PRIOR = stats.expon(scale=1)

class LogPosteriorOneRegion(object):
    def __init__(self, forward_model, positive_num, test_num, fixed_params = {}, custom_priors = {}, use_default_priors = True):
        assert len(positive_num) == len(test_num)
        self._forward_model = forward_model
        self._positives = positive_num
        self._tests = test_num
        if use_default_priors:
            self._model_params = MODEL_PARAMS.copy()
            self._model_params.update({name: params.FixedParam(value) for name, value in fixed_params.items()})
            self._model_params["beta"] = params.LogGaussianRW(RW_SD_PRIOR, self._forward_model.num_betas)
            self._model_params.update(custom_priors)
        else:
            self._model_params = custom_priors
        self.n_params = sum(len(p) for p in self._model_params.values())

    def _log_lik(self, results):
        return sum(stats.binom.logpmf(self._positives, self._tests, results[1]))

    def __call__(self, parameters):
        try:
            result, log_prior = self.simulate(parameters, True)
        except ValueError:
            return -Inf
        log_lik = self._log_lik(result)
        return log_lik + log_prior

    def simulate(self, parameters, include_log_prior=False):
        assert len(parameters) == self.n_params
        log_prior, param_values = self.param_array_to_dict(parameters)    
        result = self._forward_model.simulate(**param_values)
        if include_log_prior:
            return result, log_prior
        return result

    def param_array_to_dict(self, parameters):
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
        return log_prior,param_values


    def param_value_dict_to_array(self, values):
        array = []
        for name, config in self._model_params.items():
            value_len = len(config)
            if value_len == 0:
                continue
            value = values[name]
            if value_len == 1:
                assert np.isscalar(value_len)
                try:
                    array.append(value[0])
                except (TypeError, IndexError):
                    array.append(value)
            else:
                assert len(value) == value_len
                array.extend(list(value))
        return np.array(array)

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

class LogPosteriorStratifiedRegion(LogPosteriorOneRegion):
    def __init__(self, forward_model, positive_num, test_num, priors):

        if positive_num is None:
            assert test_num is None
            positives = []
            tests = []
        else:
            try:
                num_strata = forward_model.num_strata
            except AttributeError:
                # One strata assumed
                assert len(positive_num) == len(test_num) == forward_model.max_time
            else:
                assert positive_num.shape == test_num.shape == (forward_model.max_time, num_strata)
            assert (positive_num >= 0).all()
            assert (test_num >= 0).all()
            assert (positive_num <= test_num).all()
            self._results_index = (test_num > 0)
            positives = positive_num[self._results_index]
            tests = test_num[self._results_index]
        super().__init__(forward_model, positives, tests, custom_priors=priors, use_default_priors=False)

    def _log_lik(self, results):
        return sum(stats.binom.logpmf(self._positives, self._tests, results[1][self._results_index]))