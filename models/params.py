import numpy as np
from scipy import stats

class VectorParam(object):
    def __init__(self, length, prior):
        self._length = length
        self._prior = prior

    def __len__(self):
        return self._length

    def log_prior_jacobian(self, value):
        return self._prior.logpdf(value).sum()

    def sample_to_value(self, value):
        return value

    def draw_from_prior(self):
        raise NotImplementedError

class VectorParamMultiplePriors(VectorParam):
    def __init__(self, length, prior):
        assert len(prior) == length
        super().__init__(length, prior)

    def log_prior_jacobian(self, value):
        return sum(self._prior[i].logpdf(val) for i, val in enumerate(value))

    def draw_from_prior(self):
        return np.array([self._prior[i].rvs(1) for i in range(self._length)]).flatten()

class Param(VectorParam):
    def __init__(self, prior):
        super().__init__(1, prior)

    def draw_from_prior(self):
        return self._prior.rvs(1)

class FixedParam(Param):
    def __init__(self, value):
        self._value = value

    def __len__(self):
        return 0

    def sample_to_value(self, value):
        return self._value

    def log_prior_jacobian(self, value):
        return 0

    def draw_from_prior(self):
        return np.array([])

class ParamTransformAfterPrior(Param):
    def __init__(self, prior, transformation):
        self._transformation = transformation
        super().__init__(prior)

    def sample_to_value(self, value):
        return self._transformation(value)

class ExpAfterPrior(ParamTransformAfterPrior):
    def __init__(self, prior):
        super().__init__(prior, np.exp)

class LogGaussianRWParamSD(Param):
    def __init__(self, sd_param, walk_length=None):
        self._sd_param = sd_param
        self.walk_length = walk_length
        self._sd_len = len(sd_param)

    def __len__(self):
        return self.walk_length + self._sd_len

    def log_prior_jacobian(self, value):
        sd_params = value[0:self._sd_len]
        log_walk_increments = value[self._sd_len:]
        log_sd_prior = self._sd_param.log_prior_jacobian(sd_params)
        sd = self._sd_param.sample_to_value(sd_params)
        log_walk_density = stats.norm.logpdf(log_walk_increments, 0.0, sd).sum()
        return log_sd_prior + log_walk_density

    def sample_to_value(self, value):
        log_walk_increments = value[self._sd_len:]
        return np.exp(log_walk_increments.cumsum())

    def draw_from_prior(self):
        sd_params = self._sd_prior.draw_from_prior()
        sd = self._sd_param.sample_to_value(sd_params)
        log_walk_increments = stats.norm.rvs(size=self.walk_length, scale=sd)
        return np.insert(log_walk_increments, 0, sd_params)

class LogGaussianRW(LogGaussianRWParamSD):
    def __init__(self, sd_prior, walk_length=None):
        sd_param = Param(sd_prior)
        super().__init__(sd_param, walk_length)

class LogGaussianRWFixedSD(LogGaussianRWParamSD):
    def __init__(self, sd_prior, walk_length=None):
        sd_param = FixedParam(sd_prior)
        super().__init__(sd_param, walk_length)

class _ExtendParam(Param):
    def __init__(self, param):
        self._param = param

    def __len__(self):
        return len(self._param)

    def log_prior_jacobian(self, value):
        return self._param.log_prior_jacobian(value)

    def sample_to_value(self, value):
        return self._param.sample_to_value(value)

    def draw_from_prior(self):
        return self._param.draw_from_prior()

class MatrixSusceptibleChildren(_ExtendParam):
    def __init__(self, param, n_strata, n_children_strata):
        assert n_strata > n_children_strata
        super().__init__(param)
        self._n_strata = n_strata
        self._n_children_strata = n_children_strata

    def _value_to_matrix(self, value):
        children = np.broadcast_to(value, (self._n_children_strata, self._n_strata))
        adults = np.broadcast_to(1, (self._n_strata - self._n_children_strata, self._n_strata))
        return np.vstack([children, adults])

    def sample_to_value(self, value):
        _value = super().sample_to_value(value)
        return self._value_to_matrix(_value)