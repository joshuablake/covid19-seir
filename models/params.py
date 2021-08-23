import numpy as np
from scipy import stats

class Param(object):
    def __init__(self, prior):
        self._prior = prior

    def __len__(self):
        return 1

    def sample_to_value(self, value):
        return value

    def log_prior_jacobian(self, value):
        return self._prior.logpdf(value)

class FixedParam(Param):
    def __init__(self, value):
        self._value = value

    def __len__(self):
        return 0

    def sample_to_value(self, value):
        return self._value

    def log_prior_jacobian(self, value):
        return 0

class ParamTransformAfterPrior(Param):
    def __init__(self, prior, transformation):
        self._transformation = transformation
        super().__init__(prior)

    def sample_to_value(self, value):
        return self._transformation(value)

class ExpAfterPrior(ParamTransformAfterPrior):
    def __init__(self, prior):
        super().__init__(prior, np.exp)

class LogGaussianRW(Param):
    def __init__(self, sd_prior, walk_length=None):
        self._sd_prior = sd_prior
        self.walk_length = walk_length

    def __len__(self):
        return self.walk_length + 1

    def log_prior_jacobian(self, value):
        sd = value[0]
        log_sd_prior = self._sd_prior.logpdf(sd)
        log_walk_increments = value[1:]
        log_walk_density = stats.norm.logpdf(log_walk_increments, 0.0, sd).sum()
        return log_sd_prior + log_walk_density

    def sample_to_value(self, value):
        log_walk_increments = value[1:]
        return np.exp(log_walk_increments.cumsum())