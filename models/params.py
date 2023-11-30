import numpy as np
from scipy import stats
from scipy.special import expit, logit

class VectorParam(object):
    """Parameter of arbitrary length and prior, sampled on the natural scale."""
    def __init__(self, length, prior):
        """Main constructor
        Args:
            length (int): Length of the parameter vector
            prior (scipy.stats.distribution): Prior distribution for the parameter.
              Must have a `logpdf` method which returns a numpy array.
        """
        self._length = length
        self._prior = prior

    def __len__(self):
        return self._length

    def log_prior_jacobian(self, value):
        return self._prior.logpdf(value).sum()

    def sample_to_value(self, value):
        return value

    def value_to_sample(self, sample):
        return sample

    def draw_from_prior(self, random_state=None):
        raise NotImplementedError

class VectorParamMultiplePriors(VectorParam):
    """Parameter with a different prior for each element of the vector."""
    def __init__(self, length, prior):
        """Main constructor
        Args:
            length (int): Length of the parameter vector
            prior (list of scipy.stats.distribution): Prior distribution for each element of the parameter.
              Each element must have a `logpdf` method which returns a numpy array.
              Must be of length `length`.
        """
        assert length > 1
        assert len(prior) == length
        super().__init__(length, prior)

    def log_prior_jacobian(self, value):
        return sum(self._prior[i].logpdf(val) for i, val in enumerate(value))

    def draw_from_prior(self, random_state=None):
        return np.array([self._prior[i].rvs(1, random_state=random_state) for i in range(self._length)]).flatten()

class Param(VectorParam):
    """"A scalar parameter with an arbitrary prior distribution."""
    def __init__(self, prior):
        super().__init__(1, prior)

    def draw_from_prior(self, random_state=None):
        return self._prior.rvs(1, random_state=random_state)

class FixedParam(Param):
    """A parameter with a fixed value, i.e. is not sampled by MCMC."""
    def __init__(self, value):
        self._value = value

    def __len__(self):
        return 0

    def sample_to_value(self, value):
        return self._value

    def value_to_sample(self, sample):
        return np.array([])

    def log_prior_jacobian(self, value):
        return 0

    def draw_from_prior(self, random_state=None):
        return np.array([])

class _ParamTransformAfterPrior(Param):
    def __init__(self, prior, transformation, reverse_transformation):
        self._transformation = transformation
        self._reverse_transformation = reverse_transformation
        super().__init__(prior)

    def sample_to_value(self, value):
        return self._transformation(value)

    def value_to_sample(self, sample):
        return self._reverse_transformation(sample)

class ExpAfterPrior(_ParamTransformAfterPrior):
    """"A parameter which has its prior on the log scale, but is used by the model on the natural scale."""
    def __init__(self, prior):
        super().__init__(prior, np.exp, np.log)

class LogGaussianRWParamSD(Param):
    """A Gaussian random walk on the log scale.
    
    The following model, with T being the length of the walk.
    .. math::
        \\log(X_{t}) \\sum N(\\log(X_{t-1}), \\sigma^2)

    The prior for sigma is specified by the user, and $X_0$ is fixed at 0 but not included.
    Sampling is of the incremements, which are distributed as $N(0, \\sigma^2)$.
    """
    def __init__(self, sd_param, walk_length=None):
        """Main constructor
        
        Args:
            sd_param (Param): Parameter for the standard deviation of the increments.
            walk_length (int): Length of the walk.
        """
        self.sd_param = sd_param
        self.walk_length = walk_length
        self.sd_len = len(sd_param)

    def __len__(self):
        return self.walk_length + self.sd_len

    def log_prior_jacobian(self, value):
        sd_params = value[0:self.sd_len]
        log_walk_increments = value[self.sd_len:]
        log_sd_prior = self.sd_param.log_prior_jacobian(sd_params)
        sd = self.sd_param.sample_to_value(sd_params)
        log_walk_density = stats.norm.logpdf(log_walk_increments, 0.0, sd).sum()
        return log_sd_prior + log_walk_density

    def sample_to_value(self, sample):
        log_walk_increments = sample[self.sd_len:]
        return np.exp(log_walk_increments.cumsum())

    def value_to_sample(self, value, sd_value=None):
        """Information is lost (on the standard deviation when converting to a sample)"""
        if sd_value is None:
            raise NotImplementedError()
        sd_sample = self.sd_param.value_to_sample(sd_value)
        walk_sample = np.diff(
            np.log(value),
            prepend=[0]
        )
        return np.insert(walk_sample, 0, sd_sample)

    def draw_from_prior(self, random_state=None):
        sd_params = self.sd_param.draw_from_prior(random_state)
        sd = self.sd_param.sample_to_value(sd_params)
        log_walk_increments = stats.norm.rvs(size=self.walk_length, scale=sd, random_state=random_state)
        return np.insert(log_walk_increments, 0, sd_params)

class LogGaussianRW(LogGaussianRWParamSD):
    """Convenience class for LogGaussianRWParamSD to be called with a prior rather than a parameter."""
    def __init__(self, sd_prior, walk_length=None):
        """Main constructor
        
        Args:
            sd_prior (scipy.stats.distribution): Prior for the standard deviation of the increments.
            walk_length (int): Length of the walk.
        """
        sd_param = Param(sd_prior)
        super().__init__(sd_param, walk_length)

class LogGaussianRWNonCentred(LogGaussianRWParamSD):
    """Extension to LogGaussianRWParamSD which samples the using the non-centred parameterisation."""
    def __init__(self, sd_prior, walk_length=None, sample_log_scale=False):
        if sample_log_scale:
            sd_param = SampleLogScale(Param(sd_prior))
        else:
            sd_param = Param(sd_prior)
        super().__init__(sd_param, walk_length)
        assert self.sd_len == 1

    def log_prior_jacobian(self, value):
        sd_param = value[0]
        log_walk_increments = value[1:]
        log_sd_prior = self.sd_param.log_prior_jacobian(sd_param)
        log_walk_density = stats.norm.logpdf(log_walk_increments).sum()
        return log_sd_prior + log_walk_density

    def sample_to_value(self, sample):
        sd_param = sample[0]
        sd = self.sd_param.sample_to_value(sd_param)
        log_walk_increments = sample[1:] * sd
        return np.exp(log_walk_increments.cumsum())

    def draw_from_prior(self, random_state=None):
        sd_params = self.sd_param.draw_from_prior(random_state)
        log_walk_increments = stats.norm.rvs(size=self.walk_length, random_state=random_state)
        return np.insert(log_walk_increments, 0, sd_params)

    def value_to_sample(self, value, sd_value=None):
        """Information is lost (on the standard deviation when converting to a sample)"""
        if sd_value is None:
            raise NotImplementedError()
        sd_sample = self.sd_param.value_to_sample(sd_value)
        walk_sample = np.diff(
            np.log(value) / sd_value,
            prepend=[0]
        )
        return np.insert(walk_sample, 0, sd_sample)

class LogGaussianRWNonCentredInferSigma(LogGaussianRWParamSD):
    """A Gaussian random walk on the log scale where only the standard deviation is inferred."""
    def __init__(self, sd_prior, log_walk_increments, sample_log_scale=False):
        self.log_walk_increments = log_walk_increments
        if sample_log_scale:
            sd_param = SampleLogScale(Param(sd_prior))
        else:
            sd_param = Param(sd_prior)
        super().__init__(sd_param, len(log_walk_increments))

    def log_prior_jacobian(self, value):
        return self.sd_param.log_prior_jacobian(value)

    def sample_to_value(self, value):
        sd_param = value
        sd = self.sd_param.sample_to_value(sd_param)
        return np.exp(self.log_walk_increments.cumsum() * sd)

    def draw_from_prior(self, random_state=None):
        return self.sd_param.draw_from_prior(random_state)

    def __len__(self):
        return len(self.sd_param)

class LogGaussianRWFixedSD(LogGaussianRWParamSD):
    """A Gaussian random walk on the log scale where the standard deviation is known."""
    def __init__(self, sd_prior, walk_length=None):
        sd_param = FixedParam(sd_prior)
        super().__init__(sd_param, walk_length)

class _ExtendParam(Param):
    def __init__(self, param):
        """
        Args:
            param (Param): Parameter to be modified.
        """
        self._param = param

    def __len__(self):
        return len(self._param)

    def log_prior_jacobian(self, value):
        return self._param.log_prior_jacobian(value)

    def sample_to_value(self, value):
        return self._param.sample_to_value(value)

    def draw_from_prior(self, random_state=None):
        return self._param.draw_from_prior(random_state)

class SampleLogScale(_ExtendParam):
    """Sample a parameter on the log scale, with the prior on the natural scale."""
    def log_prior_jacobian(self, value):
        natural_value = np.exp(value)
        log_prior = super().log_prior_jacobian(natural_value)
        log_jacobian = value.sum()
        return log_prior + log_jacobian
    
    def value_to_sample(self, value):
        return super().value_to_sample(np.log(value))

    def sample_to_value(self, sample):
        return super().sample_to_value(np.exp(sample))

    def draw_from_prior(self, random_state=None):
        return np.log(super().draw_from_prior(random_state))


class SampleLogitScale(_ExtendParam):
    """Sample a parameter on the logit scale, with the prior on the natural scale."""
    def log_prior_jacobian(self, value):
        natural_value = expit(value)
        log_prior = super().log_prior_jacobian(natural_value)
        log_jacobian = (value - 2 * np.log(np.exp(value) + 1)).sum()
        return log_prior + log_jacobian

    def sample_to_value(self, sample):
        return super().sample_to_value(expit(sample))

    def value_to_sample(self, value):
        return super().sample_to_value(logit(value))

    def draw_from_prior(self, random_state=None):
        return logit(super().draw_from_prior(random_state))



class MatrixSusceptibleChildren(_ExtendParam):
    """A square matrix where the first few rows are a parameter and the rest are 1."""
    def __init__(self, param, n_strata, n_children_strata):
        """
        Args:
            param (Param): Parameter for the first rows.
            n_strata (int): Size of the matrix.
            n_children_strata (int): Number of rows which are the parameter value.
        """
        assert n_strata > n_children_strata
        super().__init__(param)
        self._n_strata = n_strata
        self._n_children_strata = n_children_strata

    def _value_to_matrix(self, value):
        children = np.broadcast_to(value, (self._n_children_strata, self._n_strata))
        adults = np.broadcast_to(1, (self._n_strata - self._n_children_strata, self._n_strata))
        return np.vstack([children, adults])

    def sample_to_value(self, sample):
        value = super().sample_to_value(sample)
        return self._value_to_matrix(value)
    
    def value_to_sample(self, value):
        raise NotImplementedError()