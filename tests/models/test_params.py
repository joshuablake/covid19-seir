from models import params
from hypothesis import given, strategies as st, assume
from hypothesis.extra.numpy import arrays
from pytest import approx
from scipy import stats
from scipy.special import expit, logit
import numpy as np

def test_vector_param_prior_mass():
    param = params.VectorParam(3, stats.norm())
    assert param.log_prior_jacobian([-3, 0, 3]) == approx(-0.9189385332046727 + 2 * -5.418938533204672)

def test_vector_param_length():
    param = params.VectorParam(3, stats.norm())
    assert len(param) == 3

@given(arrays(
    dtype=float,
    shape=st.integers(min_value=1, max_value=1000),
    elements=st.floats(allow_nan=False)
))
def test_vector_param_sample_to_value(value):
    param = params.VectorParam(value.shape[0], stats.norm())
    assert (param.sample_to_value(value) == value).all()

@given(arrays(
    dtype=float,
    shape=st.integers(min_value=1, max_value=1000),
    elements=st.floats(allow_nan=False)
))
def test_vector_param_value_to_sample(value):
    param = params.VectorParam(value.shape[0], stats.norm())
    assert (param.value_to_sample(value) == value).all()

def test_vector_param_multiple_prior_mass():
    param = params.VectorParamMultiplePriors(3, [stats.norm(loc=-3), stats.norm(), stats.norm(loc=3)])
    assert param.log_prior_jacobian([-3, 0, 3]) == approx(-0.9189385332046727 * 3)

@given(st.floats(allow_nan=False), st.floats())
def test_fixed_param_value(value1, value2):
    param = params.FixedParam(value1)
    assert param.sample_to_value(value2) == value1
    assert param.value_to_sample(value2).shape == (0,)
    assert len(param) == 0
    assert param.log_prior_jacobian(value2) == 0

@given(st.floats())
def test_fixed_param_value_nan(value2):
    param = params.FixedParam(np.nan)
    assert np.isnan(param.sample_to_value(value2))
    assert param.value_to_sample(value2).shape == (0,)
    assert len(param) == 0
    assert param.log_prior_jacobian(value2) == 0
    
def test_exp_after_prior_mass():
    param = params.ExpAfterPrior(stats.norm())
    assert param.log_prior_jacobian(0) == approx(-0.9189385332046727)
    assert param.log_prior_jacobian(-3) == approx(-5.418938533204672)
    assert param.log_prior_jacobian(3) == approx(-5.418938533204672)

@given(st.floats(allow_nan=False, allow_infinity=False))
def test_exp_after_sample_to_value(log_value):
    param = params.ExpAfterPrior(stats.norm())
    assert param.sample_to_value(log_value) == np.exp(log_value)

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=0, exclude_min=True))
def test_exp_after_value_to_sample(value):
    param = params.ExpAfterPrior(stats.norm())
    assert param.value_to_sample(value) == np.log(value)

def test_log_gaussian_RW_prior_mass():
    param = params.LogGaussianRW(stats.norm(1), 3)
    sample = np.array([1, 0, 0, 0])
    assert param.log_prior_jacobian(sample) == approx(-0.9189385332046727 * 4)
    sample[1:] = 3
    assert param.log_prior_jacobian(sample) == approx(-0.9189385332046727 -5.418938533204672 * 3)

@given(
    sd=st.floats(allow_nan=False, allow_infinity=False, min_value=0, exclude_min=True),
    walk=arrays(
        dtype=float,
        shape=st.integers(min_value=1, max_value=1000),
        elements=st.floats(allow_nan=False, allow_infinity=False),
    )
)
def test_log_gaussian_RW_sample_to_value(sd, walk):
    param = params.LogGaussianRW(stats.norm(1), walk.shape[0])
    sample = np.insert(walk, 0, sd)
    assert param.sample_to_value(sample) == approx(np.exp(walk.cumsum()))

@given(
    sd=st.floats(allow_nan=False, allow_infinity=False, min_value=0, exclude_min=True),
    walk=arrays(
        dtype=float,
        shape=st.integers(min_value=1, max_value=500),
        elements=st.floats(allow_nan=False, allow_infinity=False),
    )
)
def test_log_gaussian_RW_value_to_sample(sd, walk):
    assume(np.isfinite(np.exp(np.cumsum(walk))).all())
    assume(np.isfinite(np.log(np.exp(np.cumsum(walk)))).all())
    tolerance = abs(np.log(np.exp(np.sum(walk))) - np.sum(walk)) + 1e-12
    param = params.LogGaussianRW(stats.norm(1), walk.shape[0])
    sample = np.insert(walk, 0, sd)
    assert param.value_to_sample(param.sample_to_value(sample), sd_value=sd) == approx(sample, abs=tolerance, rel=1e-5)

def test_log_gaussian_RW_noncentered_prior_mass():
    param = params.LogGaussianRWNonCentred(stats.norm(1), 3)
    sample = np.array([1, 0, 0, 0])
    assert param.log_prior_jacobian(sample) == approx(-0.9189385332046727 * 4)
    sample[1:] = 3
    assert param.log_prior_jacobian(sample) == approx(-0.9189385332046727 -5.418938533204672 * 3)

@given(
    sd=st.floats(allow_nan=False, allow_infinity=False, min_value=0, exclude_min=True),
    walk=arrays(
        dtype=float,
        shape=st.integers(min_value=1, max_value=1000),
        elements=st.floats(allow_nan=False, allow_infinity=False),
    )
)
def test_log_gaussian_RW_noncentered_sample_to_value(sd, walk):
    param = params.LogGaussianRWNonCentred(stats.norm(1), walk.shape[0])
    sample = np.insert(walk, 0, sd)
    expected = np.exp((walk * sd).cumsum())
    assume(np.isfinite(expected).all())
    assert param.sample_to_value(sample) == approx(expected)

@given(
    sd=st.floats(allow_nan=False, allow_infinity=False, min_value=1e-5),
    walk=arrays(
        dtype=float,
        shape=st.integers(min_value=1, max_value=500),
        elements=st.floats(allow_nan=False, allow_infinity=False),
    )
)
def test_log_gaussian_RW_noncentered_value_to_sample(sd, walk):
    assume(np.isfinite(np.exp(np.cumsum(walk * sd))).all())
    assume(np.isfinite(np.log(np.exp(np.cumsum(walk * sd)))).all())
    tolerance = abs(np.log(np.exp(np.sum(walk * sd))) - np.sum(walk * sd)) + 1e-10
    param = params.LogGaussianRWNonCentred(stats.norm(1), walk.shape[0])
    sample = np.insert(walk, 0, sd)
    actual = param.value_to_sample(param.sample_to_value(sample), sd_value=sd)
    expected = approx(sample, abs=tolerance, rel=1e-5)
    assert actual == expected

def test_sample_log_scale_prior_mass():
    param = params.SampleLogScale(params.VectorParam(3, stats.norm(1)))
    sample = np.array([0, 0, 0])
    assert param.log_prior_jacobian(sample) == approx(-0.9189385332046727 * 3)
    param = params.SampleLogScale(params.VectorParam(3, stats.norm(np.exp(1))))
    assert param.log_prior_jacobian(sample + 1) == approx(-0.9189385332046727 *3 + 3)

@given(
    samples=arrays(
        dtype=float,
        shape=st.integers(min_value=1, max_value=1000),
        elements=st.floats(allow_nan=False, allow_infinity=False),
    )
)
def test_sample_log_scale_sample_to_value(samples):
    param = params.SampleLogScale(params.VectorParam(samples.shape[0], stats.norm(1)))
    assert param.sample_to_value(samples) == approx(np.exp(samples))

@given(
    samples=arrays(
        dtype=float,
        shape=st.integers(min_value=1, max_value=1000),
        elements=st.floats(allow_nan=False, allow_infinity=False),
    )
)
def test_sample_log_scale_value_to_sample(samples):
    assume(np.isfinite(np.log(np.exp(samples))).all())
    param = params.SampleLogScale(params.VectorParam(samples.shape[0], stats.norm(1)))
    assert param.value_to_sample(param.sample_to_value(samples)) == approx(samples)

def test_sample_logit_scale_prior_mass():
    param = params.SampleLogitScale(params.VectorParam(3, stats.norm(0.5)))
    sample = np.array([0, 0, 0])
    assert param.log_prior_jacobian(sample) == approx(-0.9189385332046727 * 3 - 3 * (2 * np.log(2)))
    param = params.SampleLogitScale(params.VectorParam(3, stats.norm(expit(1))))
    assert param.log_prior_jacobian(sample + 1) == approx(-0.9189385332046727 * 3 + 3 * (1 - 2 * np.log(1 + np.exp(1))))

@given(
    samples=arrays(
        dtype=float,
        shape=st.integers(min_value=1, max_value=1000),
        elements=st.floats(allow_nan=False, allow_infinity=False),
    )
)
def test_sample_logit_scale_sample_to_value(samples):
    param = params.SampleLogitScale(params.VectorParam(samples.shape[0], stats.norm(1)))
    assert param.sample_to_value(samples) == approx(expit(samples))

@given(
    samples=arrays(
        dtype=float,
        shape=st.integers(min_value=1, max_value=1000),
        elements=st.floats(allow_nan=False, allow_infinity=False),
    )
)
def test_sample_logit_scale_value_to_sample(samples):
    assume(np.isfinite(logit(expit(samples))).all())
    param = params.SampleLogitScale(params.VectorParam(samples.shape[0], stats.norm(1)))
    assert param.value_to_sample(param.sample_to_value(samples)) == approx(samples)