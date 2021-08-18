import numpy as np
from hypothesis import given, strategies as st, assume, settings
from models.prevalence_model import SingleRegionStratifiedSEIRCompartmental
from pytest import approx, mark

@st.composite
def simplex(draw, size, total=1):
    remains = total
    result = []
    for _ in range(size - 1):
        new = draw(st.floats(min_value=0, max_value=remains))
        remains -= new
        result.append(new)
    result.append(remains)
    return result

@st.composite
def seir_state(draw, num_strata = None, P = [0] * 7):
    if num_strata is None:
        num_strata = draw(st.integers(min_value=1, max_value=100))
    values = [draw(simplex(6)) + P for _ in range(num_strata)]
    return np.column_stack(values)

# Matrix array for one step and one stratum
MATRICES_SINGLE = np.array([[[1]]])

def test_inital_state1():
    model = SingleRegionStratifiedSEIRCompartmental(MATRICES_SINGLE, 1, dt=1)
    init_state, init_R0 = model._init_transmission(2, 2, 1, 1/100, 0.9, MATRICES_SINGLE[0])
    assert init_R0 == approx(32/3)
    assert model.check_state(init_state)
    assert model.get_compartment(init_state, "I1") + model.get_compartment(init_state, "I2") == approx(1/100)
    assert model.get_compartment(init_state, "I1") == approx(1/100 * (1 + np.exp(-1)) ** -1)
    assert model.get_compartment(init_state, "E2") == approx(model.get_compartment(init_state, "I1") * np.exp(1))
    assert model.get_compartment(init_state, "E1") == approx(model.get_compartment(init_state, "E2") * np.exp(1))
    assert model.get_compartment(init_state, "R") == approx(0.1)
    assert model._num_betas == 0

def test_inital_state2():
    model = SingleRegionStratifiedSEIRCompartmental(np.ones((2, 1, 1)), 10, dt=0.5)
    init_state, init_R0 = model._init_transmission(1, 2, 0.4, 1/1000, 0.95, MATRICES_SINGLE[0])
    assert init_R0 == approx(2.352)
    assert model.check_state(init_state)
    assert model.get_compartment(init_state, "I1") + model.get_compartment(init_state, "I2") == approx(1/1000)
    assert model.get_compartment(init_state, "I1") == approx(1/1000 * (1 + np.exp(-0.2)) ** -1)
    assert model.get_compartment(init_state, "E2") == approx(model.get_compartment(init_state, "I1") * np.exp(0.2) / 2)
    assert model.get_compartment(init_state, "E1") == approx(model.get_compartment(init_state, "E2") * (np.exp(0.2) + 1) / 2)
    assert model.get_compartment(init_state, "R") == approx(0.05)
    assert model._num_betas == 1

@given(n_strata = st.integers(min_value=1, max_value=100))
def test_inital_state2_multi_strata(n_strata):
    model = SingleRegionStratifiedSEIRCompartmental(np.ones((2, n_strata, n_strata)), 10, dt=0.5)
    init_state, matrix_multiplier = model._init_transmission(1, 2, 0.4, 1/1000, np.repeat(0.95, n_strata), np.ones((n_strata, n_strata)))
    assert matrix_multiplier == approx(2.352 / n_strata)
    assert model.check_state(init_state)
    assert model.get_compartment(init_state, "I1") + model.get_compartment(init_state, "I2") == approx(1/1000)
    assert model.get_compartment(init_state, "I1") == approx(1/1000 * (1 + np.exp(-0.2)) ** -1)
    assert model.get_compartment(init_state, "E2") == approx(model.get_compartment(init_state, "I1") * np.exp(0.2) / 2)
    assert model.get_compartment(init_state, "E1") == approx(model.get_compartment(init_state, "E2") * (np.exp(0.2) + 1) / 2)
    assert model.get_compartment(init_state, "R") == approx(0.05)
    assert model._num_betas == 1

@given(n_strata = st.integers(min_value=1, max_value=100))
def test_inital_shrinking_epidemic(n_strata):
    model = SingleRegionStratifiedSEIRCompartmental(np.ones((1, n_strata, n_strata)), 1, dt=1)
    init_state, matrix_multiplier = model._init_transmission(2, 2, -0.5, 1/1000, np.ones(n_strata), np.ones((n_strata, n_strata)))
    assert matrix_multiplier == approx(1/12 / n_strata)
    assert model.check_state(init_state)
    assert model.get_compartment(init_state, "I1") + model.get_compartment(init_state, "I2") == approx(1/1000)
    assert model.get_compartment(init_state, "I1") == approx(1/1000 * (1 + np.exp(0.5)) ** -1)
    assert model.get_compartment(init_state, "E2") == approx(model.get_compartment(init_state, "I1") * np.exp(-0.5))
    assert model.get_compartment(init_state, "E1") == approx(model.get_compartment(init_state, "E2") * np.exp(-0.5))
    assert model.get_compartment(init_state, "R") == approx(0)

@given(
    max_time = st.integers(min_value = 1, max_value = 10_000),
    dt_over_1 = st.integers(min_value = 1),
    dL = st.floats(min_value = 0.001),
    dI = st.floats(min_value = 0.001),
    psir = st.floats(),
    i0 = st.floats(),
    pi = st.floats(),
    n_strata = st.integers(min_value=1, max_value=100)
)
def test_init_generated(max_time, dt_over_1, dL, dI, psir, i0, pi, n_strata):
    dt = 1 / dt_over_1
    beta_step_spacing=1
    try:
        model = SingleRegionStratifiedSEIRCompartmental(np.ones((max_time, n_strata, n_strata)), max_time, dt, beta_step_spacing)
        init_state, matrix_multiplier = model._init_transmission(dL, dI, psir, i0, pi, np.ones((n_strata, n_strata)))
    except ValueError:
        return
    assert matrix_multiplier >= 0
    assert model.check_state(init_state)

"""Try taking a single step where everyone moves to the next compartment but no new infections"""
@given(
    seir_state(),
    st.floats(min_value=1/1000, max_value=1e30)
)
def test_single_step_no_infections_full_move_dt_half(start_state, dt):
    n_strata = start_state.shape[1]
    max_time = max(2 * dt, 2)
    num_matrices = int((max_time - dt) // dt) + 1
    model = SingleRegionStratifiedSEIRCompartmental(np.ones((num_matrices, n_strata, n_strata)), max_time, dt, beta_step_spacing=dt)
    final_state = model._single_step(start_state, np.zeros((n_strata, n_strata)), 2*dt, 2*dt)
    assert model.check_state(final_state)
    expected = np.vstack((
        model.get_compartment(start_state, "S"), # S
        np.zeros((1, n_strata)), # E1
        model.get_compartment(start_state, "E1"), # E2
        model.get_compartment(start_state, "E2"), # I1
        model.get_compartment(start_state, "I1"), # I2
        model.get_compartment(start_state, "R") + model.get_compartment(start_state, "I2"), # R
    ))
    actual = final_state[:6,:]
    assert np.allclose(actual, expected)

@given(seir_state(), st.floats(min_value=0.1, max_value=5), st.floats(min_value=0, max_value=100))
def test_reduction_in_S(start_state, dt, beta_t):
    model = SingleRegionStratifiedSEIRCompartmental(2*dt, dt, beta_step_spacing=dt)
    final_state = model._single_step(start_state, beta_t, 2*dt, 2*dt)
    assert model.check_state(final_state)
    actual_reduction = start_state.S - final_state.S
    expected_reduction = beta_t * start_state.S * (start_state.I1 + start_state.I2) * dt
    if expected_reduction > start_state.S:
        expected_reduction = start_state.S
    assert np.isclose(actual_reduction, expected_reduction)

@given(
    #num_steps = st.integers(min_value=142, max_value=1000),
    dt_over_1 = st.integers(min_value=1, max_value=10),
    beta = st.lists(st.integers(min_value=0, max_value=100), min_size=142, max_size=142),
    dL = st.floats(min_value=0, max_value=10, exclude_min=True),
    dI = st.floats(min_value=0, max_value=10, exclude_min=True),
    psir = st.floats(min_value=-1, max_value=1),
    i0 = st.floats(min_value=0, max_value=0.1),
    pi = st.floats(min_value=0, max_value=1)
)
@settings(deadline=1000, max_examples=10)
def test_simulate_epidemic(dt_over_1, beta, dL, dI, psir, i0, pi):
    max_time = 1000
    #dt = max_time / num_steps
    dt = 1 / dt_over_1
    try:
        model = SingleRegionStratifiedSEIRCompartmental(max_time, dt)
        nni, prev, final_state = model.simulate(beta, dL, dI, psir, i0, pi)
    except ValueError:
        assume(False)
    assert len(nni) == max_time
    assert len(prev) == max_time
    assert model.check_state(final_state)
    assert np.all(np.logical_or(0 < nni, np.isclose(nni, 0)))
    assert np.all(nni < final_state.R)
    assert np.all(np.logical_or(0 < prev, np.isclose(prev, 0)))
    assert np.all(prev < final_state.R)

@given(st.integers(min_value=1, max_value=100))
def test_num_betas(n_strata):
    assert SingleRegionStratifiedSEIRCompartmental(np.ones((1, n_strata, n_strata)), 5, 0.5)._num_betas == 0
    assert SingleRegionStratifiedSEIRCompartmental(np.ones((2, n_strata, n_strata)), 5, 0.5, 4)._num_betas == 1
    assert SingleRegionStratifiedSEIRCompartmental(np.ones((3, n_strata, n_strata)), 5, 0.5, 2)._num_betas == 2
    assert SingleRegionStratifiedSEIRCompartmental(np.ones((1, n_strata, n_strata)), 7, 0.5)._num_betas == 0