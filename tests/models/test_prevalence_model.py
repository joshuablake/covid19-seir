import numpy as np
from hypothesis import given, strategies as st, assume, settings
from models.prevalence_model import SingleRegionSingleAgeSEIRCompartmental, SEIR_state
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
def seir_state(draw, P = [0] * 7):
    values = draw(simplex(6)) + P
    return SEIR_state(*values)

def test_inital_state1():
    model = SingleRegionSingleAgeSEIRCompartmental(1, dt=1)
    init_state, init_R0 = model._init_transmission(2, 2, 1, 0.5, 0.9)
    assert init_R0 == approx(32/3)
    assert model.check_state(init_state)
    assert init_state.I1 + init_state.I2 == approx(3/32)
    assert init_state.I1 == approx(3/32 * (1 + np.exp(-1)) ** -1)
    assert init_state.E2 == approx(init_state.I1 * np.exp(1))
    assert init_state.E1 == approx(init_state.E2 * np.exp(1))
    assert init_state.R == approx(0.1)
    assert model.num_betas == 0

def test_inital_call_numpy():
    model = SingleRegionSingleAgeSEIRCompartmental(1, dt=1)
    init_state, init_R0 = model._init_transmission(2, 2, np.array([1]), np.array([0.5]),
        np.array([0.9]))
    assert init_R0 == approx(32/3)
    assert model.check_state(init_state)
    assert init_state.I1 + init_state.I2 == approx(3/32)
    assert init_state.I1 == approx(3/32 * (1 + np.exp(-1)) ** -1)
    assert init_state.E2 == approx(init_state.I1 * np.exp(1))
    assert init_state.E1 == approx(init_state.E2 * np.exp(1))
    assert init_state.R == approx(0.1)
    assert model.num_betas == 0

def test_inital_state2():
    model = SingleRegionSingleAgeSEIRCompartmental(10, dt=0.5)
    init_state, init_R0 = model._init_transmission(1, 2, 0.4, 0.5, 0.95)
    assert init_R0 == approx(2.352)
    assert model.check_state(init_state)
    assert init_state.I1 + init_state.I2 == approx(1/2.352)
    assert init_state.I1 == approx(1/2.352 * (1 + np.exp(-0.2)) ** -1)
    assert init_state.E2 == approx(init_state.I1 * np.exp(0.2) / 2)
    assert init_state.E1 == approx(init_state.E2 * (np.exp(0.2) + 1) / 2)
    assert init_state.R == approx(0.05)
    assert model.num_betas == 1

@given(
    max_time = st.integers(min_value = 1, max_value = 10_000),
    dt_over_1 = st.integers(min_value = 1),
    dL = st.floats(min_value = 0.001),
    dI = st.floats(min_value = 0.001),
    psir = st.floats(),
    i0 = st.floats(),
    pi = st.floats()
)
def test_init_generated(max_time, dt_over_1, dL, dI, psir, i0, pi):
    dt = 1 / dt_over_1
    beta_step_spacing=1
    try:
        model = SingleRegionSingleAgeSEIRCompartmental(max_time, dt, beta_step_spacing)
        init_state, init_R0 = model._init_transmission(dL, dI, psir, i0, pi)
    except ValueError:
        return
    assert init_R0 >= 0
    assert model.check_state(init_state)

"""Try taking a single step where everyone moves to the next compartment but no new infections"""
@given(seir_state(), st.floats(min_value=1/1000, max_value=1e30))
def test_single_step_no_infections_full_move_dt_half(start_state, dt):
    max_time = max(2 * dt, 2)
    model = SingleRegionSingleAgeSEIRCompartmental(max_time, dt, beta_step_spacing=dt)
    final_state = model._single_step(start_state, 0, 2*dt, 2*dt)
    assert model.check_state(final_state)
    expected = (
        start_state.S, # S
        0, # E1
        start_state.E1, # E2
        start_state.E2, # I1
        start_state.I1, # I2
        start_state.R + start_state.I2 # R
    )
    actual = final_state[:6]
    assert np.allclose(actual, expected)

@given(seir_state(), st.floats(min_value=0.1, max_value=5), st.floats(min_value=0, max_value=100))
def test_reduction_in_S(start_state, dt, beta_t):
    model = SingleRegionSingleAgeSEIRCompartmental(2*dt, dt, beta_step_spacing=dt)
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
    psir = st.floats(min_value=0, max_value=0.5, exclude_min=True),
    i0 = st.floats(min_value=1e-9, max_value=0.1),
    pi = st.floats(min_value=0.5, max_value=1),
    data = st.data()
)
@settings(deadline=1000, max_examples=10)
def test_simulate_epidemic(dt_over_1, beta, psir, i0, pi, data):
    max_time = 1000
    #dt = max_time / num_steps
    dt = 1 / dt_over_1
    dL = data.draw(st.floats(min_value=dt*4, max_value=10))
    dI = data.draw(st.floats(min_value=dt*4, max_value=10, exclude_max=True))
    try:
        model = SingleRegionSingleAgeSEIRCompartmental(max_time, dt)
        nni, prev, final_state = model.simulate(beta, dL, dI, psir, i0, pi)
    except ValueError as e:
        assume(False)
    assert len(nni) == max_time
    assert len(prev) == max_time
    assert model.check_state(final_state)
    assert np.all(np.logical_or(0 < nni, np.isclose(nni, 0)))
    assert np.all(nni < final_state.R)
    assert np.all(np.logical_or(0 < prev, np.isclose(prev, 0)))
    assert np.all(prev < final_state.R)

def test_num_betas():
    assert SingleRegionSingleAgeSEIRCompartmental(5, 0.5).num_betas == 0
    assert SingleRegionSingleAgeSEIRCompartmental(5, 0.5, 4).num_betas == 1
    assert SingleRegionSingleAgeSEIRCompartmental(5, 0.5, 2).num_betas == 2
    assert SingleRegionSingleAgeSEIRCompartmental(7, 0.5).num_betas == 0