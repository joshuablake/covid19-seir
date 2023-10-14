import numpy as np
from hypothesis.extra.numpy import arrays
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

@st.composite
def matrices(draw, n_strata = None, max_strata = 100, num_matrices = 1):
    if n_strata is None:
        n_strata = draw(st.integers(min_value=1, max_value=max_strata))
    if num_matrices == 1:
        return draw(arrays(
            np.float64,
            (n_strata, n_strata),
            elements=st.floats(min_value=0.001, max_value=50)
        ))
    return draw(arrays(
        np.float64,
        (num_matrices, n_strata, n_strata),
        elements=st.floats(min_value=0.001, max_value=50)
    ))

@st.composite
def state_and_matrix(draw):
    n_strata = draw(st.integers(min_value=1, max_value=100))
    state = draw(seir_state(n_strata))
    matrix = draw(matrices(n_strata, 1))
    return state, matrix

# Matrix array for one step and one stratum
MATRICES_SINGLE = np.array([[[1]]])

def test_inital_state1():
    model = SingleRegionStratifiedSEIRCompartmental(MATRICES_SINGLE, 1, dt=1)
    init_state, init_R0 = model._init_transmission(2, 2, 1, 1/100, 0.9, MATRICES_SINGLE[0])
    assert init_R0 == approx(32/3)
    assert model.check_state(init_state)
    assert model.get_compartment(init_state, "I1") + model.get_compartment(init_state, "I2") == approx(1/100)
    assert model.get_compartment(init_state, "I1") == approx(1/100 * 2/3)
    assert model.get_compartment(init_state, "E2") == approx(model.get_compartment(init_state, "I1") * 2)
    assert model.get_compartment(init_state, "E1") == approx(model.get_compartment(init_state, "E2") * 2)
    assert model.get_compartment(init_state, "R") == approx(0.1)
    assert model.num_betas == 0

def test_inital_state2():
    model = SingleRegionStratifiedSEIRCompartmental(np.ones((2, 1, 1)), 10, dt=0.5)
    init_state, init_R0 = model._init_transmission(1, 2, 0.4, 1/1000, 0.95, MATRICES_SINGLE[0])
    assert init_R0 == approx(2.352)
    assert model.check_state(init_state)
    assert model.get_compartment(init_state, "I1") + model.get_compartment(init_state, "I2") == approx(1/1000)
    assert model.get_compartment(init_state, "I1") == approx(1/1000 * (1 + 1/1.4) ** -1)
    assert model.get_compartment(init_state, "E2") == approx(model.get_compartment(init_state, "I1") * 0.7)
    assert model.get_compartment(init_state, "E1") == approx(model.get_compartment(init_state, "E2") * 1.2)
    assert model.get_compartment(init_state, "R") == approx(0.05)
    assert model.num_betas == 1

@given(n_strata = st.integers(min_value=1, max_value=100))
def test_inital_state2_multi_strata(n_strata):
    model = SingleRegionStratifiedSEIRCompartmental(np.ones((2, n_strata, n_strata)), 10, dt=0.5)
    init_state, matrix_multiplier = model._init_transmission(1, 2, 0.4, 1/1000, np.repeat(0.95, n_strata), np.ones((n_strata, n_strata)))
    assert matrix_multiplier == approx(2.352 / n_strata)
    assert model.check_state(init_state)
    assert model.get_compartment(init_state, "I1") + model.get_compartment(init_state, "I2") == approx(1/1000)
    assert model.get_compartment(init_state, "I1") == approx(1/1000 * (1 + 1/1.4) ** -1)
    assert model.get_compartment(init_state, "E2") == approx(model.get_compartment(init_state, "I1") * 0.7)
    assert model.get_compartment(init_state, "E1") == approx(model.get_compartment(init_state, "E2") * 1.2)
    assert model.get_compartment(init_state, "R") == approx(0.05)
    assert model.num_betas == 1

@given(
    max_time = st.integers(min_value = 1, max_value = 10_000),
    dt_over_1 = st.integers(min_value = 1),
    psir = st.floats(min_value=0, max_value=0.5, exclude_min=True),
    i0 = st.floats(min_value=1e-9, max_value=0.05),
    n_strata = st.integers(min_value=1, max_value=100),
    data = st.data()
)
def test_init_generated(max_time, dt_over_1, psir, i0, n_strata, data):
    dt = 1 / dt_over_1
    beta_step_spacing=1
    if psir < 0:
        dI_max = min(10, 2 / -psir)
    else:
        dI_max = 10
    dL = data.draw(st.floats(min_value=dt*2, max_value=10))
    dI = data.draw(st.floats(min_value=dt*2, max_value=dI_max, exclude_max=True))
    pi = data.draw(arrays(np.float64, (n_strata,), elements=st.floats(min_value=0.5, max_value=1)))
    try:
        model = SingleRegionStratifiedSEIRCompartmental(np.ones((max_time, n_strata, n_strata)), max_time, dt, beta_step_spacing)
        init_state, matrix_multiplier = model._init_transmission(dL, dI, psir, i0, pi, np.ones((n_strata, n_strata)))
    except ValueError:
        assume(False)
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
    linear_transitions = model._calc_linear_transition_matrix(2*dt, 2*dt)
    final_state = model._single_step(start_state, np.zeros((n_strata, n_strata)), linear_transitions)
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

@given(
    state_and_matrix(),
    st.floats(min_value=0.1, max_value=5)
)
def test_reduction_in_S(config, dt):
    start_state, B_t = config
    n_strata = start_state.shape[1]
    max_time = 20 * dt
    num_matrices = int((max_time - dt) // dt) + 1
    model = SingleRegionStratifiedSEIRCompartmental(np.ones((num_matrices, n_strata, n_strata)), max_time, dt, beta_step_spacing=dt)
    linear_transitions = model._calc_linear_transition_matrix(2*dt, 2*dt)
    final_state = model._single_step(start_state, B_t, linear_transitions)
    assert model.check_state(final_state)
    actual_reduction = model.get_compartment(start_state, "S") - model.get_compartment(final_state, "S")
    inital_infected = model.get_compartment(start_state, "I1") + model.get_compartment(start_state, "I2")
    expected_reduction = model.get_compartment(start_state, "S") * dt * (B_t @ inital_infected)
    expected_reduction = np.minimum(expected_reduction, model.get_compartment(start_state, "S"))
    assert np.allclose(actual_reduction, expected_reduction)

@given(
    #num_steps = st.integers(min_value=142, max_value=1000),
    dt_over_1 = st.integers(min_value=1, max_value=10),
    beta = st.lists(st.integers(min_value=0, max_value=100), min_size=5, max_size=5),
    B = matrices(num_matrices=6, max_strata=10),
    psir = st.floats(min_value=0, max_value=0.5, exclude_min=True),
    i0 = st.floats(min_value=1e-9, max_value=0.01),
    data = st.data()
)
@settings(deadline=1000)
def test_simulate_epidemic(dt_over_1, beta, B, psir, i0, data):
    dt = 1 / dt_over_1
    n_strata = B.shape[1]
    modifiers = data.draw(arrays(np.float64, (n_strata, n_strata), elements=st.floats(min_value=0.1, max_value=10)))
    max_time = 500
    dI = data.draw(st.floats(min_value=dt*4, max_value=10, exclude_max=True))
    pi = data.draw(arrays(np.float64, (n_strata,), elements=st.floats(min_value=0.5, max_value=1)))
    model = SingleRegionStratifiedSEIRCompartmental(B, max_time, dt, 90)
    try:
        nni, prev, final_state = model.simulate(beta, dL, dI, psir, i0, pi, modifiers)
    except ValueError:
        assume(False)
    num_recovered = model.get_compartment(final_state, "R")
    assert np.logical_or(num_recovered < 1, np.isclose(num_recovered, 1)).all()
    assert nni.shape == prev.shape == (max_time, n_strata)
    assert model.check_state(final_state)
    assert (np.logical_or(0 < nni, np.isclose(nni, 0))).all()
    assert (nni < num_recovered).all()
    assert (np.logical_or(0 < prev, np.isclose(prev, 0))).all()
    assert (prev < num_recovered + prev[-1]).all()

@given(st.integers(min_value=1, max_value=100))
def test_num_betas(n_strata):
    assert SingleRegionStratifiedSEIRCompartmental(np.ones((1, n_strata, n_strata)), 5, 0.5).num_betas == 0
    assert SingleRegionStratifiedSEIRCompartmental(np.ones((2, n_strata, n_strata)), 5, 0.5, 4).num_betas == 1
    assert SingleRegionStratifiedSEIRCompartmental(np.ones((3, n_strata, n_strata)), 5, 0.5, 2).num_betas == 2
    assert SingleRegionStratifiedSEIRCompartmental(np.ones((1, n_strata, n_strata)), 7, 0.5).num_betas == 0