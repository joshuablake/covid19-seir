import numpy as np
from hypothesis import given, strategies as st
from models.prevalence_model import SingleRegionSingleAgeSEIRCompartmental
from pytest import approx, mark

@st.composite
def simplex(draw, size, total=1):
    remains = total
    result = []
    for _ in range(size) - 1:
        new = draw(st.floats(min_value=0, max_value=remains))
        remains -= new
        result.append(new)
    result.append(remains)
    return result

def test_inital1():
    model = SingleRegionSingleAgeSEIRCompartmental(1, dt=1)
    init_state, init_R0 = model._init_transmission(2, 2, 1, 0.5, 0.9)
    assert init_R0 == approx(32/3)
    assert model.check_state(init_state)
    assert init_state.I1 + init_state.I2 == approx(3/32)
    assert init_state.I1 == approx(3/32 * (1 + np.exp(-1)) ** -1)
    assert init_state.E2 == approx(init_state.I1 * np.exp(1))
    assert init_state.E1 == approx(init_state.E2 * np.exp(1))
    assert init_state.R == approx(0.1)

def test_inital2():
    model = SingleRegionSingleAgeSEIRCompartmental(10, dt=0.5)
    init_state, init_R0 = model._init_transmission(1, 2, 0.4, 0.5, 0.95)
    assert init_R0 == approx(2.352)
    assert model.check_state(init_state)
    assert init_state.I1 + init_state.I2 == approx(1/2.352)
    assert init_state.I1 == approx(1/2.352 * (1 + np.exp(-0.2)) ** -1)
    assert init_state.E2 == approx(init_state.I1 * np.exp(0.2) / 2)
    assert init_state.E1 == approx(init_state.E2 * (np.exp(0.2) + 1) / 2)
    assert init_state.R == approx(0.05)

def test_inital_shrinking():
    model = SingleRegionSingleAgeSEIRCompartmental(1, dt=1)
    init_state, init_R0 = model._init_transmission(2, 2, -0.5, 1/1000, 1)
    assert init_R0 == approx(1/12)
    assert model.check_state(init_state)
    assert init_state.I1 + init_state.I2 == approx(24e-3)
    assert init_state.I1 == approx(24e-3 * (1 + np.exp(0.5)) ** -1)
    assert init_state.E2 == approx(init_state.I1 * np.exp(-0.5))
    assert init_state.E1 == approx(init_state.E2 * np.exp(-0.5))
    assert init_state.R == approx(0)

@given(
    max_time = st.integers(min_value = 1, max_value = 10_000),
    dt_over_1 = st.integers(min_value = 1),
    beta_step_spacing = st.integers(min_value = 1, max_value = 100),
    dL = st.floats(min_value = 0.001),
    dI = st.floats(min_value = 0.001),
    psir = st.floats(),
    i0 = st.floats(),
    pi = st.floats()
)
def test_init_generated(max_time, dt_over_1, beta_step_spacing, dL, dI, psir, i0, pi):
    dt = 1 / dt_over_1
    try:
        model = SingleRegionSingleAgeSEIRCompartmental(max_time, dt, beta_step_spacing)
        init_state, init_R0 = model._init_transmission(dL, dI, psir, i0, pi)
    except ValueError:
        return
    assert init_R0 >= 0
    assert model.check_state(init_state)
