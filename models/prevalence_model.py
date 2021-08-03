from collections import namedtuple
import numpy as np
from numpy.core.numeric import isclose

P_STATES = 7
P_TRANSITIONS = np.array([
    -1/3, 0, 0, 0, 0, 0, 0,
    1/3, -0.2279314474033871,0,0,0,0,0,
    0, 0.2279314474033871,-0.2279314474033871,0,0,0,0,
    0, 0,0.2279314474033871,-0.2279314474033871,0,0,0,
    0, 0,0,0.2279314474033871,-0.2279314474033871,0,0,
    0, 0,0,0,0.2279314474033871,-1.1287836677440286,0,
    0, 0,0,0,0,3.1437857603953896e-4,-0.013204089629658078
]).reshape((P_STATES, P_STATES))

SEIR_state = namedtuple(
    "SEIR_state",
    ["S", "E1", "E2", "I1", "I2", "R"] + ["P{}".format(i) for i in range(7)]
)

class SingleRegionSingleAgeSEIRCompartmental(object):
    """Create a model, with configuration that doesn't change.

    Params:
        max_time: number of days to run for
        dt: size of timestep for ODE discreteisation (in days)
        beta_step_spacing: days between changes in the beta parameter
        pi: proportion of the population that is infected initally
    """
    def __init__(self, max_time, dt = 0.5, beta_step_spacing = 7):
        if max_time < dt:
            raise ValueError("Need at least one step")
        self._dt = dt
        self._max_time = max_time
        self._num_steps = int(np.round(max_time / dt))
        if self._num_steps > 10_000:
            raise ValueError("Too many steps required")
        assert np.isclose(self._num_steps * self._dt, self._max_time)
        time_of_step = np.linspace(0, max_time - self._dt, self._num_steps)
        # Index of beta value for each timestep
        self._beta_of_step = [int(i) for i in time_of_step // beta_step_spacing]
        self._num_betas = self._beta_of_step[-1]
        if self._num_steps < self._num_betas + 1 or not np.isclose(beta_step_spacing % dt, 0):
            raise ValueError("beta_step_spacing should be a multiple of dt")

    def check_state(self, state):
        return (
            np.isclose(sum(state[:6]), 1) and
            np.all(np.logical_or(np.array(state) >= 0, np.isclose(state, 0)))
        )

    """Calculate the system state at time 0.
    
    Params:
        dL: length of the latent period
        dI: length of the infectious period
        psir: initial rate of exponential growth?
        i0: proportion of population initially infected 
        pi: proportion of population with prior immunity
    """
    def _init_transmission(self, dL, dI, psir, i0, pi):
        if not (
                # If either period too short than finite differences is bad
                self._dt <= dL and
                self._dt <= dI and
                # Avoid overflows
                abs(psir) <= 10 and
                # Positive number of initial exposed and infecteds
                i0 >= 0 and
                (psir >= 0 or (dL < 2/(1-np.exp(psir*self._dt)) and
                dI < 2/(1-np.exp(psir*self._dt)))) and
                # pi is a proportion
                0 <= pi <= 1 and
                # Require R0 > 0
                (psir >= 0 or dI * -psir < 2)
            ):
            raise ValueError("Impossible values for model")

        # Change periods into rates per day (not per step as elsewhere)
        sigma = 2./dL
        gamma = 2./dI
        
        # See PNAS paper for explanation of parameterisation
        if np.isclose(psir, 0):
            r0_init = 1
        else:
            r0_init = psir*dI*((psir*dL/2 + 1)**2)/ (1- (1/(psir*dI/2 + 1)**2))
        if dI * i0 > r0_init:
            raise ValueError("Hazard too high: would mean more than whole population infected")
        alpha = np.exp(psir*self._dt) - 1
        I_init = (dI * i0) / r0_init
        i1_init = I_init*(1 + (gamma/(alpha + gamma)))**-1
        i2_init = I_init - i1_init
        e2_init = i1_init*(alpha + gamma)/sigma
        e1_init = e2_init*(alpha + sigma)/sigma
        r_init = 1 - pi
        s_init = 1 - i1_init - i2_init - e1_init - e2_init - r_init
        if s_init <= 0:
            raise ValueError("Values of pi and i0 imply more than entire population")

        init_state = SEIR_state(
            s_init, e1_init, e2_init, i1_init, i2_init, r_init,
            *([0] * P_STATES)
        )
        assert self.check_state(init_state)
        return init_state, r0_init

    def _calculate_gradient(self, start_state, beta_t, dL, dI):
        # Change periods into rates per timestep
        sigma = 2./dL
        gamma = 2./dI

        S_to_E1 = beta_t * start_state.S * (start_state.I1 + start_state.I2)
        E1_to_E2 = sigma * start_state.E1
        E2_to_I1 = sigma * start_state.E2
        I1_to_I2 = gamma * start_state.I1
        I2_to_R = gamma * start_state.I2

        # Hack to make sure we keep susceptibles non-negative
        if S_to_E1 * self._dt > start_state.S:
            S_to_E1 = start_state.S / self._dt

        P_start = start_state[6:]
        P_gradient = (P_TRANSITIONS @ P_start)
        P_gradient[0] += S_to_E1

        gradient = SEIR_state(
            -S_to_E1,
            S_to_E1 - E1_to_E2,
            E1_to_E2 - E2_to_I1,
            E2_to_I1 - I1_to_I2,
            I1_to_I2 - I2_to_R,
            I2_to_R,
            *P_gradient
        )
        assert np.isclose(sum(gradient[:6]), 0)
        return gradient

    def _single_step(self, start_state, beta_t, dL, dI):
        gradient = self._calculate_gradient(start_state, beta_t, dL, dI)
        new_state = np.array(start_state) + np.array(gradient) * self._dt
        assert self.check_state(new_state)
        return SEIR_state(*new_state)

    def simulate(self, beta, dL, dI, psir, i0, pi):
        assert len(beta) == self._num_betas
        nni = np.zeros(int(self._num_steps * self._dt))
        prev = np.zeros(int(self._num_steps * self._dt))
        prev_state, r0_init = self._init_transmission(dL, dI, psir, i0, pi)
        cur_state = prev_state
        beta = np.array([1] + beta) * r0_init / dI
        for i in range(self._num_steps):
            day_num = int(i * self._dt)
            beta_t = beta[self._beta_of_step[i]]
            cur_state = self._single_step(cur_state, beta_t, dL, dI)
            nni[day_num] += prev_state.S - cur_state.S
            prev[day_num] += sum(cur_state[-(P_STATES-1):]) / 2
            prev_state = cur_state
        return nni, prev, cur_state
