from collections import namedtuple
import numpy as np

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
        self._dt = dt
        self._max_time = max_time
        self._num_steps = int(np.ceil(max_time / dt))
        assert self._num_steps * self._dt == self._max_time # Check for floating point issues
        time_of_step = np.arange(0, max_time + dt, dt)
        # Index of beta value for each timestep
        self._beta_of_step = [int(i) for i in time_of_step // beta_step_spacing]
        self._num_betas = self._beta_of_step[-1]

    def check_state(self, state):
        return (
            np.isclose(sum(state[:6]), 1) and
            np.all(np.array(state) >= 0)
        )

    """Calculate the system state at time 0.
    
    Params:
        dL: length of the latent period
        dI: length of the infectious period
        psir: initial rate of exponential growth?
        I0: number of initially infected individuals?
        pi: proportion of the population initially susceptible
    """
    def _init_transmission(self, dL, dI, psir, I0, pi):
        # Change periods into rates
        sigma = 2./dL * self._dt
        gamma = 2./dI * self._dt
        
        # See PNAS paper for explanation of parameterisation
        r0_init = psir*dI*((psir*dL/2 + 1)**2)/ (1- (1/(psir*dI/2 + 1)**2))
        alpha = np.exp(psir*self._dt) - 1
        I_init = (dI * np.exp(I0))/r0_init

        i1_init = I_init*(1 + (gamma/(alpha + gamma)))**-1
        i2_init = I_init - i1_init
        e2_init = i1_init*(alpha + gamma)/sigma
        e1_init = e2_init*(alpha + sigma)/sigma
        r_init = 1 - pi
        s_init = 1 - i1_init - i2_init - e1_init - e2_init - r_init

        init_state = SEIR_state(
            s_init, e1_init, e2_init, i1_init, i2_init, r_init,
            *([0] * P_STATES)
        )
        assert self.check_state(init_state)
        return init_state, r0_init

    def _calculate_gradient(self, start_state, beta_t, dL, dI):
        # Change periods into rates
        sigma = 2./dL * self._dt
        gamma = 2./dI * self._dt

        S_to_E1 = beta_t * start_state.S * (start_state.I1 + start_state.I2) 
        E1_to_E2 = sigma * start_state.E1
        E2_to_I1 = sigma * start_state.E2
        I1_to_I2 = gamma * start_state.I1
        I2_to_R = gamma * start_state.I2

        P_start = start_state[6:]
        P_gradient = (P_TRANSITIONS @ P_start) * self._dt
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

    def simulate(self, beta, dL, dI, psir, I0, pi):
        assert len(beta) == self._num_betas
        nni = [0] * int(self._num_steps * self._dt)
        prev = [0] * int(self._num_steps * self._dt)
        prev_state, r0_init = self._init_transmission(dL, dI, psir, I0, pi)
        cur_state = prev_state
        beta = np.array([1] + beta) * r0_init / dI
        for i in range(self._num_steps):
            day_num = int(i * self._dt)
            beta_t = beta[self._beta_of_step[i]]
            cur_state = self._single_step(cur_state, beta_t, dL, dI)
            nni[day_num] += prev_state.S - cur_state.S
            prev[day_num] += sum(cur_state[-(P_STATES-1):]) / 2
            prev_state = cur_state
        return nni, prev


p = SingleRegionSingleAgeSEIRCompartmental(100)
t0, _ = p._init_transmission(2, 2, 1, np.log(0.0001), 0.5)
t1 = p._single_step(t0, 1, 1, 1)
out = p.simulate([1] * p._num_betas, 1, 1, 1, np.log(0.0001), 0.5)
print("a")