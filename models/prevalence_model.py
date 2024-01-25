from collections import namedtuple
import numpy as np
from scipy.linalg import block_diag

P_STATES = 4
KAPPA_Y = 0.082
KAPPA_X1 = 0.126
KAPPA_X2 = 0.0223
P_X = 0.00973
# Transition rates between P states
P_TRANSITIONS = np.array([
    -1/3,   0,           0,                 0,
    1/3,    -KAPPA_Y,    0,                 0,
    0,      KAPPA_Y,     -KAPPA_X1,         0,
    0,      0,           P_X * KAPPA_X1,    -KAPPA_X2,
]).reshape((P_STATES, P_STATES))
# Matrix to multiply current P to give new P
P_STEP = np.eye(P_STATES) + P_TRANSITIONS

SEIR_state = namedtuple(
    "SEIR_state",
    ["S", "E1", "E2", "I1", "I2", "R"] + ["P{}".format(i) for i in range(P_STATES)]
)

class BaseModel(object):
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
        self.max_time = max_time
        self._num_steps = int(np.round(max_time / dt))
        if self._num_steps > 10_000:
            raise ValueError("Too many steps required")
        if not np.isclose(self._num_steps, max_time / dt):
            raise ValueError("Max time must be divisible by step size")
        assert np.isclose(self._num_steps * self._dt, self.max_time)
        time_of_step = np.linspace(0, max_time - self._dt, self._num_steps)
        # Index of beta value for each timestep
        self._beta_of_step = [int(i) for i in time_of_step // beta_step_spacing]
        self.num_betas = self._beta_of_step[-1]
        if self._num_steps < self.num_betas + 1 or not (np.isclose(beta_step_spacing % dt, 0) or np.isclose(beta_step_spacing % dt, dt)):
            raise ValueError("beta_step_spacing should be a multiple of dt")
    
    def _check_inital_arg_values(self, dL, dI, psir, i0, pi):
        pi = np.array(pi)
        if (
                # If either period too short than finite differences is bad
                self._dt <= dL / 2 and
                self._dt <= dI / 2 and
                # Avoid overflows
                abs(psir) <= 10 and
                # Positive number of initial exposed and infecteds
                i0 > 0 and
                psir > 0 and
                # pi is a proportion
                (0 <= pi).all() and (pi <= 1).all()
            ):
            return
        raise ValueError("Impossible values for model")

    def _calc_r0(self, dL, dI, psir):        
        # See PNAS paper for explanation of parameterisation
        if np.isclose(psir, 0):
            r0_init = 1
        else:
            r0_init = psir*dI*((psir*dL/2 + 1)**2)/ (1- (1/(psir*dI/2 + 1)**2))

        return r0_init

class SingleRegionSingleAgeSEIRCompartmental(BaseModel):
    def check_state(self, state):
        state = np.array(state)
        return (
            np.isclose(sum(state[:6]), 1) and
            np.all(np.logical_or(state >= 0, np.isclose(state, 0)))
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
        self._check_inital_arg_values(dL, dI, psir, i0, pi)
        r0_init = self._calc_r0(dL, dI, psir)
        
        if dI * i0 > r0_init:
            raise ValueError("Hazard too high: would mean more than whole population infected")
        I_init = (dI * i0) / r0_init
        
        # Change periods into rates per day
        sigma = 2./dL
        gamma = 2./dI

        alpha = np.exp(psir*self._dt) - 1
        i1_init = I_init*(1 + (gamma/(alpha + gamma)))**-1
        i2_init = I_init - i1_init
        e2_init = i1_init*(alpha + gamma)/sigma
        e1_init = e2_init*(alpha + sigma)/sigma
        r_init = 1 - pi
        s_init = 1 - i1_init - i2_init - e1_init - e2_init - r_init
        if s_init <= 0:
            raise ValueError("Values of pi and i0 imply more than entire population")

        init_state = SEIR_state(
            float(s_init), float(e1_init), float(e2_init),
            float(i1_init), float(i2_init), float(r_init),
            *([0] * P_STATES)
        )
        assert self.check_state(init_state)
        return init_state, r0_init

    def _calculate_gradient(self, start_state, beta_t, dL, dI):
        # Change periods into rates
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
        return gradient

    def _single_step(self, start_state, beta_t, dL, dI):
        gradient = self._calculate_gradient(start_state, beta_t, dL, dI)
        new_state = np.array(start_state) + np.array(gradient) * self._dt
        return SEIR_state(*new_state)

    def simulate(self, beta, dL, dI, psir, i0, pi):
        assert len(beta) == self.num_betas
        nni = np.zeros(int(self._num_steps * self._dt))
        prev = np.zeros(int(self._num_steps * self._dt))
        prev_state, r0_init = self._init_transmission(dL, dI, psir, i0, pi)
        cur_state = prev_state
        beta = np.insert(beta, 0, 1) * r0_init / dI
        prev_day = -1
        steps_since_day_change = 0
        for i in range(self._num_steps):
            day_num = int(i * self._dt)
            if day_num == prev_day:
                steps_since_day_change += 1
            else:
                steps_since_day_change = 1
                prev_day = day_num
            beta_t = beta[self._beta_of_step[i]]
            cur_state = self._single_step(cur_state, beta_t, dL, dI)
            nni[day_num] += prev_state.S - cur_state.S
            prev[day_num] = (sum(cur_state[-(P_STATES-1):]) + prev[day_num] * (steps_since_day_change - 1)) / steps_since_day_change
            prev_state = cur_state
        return nni, prev, cur_state




class SingleRegionStratifiedSEIRCompartmental(BaseModel):
    STATE_NAMES = tuple(["S", "E1", "E2", "I1", "I2", "R"] + ["P{}".format(i) for i in range(7)])
    STATE_POS = {name: i for i, name in enumerate(STATE_NAMES)}
    P_STATES = slice(7, 6+7)

    """Create a model, with configuration that doesn't change.

    Params:
        matrices: numpy array of (steps, num_strata, num_strata) giving contact matrices at each step
    """
    def __init__(self, matrices, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(matrices.shape) == 3
        assert matrices.shape[1] == matrices.shape[2]
        assert matrices.shape[0] == self.num_betas + 1
        self.num_strata = matrices.shape[2]
        self._base_matrices = matrices

    def check_state(self, state):
        return (
            np.allclose(state[:6,].sum(axis=0), 1) and
            np.all(np.logical_or(np.array(state) >= 0, np.isclose(state, 0)))
        )

    def get_compartment(self, state, compartment):
        return state[self.STATE_POS[compartment], :]

    def get_stratum(self, state, stratum):
        return state[:, stratum]

    """Returns the dominant eigenvalue and corresponding eigenvector scaled so largest element is 1"""
    def _dominant_eigen(self, matrix):
        eigenvals, eigenvecs = np.linalg.eig(matrix)
        dominant_eigenval_pos = np.argmax(abs(eigenvals))
        dominant_eigenval = abs(eigenvals[dominant_eigenval_pos])
        assert np.isscalar(dominant_eigenval)
        dominant_eigenvec = eigenvecs[:, dominant_eigenval_pos]
        if not np.isrealobj(dominant_eigenvec):
            assert np.isreal(dominant_eigenvec).all()
            dominant_eigenvec = np.real(dominant_eigenvec)
        if (dominant_eigenvec < 0).any():
            dominant_eigenvec *= -1
        dominant_eigenvec /= max(dominant_eigenvec)
        assert (dominant_eigenvec <= 1).all()
        return dominant_eigenval, dominant_eigenvec


    """Calculate the system state at time 0.
    
    Params:
        dL: length of the latent period
        dI: length of the infectious period
        psir: initial rate of exponential growth?
        i0: proportion iniitally infected in the strata with highest initial proportion infected
        pi: vector (one per strata) of proportion of population with prior immunity
    """
    def _init_transmission(self, dL, dI, psir, i0, pi, first_matrix):
        assert (self.num_strata == 1 and np.isscalar(pi)) or len(pi) == self.num_strata
        self._check_inital_arg_values(dL, dI, psir, i0, pi)
        
        # Equations S2-S5 of PNAS paper
        r0_init = self._calc_r0(dL, dI, psir)
        r0_star, inf_dist = self._dominant_eigen(first_matrix)
        assert (inf_dist >= 0).all()
        assert r0_star > 0
        matrix_multiplier = r0_init / r0_star
        I_init = i0 * np.broadcast_to(inf_dist, (self.num_strata,))

        # Change periods into rates per day (not per step as elsewhere)
        sigma = 2./dL
        gamma = 2./dI

        i1_init = I_init*(1 + (gamma/(psir + gamma)))**-1
        i2_init = I_init - i1_init
        e2_init = i1_init*(psir + gamma)/sigma
        e1_init = e2_init*(psir + sigma)/sigma
        r_init = 1 - pi
        s_init = 1 - i1_init - i2_init - e1_init - e2_init - r_init
        if np.logical_and(s_init < 0, np.logical_not(np.isclose(s_init, 0))).any():
            raise ValueError("Values of pi and i0 imply more than entire population")
        p_init = self._calculate_initial_prev_state(e1_init, sigma, psir)

        init_state = np.vstack((s_init, e1_init, e2_init, i1_init, i2_init, r_init, p_init))
        assert self.check_state(init_state)
        return init_state, matrix_multiplier

    def _calculate_initial_prev_state(self, e1_init, sigma, psir):
        p_init = np.empty((P_STATES, self.num_strata))
        p_init[0] = (sigma + psir) / (psir + -P_TRANSITIONS[1, 1]) * e1_init

        P_OUT = -np.diag(P_TRANSITIONS)
        P_IN = P_TRANSITIONS @ np.ones(P_STATES) + P_OUT
        for i in range(1, P_STATES):
            p_init[i] = (P_IN[i] / (psir + P_OUT[i])) * p_init[i - 1]
        return p_init

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
        return gradient

    def _calc_linear_transition_matrix(self, dL, dI):
        # Change periods into rates per timestep
        sigma = 2./dL * self._dt
        gamma = 2./dI * self._dt

        # Leaving states
        transition_rates = np.array((0, sigma, sigma, gamma, gamma, 0))
        linear_transitions = np.diag(1 - transition_rates)
        # Entering states
        states_entering = np.arange(1, linear_transitions.shape[0])
        linear_transitions[states_entering, states_entering-1] = transition_rates[:-1]
        return block_diag(linear_transitions, np.eye(P_STATES) + P_TRANSITIONS * self._dt)


    def _single_step(self, start_state, B_t, linear_transitions):
        new_state = linear_transitions @ start_state
        currently_infected = self.get_compartment(start_state, "I1") + self.get_compartment(start_state, "I2")
        new_infections = self._dt * self.get_compartment(start_state, "S") * (B_t @ currently_infected)
        new_infections = np.minimum(self.get_compartment(start_state, "S"), new_infections)
        new_state[self.STATE_POS["S"], :] -= new_infections
        new_state[self.STATE_POS["E1"], :] += new_infections
        new_state[self.STATE_POS["P0"], :] += new_infections

        return new_state

    """Simulate a whole epidemic.

    Params:
        beta: a vector of length _num_betas with a scalar that multiplies the matrix in each step
        dL: mean length of the latent period
        dI: mean length of the infectious period
        psir: initial growth rate
        i0: proportion iniitally infected in the strata with highest initial proportion infected
        pi: the proportion of the population without prior immunity
        matrix_modifiers: a matrix which is element-wise multiplied by the base contact matrix.
            May encode strata-specific infectiousness or susceptibility.

    Returns:
        nni: a (max_time, num_strata) array of number of new infections per day
        prevalence: a (max_time, num_strata) array of number of prevalence on each day
        final_state: the state at the end of the epidemic
    """
    def simulate(self, beta, dL, dI, psir, i0, pi, matrix_modifiers, include_new_pcr_pos=False):
        assert len(matrix_modifiers.shape) == 2
        assert matrix_modifiers.shape[0] == matrix_modifiers.shape[1] == self.num_strata
        assert len(beta) == self.num_betas

        beta = np.insert(beta, 0, 1) # First beta is always 1
        matrices =  self._base_matrices * matrix_modifiers

        nni = np.zeros((self.max_time, self.num_strata))
        prev = np.zeros((self.max_time, self.num_strata))
        new_pcr_pos = np.zeros((self.max_time, self.num_strata))
        prev_state, matrix_scaling = self._init_transmission(dL, dI, psir, i0, pi, matrices[0])
        linear_transitions = self._calc_linear_transition_matrix(dL, dI)
        cur_state = prev_state
        matrices *= matrix_scaling / dI * beta.reshape(self._base_matrices.shape[0], 1, 1)
        prev_day = -1
        steps_since_day_change = 0
        for i in range(self._num_steps):
            day_num = int(i * self._dt)
            if day_num == prev_day:
                steps_since_day_change += 1
            else:
                steps_since_day_change = 1
                prev_day = day_num
            B_t = matrices[self._beta_of_step[i]]
            cur_state = self._single_step(cur_state, B_t, linear_transitions)
            nni_in_timestep = self.get_compartment(prev_state, "S") - self.get_compartment(cur_state, "S")
            nni[day_num] += nni_in_timestep
            # New PCR positive = number leaving P0 = enterance to P0 - net change in P0 = nni - net change in P0
            net_change_P0_compartment = self.get_compartment(cur_state, "P0") - self.get_compartment(prev_state, "P0")
            new_pcr_pos[day_num] += nni_in_timestep - net_change_P0_compartment
            # Build up average day-by-day without assuming number of steps per day
            prev[day_num] = (sum(cur_state[-(P_STATES-1):]) + prev[day_num] * (steps_since_day_change - 1)) / steps_since_day_change
            prev_state = cur_state
        if include_new_pcr_pos:
            return nni, prev, cur_state, new_pcr_pos
        return nni, prev, cur_state
