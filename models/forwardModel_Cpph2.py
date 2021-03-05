import numpy as np
import scipy
import scipy.stats as stats
import scipy.signal as signal
from .CPP import seeiir_odeh2
class ForwardModelRegionWise(object):
    def __init__(self, last_states, dt, time, A, pop_size_N, mixing_matrix, 
                        breakpoints_list, dayindex, \
                        MaxConv_dead = None, dL = None):
        self._last_states = last_states
        self._dt = dt
        self._time = time
        self._T = len(time)
        self._N = pop_size_N
        self._A = A
        self._regions = 1
        if dL == None:
            self._dL = 3.0
        else:
            self._dL = dL
        if MaxConv_dead == None:
            self._MaxConv_dead = int(200)
        else:
            self._MaxConv_dead = MaxConv_dead

        self._pi = np.ones(self._A)     
        self._breakpoints_list = breakpoints_list
        self._dayindex = dayindex
    
        def conv_cdfs(dt, MaxConv_dead):
            mean = (4.0 + 15.0)
            sd = np.sqrt(1.41**2 + 12.1**2)
            alpha = mean**2/sd**2
            beta = mean/sd**2 
            delay_rv_dead = stats.gamma(a=alpha, scale=1/beta)
            cdfs_dead = np.zeros(MaxConv_dead)
            for n in range(MaxConv_dead-1):
                cdfs_dead[n] = (delay_rv_dead.cdf((n+1)*dt)-delay_rv_dead.cdf((n)*dt))
            cdfs_dead[MaxConv_dead-1] = 1-delay_rv_dead.cdf((MaxConv_dead-1)*dt)
            return cdfs_dead

        self.cdf_dead = conv_cdfs(self._dt, self._MaxConv_dead)
        self._mixing_matrix = mixing_matrix
        self._real_data = False

    def real_data(self):
        self._real_data = False  
           

    def transmission(self, dI, mix_mat, m_mul, beta):
        sigma = 2./self._dL
        gamma = 2./dI
        dt = self._dt
        dL = self._dL
        A = self._A
        region = self._regions
        T = self._T
        N = self._N    

        m_postlock = np.ones((A,A))
        m_postlock[:A-1,:]= np.exp(m_mul[0])
        m_postlock[A-1,:]= np.exp(m_mul[1])

        r0_init = self._last_states[6,0]
        r0star = self._last_states[7,0]
        
        s = np.zeros((T,A))
        r = np.zeros((T,A))
        lamda = np.zeros((T,A))
        i1 = np.zeros((T,A))
        i2 = np.zeros((T,A))
        e1 = np.zeros((T,A))
        e2 = np.zeros((T,A))
        
        #print(beta.dtype)
        
        s[0,:] = self._last_states[0,:]
        e1[0,:] = self._last_states[1,:]
        e2[0,:] = self._last_states[2,:]
        i1[0,:] = self._last_states[3,:]
        i2[0,:] = self._last_states[4,:]
        r[0,:] = self._last_states[5,:]
        Tlock = 200#dummy val
        beta = beta.reshape((-1,))
        seeiir_odeh2.SEEIIR_ODE(beta, lamda, i1, i2, e1, e2, r, s, \
            mix_mat, m_postlock, r0_init, r0star, \
                Tlock, gamma, sigma, dt, T, A)
        
        return s, lamda

    def convolution(self, S, lamda, p_dead):
        T = self._T
        A = self._A
        region = self._regions
        cdf_dead = self.cdf_dead
            
        MaxConv_dead = self._MaxConv_dead
        delta_infec = S*lamda
        shifted = np.roll(delta_infec,1,axis=0) ### Shift in time dimension 
        shifted[0,:] = shifted[1,:]     ### Copy \Delta (t=0) = \Delta(t=1)
        NNI = shifted
        delta = NNI
        con_len = len(cdf_dead)-1
        conv_series = np.zeros((T,A))

        seeiir_odeh2.convolution_2D(delta, conv_series, cdf_dead, con_len)

        p = np.zeros(8)
        p[0] = p_dead[0]
        p[1:] = p_dead
        for a in range(A):
            conv_series[:,a] = conv_series[:,a]*p[a]
     
        return conv_series, NNI

    def daily_betas(self, beta):
        dt = self._dt
        breakpoints_list = self._breakpoints_list 
        region = 1 
        dayindex = self._dayindex 

        grid_space = int(1/dt)    
        beta = np.hstack((self._last_states[8,0], beta))        
        betasplit = beta[:,None]
        
        beta_mat = np.ones((grid_space*7,region))*(beta[0])
        for i in range(3, betasplit.shape[0]+1):
            beta_mat = np.vstack((beta_mat,np.tile(np.sum(betasplit[:i,:],axis=0),(grid_space*7,1)))) 
        
        beta_mat = np.vstack((beta_mat,np.tile(np.sum(betasplit,axis=0),(((breakpoints_list[2]+1) - (breakpoints_list[1])),1)))) 
        assert len(dayindex) == beta_mat.shape[0]
        return beta_mat

    def simulate(self, _dI, _beta_rw, _p_dead, _m_mul, generate=False): 
        
        ### Solve the SEEIIR ODE using Euler regionwise
        _dI = _dI + 2.0
        daily_beta_rw = self.daily_betas(_beta_rw)
        daily_mix_matrix = self._mixing_matrix

        s, lamda= self.transmission(_dI, daily_mix_matrix, _m_mul, daily_beta_rw)            
        expected_deaths, nnis = self.convolution(s, lamda, _p_dead)
        expected_deaths = expected_deaths.reshape((s.shape[0],self._A))
        if generate:
            return nnis
        else:
            return expected_deaths, s

    def test(self):
        dt = self._dt 
        self._time 
        regions = 7
        dI = 0.472845081832237
        A = self._A 
        T = self._T
        N = self._N
        
        I0 = np.array([-16.63574, -13.90437, -15.33340, -17.67026, -15.51458, -15.29906, -16.09737])
        psir = np.array([0.2759202, 0.2319129, 0.2385402, 0.2977007, 0.2418800, 0.2210527, 0.2229998])
        p_dead = np.array([1.149820e-05, 2.444837e-05, 5.711427e-05, 3.656723e-04,
                        4.959340e-03, 3.428846e-02, 2.210078e-01])

        beta_rw_load = np.loadtxt('./model_data/log_beta_rw.txt')
        beta_rw_load = np.array(np.split(beta_rw_load,regions, axis=0)).T     
        daily_mix_matrix = self._mixing_matrix
        m_mul = np.array(np.split(np.loadtxt('./model_data/contact_parameters.txt'),regions,axis=0)).reshape((regions,3)).T  
        
        
        deaths_allage_r7 =  self.simulate(dI, psir[-1], beta_rw_load[:,-1], I0[-1], p_dead, m_mul[:,-1])[0] 
        S_allage_r7 =  self.simulate(dI, psir[-1], beta_rw_load[:,-1], I0[-1], p_dead, m_mul[:,-1])[1]    
        
        test_deaths = np.loadtxt('./model_data/deaths.txt')  
        test_S = np.loadtxt('./model_data/S.txt')
        
        np.testing.assert_allclose(S_allage_r7,test_S)
        #np.testing.assert_allclose(deaths_allage_r7,test_deaths)
        return deaths_allage_r7, test_deaths

