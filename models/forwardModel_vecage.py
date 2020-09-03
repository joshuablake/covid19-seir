import numpy as np
import scipy
import scipy.stats as stats
import scipy.signal as signal

class ForwardModelRegionWise(object):
    def __init__(self, dt, time, A, pop_size_N, mixing_matrix, \
                        seventh_breakpoint, lockdn_breakpoint, 
                        breakpoints_list, dayindex, \
                        MaxConv_dead = None, dL = None):
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
            self._MaxConv_dead = int(MaxConv)

        self._pi = np.ones(self._A)
        self._seventh_breakpoint = seventh_breakpoint        
        self._lockdn_breakpoint = lockdn_breakpoint
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
            cdfs_dead[MaxConv_dead-1] = 1-delay_rv_dead.cdf(MaxConv_dead-1*dt)
            return cdfs_dead

        self.cdf_dead = conv_cdfs(self._dt, self._MaxConv_dead)
        self._mixing_matrix = mixing_matrix

    def init_transmission(self, dI, psir, ngen_M0, I0):
        dt = self._dt
        A = self._A
        N = self._N
        dL = self._dL
        region = self._regions
        pi = self._pi
        
        sigma = 2./dL
        gamma = 2./dI
        
        r0_init = psir*dI*((psir*dL/2 + 1)**2)/ (1- (1/(psir*dI/2 + 1)**2))
        alpha = np.exp(psir*dt) - 1
        p = 0.1
        I_init = (dI * np.exp(I0) * N.sum())/(p*r0_init) # later parameterise p
        eig, eigvec =  np.linalg.eig(ngen_M0)
        dom_eigvec = eigvec[:,np.argsort(abs(eig))[-1]] ## Find dominant eigenvector of nextgen matrix
        i_tot_a = I_init*(dom_eigvec/dom_eigvec.sum()) ## Normalise the eigenvector? and scale with (\sum_age I0_age)?
        

        i1_init = np.zeros(A)
        i2_init = np.zeros(A)
        e1_init = np.zeros(A)
        e2_init = np.zeros(A)
        r_init = np.zeros(A)
        s_init = np.zeros(A)
        i1_init = i_tot_a*(1 + (gamma*dt/(alpha + gamma*dt)))**-1
        i2_init = i_tot_a - i1_init
        e2_init = i1_init*(alpha + gamma*dt)/(sigma*dt)
        e1_init = e2_init*(alpha + sigma*dt)/(sigma*dt)
        r_init = (1 - pi)*N
        s_init = pi*N
        
        return i1_init, i2_init, e1_init, e2_init, r_init, s_init, r0_init   
    
    def hazard_func(self, beta, mat, I1, I2, r0s, r_init, gamma):
        A = self._A
        dt = self._dt
        prod = np.ones((A,A))
        b_mat = (np.exp(beta)* (r_init/r0s))*mat
        prod = (1 - (b_mat))**(I1 + I2)
        pr = np.cumprod(prod,axis=1)[:,-1]
        return (1 - pr)*dt


    def transmission(self, dI, psir, mix_mat, m_mul, I0, beta, tlock):
        sigma = 2./self._dL
        gamma = 2./dI
        dt = self._dt
        dL = self._dL
        A = self._A
        region = self._regions
        T = self._T
        N = self._N
        pi = self._pi
    

        m_prelock, m_postlock = np.ones((A,A)), np.ones((A,A))
        m_prelock[A-1,:]=np.exp(m_mul[0])
        

        m_postlock[:A-1,:]= np.exp(m_mul[1])
        m_postlock[A-1,:]= np.exp(m_mul[2])
        
        init_mix_mat = mix_mat[:,:,0]*m_prelock    
        nxtgen_mix_mat = np.array([init_mix_mat[rw,:]*N[rw]*dI for rw in range(A)])
        
        r0star = np.sort(abs(np.linalg.eigvals(nxtgen_mix_mat)))[-1]
        
        
        
        s = np.zeros((T,A))
        r = np.zeros((T,A))
        lamda = np.zeros((T,A))
        i1 = np.zeros((T,A))
        i2 = np.zeros((T,A))
        e1 = np.zeros((T,A))
        e2 = np.zeros((T,A))
        
        
        
        i1[0,:], i2[0,:], e1[0,:], e2[0,:], r[0,:], s[0,:], r0_init \
        = self.init_transmission(dI, psir, nxtgen_mix_mat, I0)
        Tlock = tlock

        for i in range(1,T):                
            if i < Tlock+1:
                lamda[i-1,:] = self.hazard_func(beta[i-1], \
                                            mix_mat[:,:,i-1]*m_prelock, \
                                            i1[i-1,:], i2[i-1,:], r0star, \
                                            r0_init, gamma)
                
                s[i,:] = s[i-1,:]*(1 - lamda[i-1,:])
                e1[i,:] = e1[i-1,:]*(1- sigma*dt) + s[i-1,:]*lamda[i-1,:]
            else:
                lamda[i-1,:] = self.hazard_func(beta[i-1], \
                                            mix_mat[:,:,i-1]*m_postlock, \
                                            i1[i-1,:], i2[i-1,:], r0star, \
                                            r0_init, gamma)
                s[i,:] = s[i-1,:]*(1 - lamda[i-1,:])
                e1[i,:] = e1[i-1,:]*(1- sigma*dt) + s[i-1,:]*lamda[i-1,:]
        
            e2[i,:] = e2[i-1,:]*(1- sigma*dt) + e1[i-1,:]*sigma*dt
            i1[i,:] = i1[i-1,:]*(1- gamma*dt) + e2[i-1,:]*sigma*dt
            i2[i,:] = i2[i-1,:]*(1- gamma*dt) + i1[i-1,:]*gamma*dt
            r[i,:] =  r[i-1,:] + i2[i-1,:]*gamma*dt  
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
        conv_series_kids_summed = np.zeros((T,A-1))

        cdf_dead = np.tile(cdf_dead.reshape((-1,1)),(1,A))
        conv_series = signal.convolve(delta, cdf_dead, mode='full', method='fft')[:334,-8:]
        conv_series_kids_summed[:,0] = conv_series[:,0] + conv_series[:,1]
        conv_series_kids_summed[:,1:] = conv_series[:,2:]  
        for a in range(A-1):
            conv_series_kids_summed[:,a] = conv_series_kids_summed[:,a]*p_dead[a]
        
        return conv_series_kids_summed

    def daily_betas(self, beta):
        dt = self._dt
        seventh_breakpoint = self._seventh_breakpoint 
        breakpoints_list = self._breakpoints_list 
        region = 1 
        dayindex = self._dayindex 

        grid_space = int(1/dt)            
        betasplit = beta[:,None]
        beta_mat = np.zeros((seventh_breakpoint,region))
        for i in range(2, betasplit.shape[0]+1):
            beta_mat = np.vstack((beta_mat,np.tile(np.sum(betasplit[:i,:],axis=0),(grid_space*7,1)))) 
        
        beta_mat = np.vstack((beta_mat,np.tile(np.sum(betasplit,axis=0),(((breakpoints_list[2]+1) - (breakpoints_list[1])),1)))) 
        assert len(dayindex) == beta_mat.shape[0]
        return beta_mat

    def simulate(self, _dI, _psir, _beta_rw, _I0, _p_dead, _m_mul, test=False): 
        
        ### Solve the SEEIIR ODE using Euler regionwise
        _dI = _dI + 2.0
        daily_beta_rw = self.daily_betas(_beta_rw)
        daily_mix_matrix = self._mixing_matrix

        s, lamda= self.transmission(_dI, _psir, daily_mix_matrix, _m_mul, 
                                _I0, daily_beta_rw, self._lockdn_breakpoint)            
        expected_deaths = self.convolution(s, lamda, _p_dead).reshape((s.shape[0],self._A - 1))        
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
        deaths_allage_r7 =  self.simulate(dI, psir[-1], beta_rw_load[:,-1], I0[-1], p_dead, m_mul[:,-1], True)[0] 
        S_allage_r7 =  self.simulate(dI, psir[-1], beta_rw_load[:,-1], I0[-1], p_dead, m_mul[:,-1], True)[1]    
        test_deaths = np.loadtxt('./model_data/deaths.txt')  
        test_S = np.loadtxt('./model_data/S.txt')
        
        np.testing.assert_allclose(S_allage_r7,test_S)
        np.testing.assert_allclose(deaths_allage_r7,test_deaths)
"""

"""
