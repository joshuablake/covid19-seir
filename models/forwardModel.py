import numpy as np
import scipy
import scipy.stats as stats
import scipy.signal as signal

class ForwardModel(object):
    def __init__(self, dt, time, pop_size_N, mixing_matrix, \
                        seventh_breakpoint, lockdn_breakpoint, 
                        breakpoints_list, dayindex, \
                        MaxConv_dead = None, dL = None):
        self._dt = dt
        self._time = time
        self._T = len(time)
        self._N = pop_size_N
        self._A = pop_size_N.shape[0] 
        self._regions = pop_size_N.shape[1]
        if dL == None:
            self._dL = 3.0
        else:
            self._dL = dL
        if MaxConv_dead == None:
            self._MaxConv_dead = int(200)
        else:
            self._MaxConv_dead = int(MaxConv)

        self._pi = np.ones((self._A,self._regions))
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
        p = np.ones(region)*0.1
        I_init = (dI * np.exp(I0) * N.sum(axis=0))/(p*r0_init) # later parameterise p
        eiglist=[np.linalg.eig(ngen_M0[:,:,r]) for r in range(region)]
        dom_eigvec = np.array([eiglist[r][1][:,np.argsort(abs(eiglist[r][0]))[-1]] for r in range(region)])
        i_tot_a = I_init*(dom_eigvec.T/dom_eigvec.sum(axis=1)) ## Normalise the eigenvector? and scale with (\sum_age I0_age)?

        i1_init = np.zeros((A,region))
        i2_init = np.zeros((A,region))
        e1_init = np.zeros((A,region))
        e2_init = np.zeros((A,region))
        r_init = np.zeros((A,region))
        s_init = np.zeros((A,region))
        i1_init = i_tot_a*(1 + (gamma*dt/(alpha + gamma*dt)))**-1
        i2_init = i_tot_a - i1_init
        e2_init = i1_init*(alpha + gamma*dt)/(sigma*dt)
        e1_init = e2_init*(alpha + sigma*dt)/(sigma*dt)
        r_init = (1 - pi)*N
        s_init = pi*N
        
        return i1_init, i2_init, e1_init, e2_init, r_init, s_init, r0_init   

    def hazard_func(self, beta, mat, I1, I2, r0s, r_init, gamma):
        A = self._A
        region = self._regions
        dt = self._dt
        prod = np.ones((A,A,region))
        b_mat = (np.exp(beta)* (r_init/r0s))*mat
        prod = (1 - (b_mat))**(I1 + I2)
        pr = np.cumprod(prod,axis=1)[:,-1,:]
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
    

        m_prelock, m_postlock = np.ones((A,A,region)), np.ones((A,A,region))
        m_prelock[A-1,:,:]=np.exp(m_mul[0])
        m_postlock[:A-1,:,:]= np.exp(m_mul[1])
        m_postlock[A-1,:,:]= np.exp(m_mul[2])
        
        init_mix_mat = mix_mat[:,:,0,:]*m_prelock    
        nxtgen_mix_mat = np.array([init_mix_mat[rw,:,:]*N[rw]*dI for rw in range(A)])
        r0star = np.array([np.sort(abs(np.linalg.eigvals(nxtgen_mix_mat[:,:,r])))[-1] for r in range(region)])

        s = np.zeros((T,A,region))
        r = np.zeros((T,A,region))
        lamda = np.zeros((T,A,region))
        i1 = np.zeros((T,A,region))
        i2 = np.zeros((T,A,region))
        e1 = np.zeros((T,A,region))
        e2 = np.zeros((T,A,region))
        
        i1[0,:,:], i2[0,:,:], e1[0,:,:], e2[0,:,:], r[0,:,:], s[0,:,:], r0_init \
        = self.init_transmission(dI, psir, nxtgen_mix_mat, I0)
        Tlock = tlock
        
        for i in range(1,T):                
            if i < Tlock+1:
                lamda[i-1,:,:] = self.hazard_func(beta[i-1,:], \
                                            mix_mat[:,:,i-1,:]*m_prelock, \
                                            i1[i-1,:,:], i2[i-1,:,:], r0star, \
                                            r0_init, gamma)
                
                s[i,:,:] = s[i-1,:,:]*(1 - lamda[i-1,:,:])
                e1[i,:,:] = e1[i-1,:,:]*(1- sigma*dt) + s[i-1,:,:]*lamda[i-1,:,:]
            else:
                lamda[i-1,:,:] = self.hazard_func(beta[i-1,:], \
                                            mix_mat[:,:,i-1,:]*m_postlock, \
                                            i1[i-1,:,:], i2[i-1,:,:], r0star, \
                                            r0_init, gamma)
                s[i,:,:] = s[i-1,:,:]*(1 - lamda[i-1,:,:])
                e1[i,:,:] = e1[i-1,:,:]*(1- sigma*dt) + s[i-1,:,:]*lamda[i-1,:,:]
        
            e2[i,:,:] = e2[i-1,:,:]*(1- sigma*dt) + e1[i-1,:,:]*sigma*dt
            i1[i,:,:] = i1[i-1,:,:]*(1- gamma*dt) + e2[i-1,:,:]*sigma*dt
            i2[i,:,:] = i2[i-1,:,:]*(1- gamma*dt) + i1[i-1,:,:]*gamma*dt
            r[i,:,:] =  r[i-1,:,:] + i2[i-1,:,:]*gamma*dt
        
        return s, lamda

    def convolution(self, S, lamda, p_dead):
        T = self._T
        A = self._A
        region = self._regions
        cdf_dead = self.cdf_dead
            
        MaxConv_dead = self._MaxConv_dead
        delta_infec = S*lamda
        shifted = np.roll(delta_infec,1,axis=0) ### Shift in time dimension 
        shifted[0,:,:] = shifted[1,:,:]     ### Copy \Delta (t=0) = \Delta(t=1)
        NNI = shifted
        delta = NNI
        con_len = len(cdf_dead)-1
        conv_series = np.zeros((T,A,region))
        conv_series_kids_summed = np.zeros((T,A-1,region))

        cdf_dead = np.tile(cdf_dead.reshape((-1,1,1)),(1,A,region))
        conv_series = signal.convolve(delta, cdf_dead, mode='full', method='fft')[:334,-8:,:7]
        conv_series_kids_summed[:,0,:] = conv_series[:,0,:] + conv_series[:,1,:]
        conv_series_kids_summed[:,1:,:] = conv_series[:,2:,:] 
        conv_series_kids_summed = conv_series_kids_summed*np.tile(p_dead.reshape((1,-1,1)),(T,1,7))
        
        return conv_series_kids_summed

    def daily_betas(self, beta):
        dt = self._dt
        seventh_breakpoint = self._seventh_breakpoint 
        breakpoints_list = self._breakpoints_list 
        region = self._regions 
        dayindex = self._dayindex 

        grid_space = int(1/dt)            
        betasplit = np.array(np.split(beta,region, axis=0)).T
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
        if not test:
            m_mul = np.array(np.split(_m_mul,self._regions,axis=0)).reshape((self._regions,3)).T
        else:
            m_mul = _m_mul
        s, lamda= self.transmission(_dI, _psir, daily_mix_matrix, m_mul, 
                                _I0, daily_beta_rw, self._lockdn_breakpoint)            
        expected_deaths = self.convolution(s, lamda, _p_dead).reshape((s.shape[0],self._A - 1,self._regions))        
        return expected_deaths, s

    def test(self):
        dt = self._dt 
        self._time 
        regions = self._regions
        dI = 0.472845081832237
        A = self._A 
        T = self._T
        N = self._N
        
        I0 = np.array([-16.63574, -13.90437, -15.33340, -17.67026, -15.51458, -15.29906, -16.09737])
        psir = np.array([0.2759202, 0.2319129, 0.2385402, 0.2977007, 0.2418800, 0.2210527, 0.2229998])
        p_dead = np.array([1.149820e-05, 2.444837e-05, 5.711427e-05, 3.656723e-04,
                        4.959340e-03, 3.428846e-02, 2.210078e-01])

        beta_rw_load = np.loadtxt('./model_data/log_beta_rw.txt')            
        daily_mix_matrix = self._mixing_matrix
        m_mul = np.array(np.split(np.loadtxt('./model_data/contact_parameters.txt'),regions,axis=0)).reshape((regions,3)).T  
        deaths =  self.simulate(dI, psir, beta_rw_load, I0, p_dead, m_mul, True)[0]    
        test_deaths = np.loadtxt('./model_data/deaths.txt')  
        deaths_allage_r7 = deaths[:,:,-1]
        np.testing.assert_allclose(deaths_allage_r7,test_deaths)
