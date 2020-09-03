import numpy as np
import scipy
import scipy.stats as stats
#from cython_convolution import convolution_in_C

class ForwardModel(object):

    def __init__(self, dt, time, pop_size_N, polymod_file_path, MaxConv = None, dL = None,  breakpoints_gp = None, 
                                                                        pi = None, delay_means = None, delay_sds =None):

        self._dt = dt
        self._time = time
        self._T = len(time)
        self._A = len(pop_size_N)
        self._N = pop_size_N
        if dL == None:
            self._dL = 2.0
        else:
            self._dL = dL
        if MaxConv == None:
            self.MaxConv_cc = int(67)
            self.MaxConv_gp = int(67)
        else:
            self.MaxConv_cc = int(MaxConv[0])
            self.MaxConv_gp = int(MaxConv[1])
        if breakpoints_gp == None:
            self._breakpoints_gp = [83, 130, 178]
        else:
            self._breakpoints_gp = breakpoints_gp
        if pi == None:
            self._pi = [1, 0.98, 0.969, 0.845, 0.92, 0.865, 0.762]
        else:
            self._pi = pi

        if delay_means == None:
            self._delay_means = [1.62 + 1.968 + 0.5, 1.62 + 6.577]
        else:
            self._delay_means = delay_means
        if delay_sds == None:
            self._delay_sds = [1.791**2 + 1.199**2 + 0.5**2, 1.791**2 + 3.722**2]
        else:
            self._delay_sds = delay_sds
        self.m_poly = np.loadtxt(polymod_file_path, dtype=np.float)

    def init_transmission(self, dI, psir, nu, M, pi, pgp):
        A = self._A
        dt = self._dt
        N = self._N
        sigma = 2./self._dL
        gamma = 2./dI
        
        r0_init = psir*dI*((psir*self._dL/2 + 1)**2)/ (1- (1/(psir*dI/2 + 1)**2))
        alpha = np.exp(psir*dt) - 1
        
        i_tot = N.sum()
        i_tot *= dI*np.exp(nu)/(pgp*r0_init) ## Equation S5
        eig, eigvec =  np.linalg.eig(M)
        dom_eigvec = eigvec[:,np.argsort(abs(eig))[-1]] ## Find dominant eigenvector
        i_tot_a = i_tot*(dom_eigvec/dom_eigvec.sum()) ## Equation S6
        

        i1_init = np.zeros(A)
        i2_init = np.zeros(A)
        e1_init = np.zeros(A)
        e2_init = np.zeros(A)
        r_init = np.zeros(A)
        s_init = np.zeros(A)
        
        for a in range(A):
            i1_init[a] = i_tot_a[a]*(1 + (gamma*dt/(alpha + gamma*dt)))**-1
            i2_init[a] = i_tot_a[a] - i1_init[a]
            e2_init[a] = i1_init[a]*(alpha + gamma*dt)/(sigma*dt)
            e1_init[a] = e2_init[a]*(alpha + sigma*dt)/(sigma*dt)
            r_init[a] = (1 - pi[a])*N[a]
            s_init[a] = pi[a]*N[a]

        return i1_init, i2_init, e1_init, e2_init, r_init, s_init, r0_init   

    def hazard_func(self, mat, I1, I2, r0, r_init, gamma, a, i):
        dt = self._dt
        A = self._A
        prod = 1.
        scaled_mat = mat/ r0
        lam_in = []
        for b in range(A):
            prod *= (1 - (scaled_mat[a,b]*r_init*gamma/2.))**(I1[b] + I2[b])
            lam_in.append(scaled_mat[a,b])        
            
        return (1 - prod)*dt

    def transmission(self, dI, psir, nu, pi, mix_mat, pgp):
        dt = self._dt
        T = self._T
        A = self._A 
        sigma = 2./self._dL
        gamma = 2./dI
        N = self._N
        init_mix_mat = mix_mat[:,:,0]
        
        nxtgen_mix_mat = np.array([init_mix_mat[rw,:]*N[rw] for rw in range(A)])
        r0star = np.sort(abs(np.linalg.eigvals(nxtgen_mix_mat)))[-1]
        
        s = np.zeros((T,A))
        r = np.zeros((T,A))
        lamda = np.zeros((T,A))
        i1 = np.zeros((T,A))
        i2 = np.zeros((T,A))
        e1 = np.zeros((T,A))
        e2 = np.zeros((T,A))
        
        i1[0,:], i2[0,:], e1[0,:], e2[0,:], r[0,:], s[0,:], r0_init \
        = self.init_transmission(dI, psir, nu, nxtgen_mix_mat, pi, pgp)

        for i in range(T):
            for a in range(A):
                lamda[i,a] = self.hazard_func(mix_mat[:,:,i], i1[i,:], i2[i,:], r0star, r0_init, gamma, a, i)
                
                if i<T-1:
                    
                    s[i+1,a] = s[i,a]*(1 - lamda[i,a])
                    e1[i+1,a] = e1[i,a]*(1- sigma*dt) + s[i,a]*lamda[i,a]
                    e2[i+1,a] = e2[i,a]*(1- sigma*dt) + e1[i,a]*sigma*dt
                    i1[i+1,a] = i1[i,a]*(1- gamma*dt) + e2[i,a]*sigma*dt
                    i2[i+1,a] = i2[i,a]*(1- gamma*dt) + i1[i,a]*gamma*dt
                    r[i+1,a] =  r[i,a] + i2[i,a]*gamma*dt

        return s, lamda

    def conv_cdfs(self, means, sds):
        dt = self._dt
        MaxConv_cc = self.MaxConv_cc
        MaxConv_gp = self.MaxConv_gp
        
        shape_gp, shape_cc = means[0]**2/sds[0], means[1]**2/sds[1]
        scale_gp, scale_cc = sds[0]/means[0], sds[1]/means[1]
        delay_rv_gp = stats.gamma(a=shape_gp, scale=scale_gp)
        delay_rv_cc = stats.gamma(a=shape_cc, scale=scale_cc)
        cdfs_gp, cdfs_cc = np.zeros(MaxConv_gp), np.zeros(MaxConv_cc)
        for n in range(MaxConv_gp-1):
            cdfs_gp[n] = (delay_rv_gp.cdf((n+1)*dt)-delay_rv_gp.cdf((n)*dt))
        for m in range(MaxConv_cc-1):
            cdfs_cc[m] = (delay_rv_cc.cdf((m+1)*dt)-delay_rv_cc.cdf((m)*dt))
        cdfs_gp[MaxConv_gp-1] = 1-delay_rv_gp.cdf(MaxConv_gp-1*dt)
        cdfs_cc[MaxConv_cc-1] = 1-delay_rv_cc.cdf((MaxConv_cc-1)*dt)
        return cdfs_gp, cdfs_cc

    def convolution(self, S, lamda, p_gp, p_cc, theta, cdf_gp, cdf_cc):
        T = self._T
        A = self._A
        MaxConv_cc = self.MaxConv_cc
        MaxConv_gp = self.MaxConv_gp
        delta_infec = S*lamda
        shifted = np.roll(delta_infec,1,axis=0) ### Shift in time dimension 
        shifted[0,:] = shifted[1,:]     ### Copy \Delta (t=0) = \Delta(t=1)
        NNI = shifted
        delta = theta*NNI
        mu_gp, mu_cc = np.zeros((T,A)), np.zeros((T,A))
        def convolution_in_C(series1, series2,\
                    conv_window, length, width):
            conv_series = np.zeros((length,width))
            
            for n in range(length):
                for a in range(width):
                    if n < conv_window:
                        con_len = n
                    else:
                        con_len = conv_window-1

                    conv_temp = 0.0
                    for k in range(con_len+1):
                        conv_temp += series1[n-k,a]*series2[k]
                    conv_series[n,a] = conv_temp
            return conv_series
        
        conv_gp = convolution_in_C(delta, cdf_gp, MaxConv_gp, T, A)
        conv_cc = convolution_in_C(delta, cdf_cc, MaxConv_cc, T, A)
        mu_gp = p_gp*conv_gp
        mu_cc = p_cc*conv_cc
        return mu_gp, mu_cc, S, NNI, delta, conv_gp, cdf_cc

    def gp_prob(self, p_gp_par):
        time = self._time
        T = self._T
        A = self._A
        p0, p1, p2, p3, p4, p5, p6, p7, p_cc = p_gp_par
        breakpoints = self._breakpoints_gp
        p_gp = np.zeros((T,A))
        for a in range(A):
            for i in range(T):
                if a < 3 and time[i] < breakpoints[0]:
                    p_gp[i,a] = p0
                elif a >= 3 and time[i] < breakpoints[0]:
                    p_gp[i,a] = p1
                elif a < 3 and (time[i] >= breakpoints[0] and time[i]< breakpoints[1]):
                    p_gp[i,a] = p0*p2
                elif a >= 3 and (time[i] >= breakpoints[0] and time[i]< breakpoints[1]):
                    p_gp[i,a] = p1*p3
                elif a < 3 and (time[i] >= breakpoints[1] and time[i]< breakpoints[2]):
                    p_gp[i,a] = p0*p4
                elif a >= 3 and (time[i] >= breakpoints[1] and time[i]< breakpoints[2]):
                    p_gp[i,a] = p1*p5
                elif a < 3 and time[i] >= breakpoints[2]:
                    p_gp[i,a] = p0*p6
                elif a >= 3 and time[i] >= breakpoints[2]:
                    p_gp[i,a] = p1*p7
        
        return p_gp, p_cc

    def create_mix_matrix(self, m_par):
        A = self._A
        time = self._time
        T = self._T
        m1, m2, m3, m4, m5 = m_par
        m_mix = np.zeros((A,A,T))
        a_base, a_summer, a_hterm = np.ones((A,A))*m1, np.ones((A,A))*m1, np.ones((A,A))*m1
        for i in range(3):
            for j in range(3):
                a_base[i,j] ,a_summer[i,j], a_hterm[i,j] = np.ones(3)
        a_summer[1,1], a_summer[2,2] = m2, m3
        a_hterm[1,1], a_hterm[2,2] = m4, m5

        for i in range(T): 
            if time[i]>=78 and time[i]< 129:
                m_mix[:,:,i] = a_summer*self.m_poly
            elif (time[i]>=23 and time[i]<30) or (time[i]>=177 and time[i]<184) or (time[i]>=233 and time[i]<245):
                m_mix[:,:,i] = a_hterm*self.m_poly
            else:
                m_mix[:,:,i] = a_base*self.m_poly
        return m_mix

    def simulate(self, ode_par, m_par, p_gp_par):
        
        ### Unpack parameters for estimation
        dI, theta, psir, nu = ode_par #
        dI = dI +2
        ### Create Mixing Matrices
        mix_mat = self.create_mix_matrix(m_par)
        
        ### Create GP propensities
        p_gp_par_it = np.copy(p_gp_par)
        p_gp_par_it[:-1] = np.exp(p_gp_par_it[:-1])/(1+ np.exp(p_gp_par_it[:-1]))
        p_gp, p_cc = self.gp_prob(p_gp_par_it)
        
        ## Define fixed parameters
        pi = self._pi #susceptibility
        
        ### Solve the SEEIIR ODE using Euler
        s, lamda = self.transmission(dI, psir, nu, pi, mix_mat, p_gp[0,0])
        
        ### Use the ODE solutions to build the convolution
        cdfs_gp, cdfs_cc = self.conv_cdfs(self._delay_means, self._delay_sds)
        return self.convolution(s, lamda, p_gp, p_cc, theta, cdfs_gp, cdfs_cc)

"""
    def convolution_in_C(self, series1, series2, conv_window, length, width):
        conv_series = np.zeros((length,width))
        for n in range(length):
            for a in range(width):
                if n < conv_window:
                    con_len = n
                else:
                    con_len = conv_window-1
 
                conv_temp = 0.0
                for k in range(con_len+1):
                    conv_temp += series1[n-k,a]*series2[k]
                conv_series[n,a] = conv_temp
        return conv_series
"""