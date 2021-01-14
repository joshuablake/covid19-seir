import numpy as np
import scipy
import scipy.stats as stats
import scipy.special as special
# import pints

class LogPosteriorRegionWise(object):#pints.LogPDF

    def __init__(self, forward_model, which_region, data, sero_denom, testing = False):

        self._forward_model = forward_model
        self.death_data = data[0][:,:,which_region]
        self.sero_data = data[1][:,:,which_region]
        self.sero_denom = sero_denom
        self.sero_length = 152
        
        #self._N = pop_size_N
        self._testing = testing

        self.n_params = 27
        self.n_params_global = 12
        self.n_params_regional = 15
        #self.n_params = self.n_params_global + self.n_params_regional
        self._transform = False

    def death_lik(self, ag_mu_death, eta, sample = False):
        if sample:
            return stats.nbinom(n=r, p=1-prob).rvs()
        else:
            if np.any(ag_mu_death<0):
                return -np.inf
            else:
                lfx = 0.0
                #r = ag_mu_death/eta
                #prob = 1 - (1/(eta+1))
                for i in range(len(ag_mu_death)):
                    for j in range(8):
                        mu = ag_mu_death[i,j]
                        x = self.death_data[i,j]
                        if((mu == 0) & (x == 0)):
                            lfx += 0.
                        elif ((mu == 0) & (x != 0)):
                            lfx += -np.inf
                        else:
                            if (eta > 1.5e-08):
                                r = mu / eta
                                lfx += special.loggamma(x + r) - special.loggamma(r)
                                p = 1 - (1 / (eta + 1))
                                lfx += ((r * np.log(1 - p)) + (x * np.log(p)))
                            else:
                                lfx += (x * np.log(mu)) - mu
                return lfx

    def sero_lik(self, S, ksens, kspec, sample = False):

        ag_S = S[1::2,:]
        ag_S =ag_S[:self.sero_length,:] ### change this later
        sero_ratio = ag_S/self._forward_model._N
        prob = (ksens*(1 - sero_ratio)) + ((1 - kspec)*sero_ratio)
        if sample:           
            return stats.binom(n=np.array(self.sero_denom, dtype=np.int64), p=prob).rvs()
        else:
            return np.sum(stats.binom(n=np.array(self.sero_denom, dtype=np.int64), 
            p=prob).logpmf(self.sero_data))

    def __call__(self, parameters, joint_reg=False):
        assert len(parameters) == self.n_params
        if not joint_reg:
            dI = parameters[0]
            ksens, kspec, eta = parameters[1:4]
            psir, I0, p_dead  = parameters[4], parameters[5], parameters[6:13]
            
        else:
            dI = parameters[0]
            ksens, kspec, eta = parameters[1:4]
            p_dead = parameters[4:11]
            beta_rw_sd = parameters[11]
            psir, I0  = parameters[12], parameters[13]  
                     
        m_mul = parameters[14:17]          
        beta_rw = parameters[17:27]  
        """   
        init = beta_rw[-1]     
        beta_rw = np.hstack((beta_rw[:-1], [beta_rw_sd])) 
        beta_rw = np.hstack((beta_rw, [init])) 
        """
        beta_rw = np.hstack((beta_rw, [beta_rw_sd]))
        # Call the simulator to get the expected death, sero series
        mu_death, S_ = self._forward_model.simulate(dI, psir, beta_rw, I0, p_dead, m_mul)

        # Aggregate half days for expected death series
        ag_mu_gp = np.add.reduceat(mu_death,np.arange(0,len(mu_death),2))

        # Now calculate the individual data stream log Likelihoods
        log_death_lik = self.death_lik(ag_mu_gp, eta)
        log_sero_lik = self.sero_lik(S_, ksens, kspec)
                
        # Now calculate the random-walk likelihood
        
        log_rw_lik = np.sum(stats.norm.logpdf(beta_rw[:-1],0.0,1.0))
        
        log_likelihood =  log_death_lik + log_sero_lik #+ log_rw_lik

        ## Now the priors #####
        # exponential_growth_rate
        logPrior_psir = stats.gamma.logpdf(psir, a = 31.36, scale = 1/224)
        # initial infectives
        logPrior_I0 = stats.norm.logpdf(I0, -17.5, 1.25)
        # contact multipliers
        logPrior_m_mul = np.sum(stats.norm.logpdf(m_mul,-0.1115718, 0.4723807)) 
        
        log_prior = logPrior_psir + logPrior_I0 + logPrior_m_mul 
        return log_likelihood #+ log_prior
    
    def sample(self, parameters, joint_reg=False):
        assert len(parameters) == self.n_params        
        if not joint_reg:            
            dI = parameters[0]
            ksens, kspec, eta = parameters[1:4]
            psir, I0, p_dead  = parameters[4], parameters[5], parameters[6:13]
            beta_rw_sd = parameters[13]
            
        else:            
            dI = parameters[0]
            ksens, kspec, eta = parameters[1:4]
            p_dead = parameters[4:11]
            beta_rw_sd = parameters[11]
            psir, I0  = parameters[12], parameters[13]                     

        m_mul = parameters[14:17]          
        beta_rw = parameters[17:33]  
        beta_rw = np.hstack(([0], beta_rw))
        # Call the simulator to get the expected deaths, sero assay series
        first_beta = beta_rw[:1]
        assert first_beta == 0
        mu_death, S_ = self._forward_model.simulate(dI, psir, beta_rw, I0, p_dead, m_mul)

        # Aggregate half days for death series
        ag_mu_gp = np.add.reduceat(mu_death,np.arange(0,len(mu_death),2))

        # Now sample the individual data streams
        death_ppc = self.death_lik(ag_mu_gp, eta, True)
        sero_ppc = self.sero_lik(S_, ksens, kspec, True)  
        return  death_ppc, sero_ppc 

    def generate_R(self, parameters, joint_reg=False): 
        assert len(parameters) == self.n_params          
        if not joint_reg:            
            dI = parameters[0]
            ksens, kspec, eta = parameters[1:4]
            psir, I0, p_dead  = parameters[4], parameters[5], parameters[6:13]
            beta_rw_sd = parameters[13]
            
        else:            
            dI = parameters[0]
            ksens, kspec, eta = parameters[1:4]
            p_dead = parameters[4:11]
            beta_rw_sd = parameters[11]
            psir, I0  = parameters[12], parameters[13]                     

        m_mul = parameters[14:17]          
        beta_rw = parameters[17:33]  
        beta_rw = np.hstack(([0], beta_rw))
        # Call the simulator to get the expected deaths, sero assay series
        first_beta = beta_rw[:1]
        assert first_beta == 0
        return self._forward_model.simulate(dI, psir, beta_rw, I0, \
            p_dead, m_mul, get_R = True)

    def global_priors(self, global_parameters):
        assert len(global_parameters) == self.n_params_global
        dI = global_parameters[0]
        ksens, kspec, eta = global_parameters[1:4]
        p_dead = global_parameters[4:11]
        beta_rw_sd = global_parameters[11]                

        # infectious_period 
        logPrior_dI = stats.gamma.logpdf(dI, a = 1.43, scale = 1/0.549)# add 1.43,
        # sero sensitivity
        logPrior_ksens = stats.beta.logpdf(ksens, 23.5, 9.5)# 23.5, 9.5
        # sero specificity
        logPrior_kspec = stats.beta.logpdf(kspec, 569.5, 5.5) #569.5, 5.5
        # negbin_overdispersion
        logPrior_eta = stats.gamma.logpdf(eta, a = 1.0, scale = 1/0.2)# rate parameterisation 

        # prop_of_death (IFR)
        p_dead_loc = np.array([1., 1., 1., 1., 1., 1., 9.5])
        p_dead_scale = np.array([62110.801242236, 23363.4859813084, \
            5290.00529100529, 1107.64745011086, 120.951219512195, \
            31.1543408360129, 112.])
        
        logPrior_p_dead = np.sum(stats.beta.logpdf(p_dead, p_dead_loc, p_dead_scale))        
        # random-walk sd
        logPrior_beta_rw_sd = stats.gamma.logpdf(beta_rw_sd, a = 1.0, scale = 1/100)#WIP was 1/100
        #logPrior_beta_rw_sd = stats.invgamma.logpdf(beta_rw_sd,a=5.0,scale=5.0)                             

        logGlobPrior = logPrior_dI + logPrior_ksens + logPrior_kspec + \
            logPrior_eta + logPrior_p_dead + logPrior_beta_rw_sd    
        return logGlobPrior
        
    def n_parameters(self):
        return self.n_params
            
        


