import numpy as np
import scipy
import scipy.stats as stats
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

        self.n_params = 34
        self._transform = False

    def death_lik(self, ag_mu_death, eta, sample = False):
        
        r = ag_mu_death/eta
        prob = (r/(r+ag_mu_death))
        if sample:
            return stats.nbinom(n=r, p=prob).rvs()
        else:
            return np.sum(stats.nbinom.logpmf(self.death_data, n=r, p=prob ))

    def sero_lik(self, S, ksens, kspec, sample = False):

        ag_S = S[1::2,:]
        ag_S =ag_S[:self.sero_length,:] ### change this later
        sero_ratio = ag_S/self._forward_model._N
        prob = (ksens*(1 - sero_ratio)) + ((1 - kspec)*sero_ratio)
        if sample:
            return stats.binom(n=np.array(self.sero_denom, dtype=np.int64), p=prob).rvs()
        else:
            return np.sum(stats.binom(n=np.array(self.sero_denom, dtype=np.int64), p=prob).logpmf(self.sero_data))

    def __call__(self, parameters, joint_reg=False):
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
        beta_rw = parameters[17:34]          
    
        # Call the simulator to get the GP consultation, Hospitalisation series
        mu_death, S_ = self._forward_model.simulate(dI, psir, beta_rw, I0, p_dead, m_mul)

        # Aggregate all series
        ag_mu_gp = np.add.reduceat(mu_death,np.arange(0,len(mu_death),2))


        # Now calculate the individual data stream log Likelihoods
        log_death_lik = self.death_lik(ag_mu_gp, eta)
        log_sero_lik = self.sero_lik(S_, ksens, kspec)
                
        # Now calculate the random-walk likelihood
        log_rw_lik = np.sum(stats.norm.logpdf(beta_rw,0.0,beta_rw_sd))

        log_likelihood = log_death_lik + log_sero_lik + log_rw_lik

        ## Now the priors ####

        # infectious_period 
        logPrior_dI = stats.gamma.logpdf(dI, a = 1.43, scale = 1/0.549)# add 1.43,
        # sero sensitivity
        logPrior_ksens = stats.beta.logpdf(ksens, 23.5, 9.5)
        # sero specificity
        logPrior_kspec = stats.beta.logpdf(kspec, 569.5, 5.5) 
        # negbin_overdispersion
        logPrior_eta = stats.gamma.logpdf(eta, a = 1.0, scale = 1/0.2)# rate parameterisation             
        # exponential_growth_rate
        logPrior_psir = stats.gamma.logpdf(psir, a = 31.36, scale = 1/224)
        # initial infectives
        logPrior_I0 = stats.norm.logpdf(I0, -17.5, 1.25)
        # prop_of_death
        p_dead_loc = np.array([1., 1., 1., 1., 1., 1., 9.5])
        p_dead_scale = np.array([62110.801242236, 23363.4859813084, \
            5290.00529100529, 1107.64745011086, 120.951219512195, \
            31.1543408360129, 112.])
        logPrior_p_dead = np.sum(stats.beta.logpdf(p_dead, p_dead_loc, p_dead_scale))        
        # random-walk sd
        logPrior_beta_rw_sd = stats.gamma.logpdf(beta_rw_sd, a = 1.0, scale = 1/100)
        # contact multipliers
        logPrior_m_mul = np.sum(stats.norm.logpdf(m_mul,-0.1115718, 0.4723807**2)) 
         

        log_prior = logPrior_dI + logPrior_ksens + logPrior_kspec + logPrior_eta + \
                    + logPrior_psir + logPrior_I0 + logPrior_p_dead + \
                        logPrior_beta_rw_sd + logPrior_m_mul 
        return log_likelihood + log_prior
    
    def sample(self, parameters):
        dI = parameters[0]
        ksens, kspec, eta = parameters[1:4]
        psir, I0, p_dead  = parameters[4], parameters[5], parameters[6:13]
        beta_rw_sd = parameters[13]
        m_mul = parameters[14:17]
        
        beta_rw = parameters[17:]
        
        # Call the simulator to get the GP consultation, Hospitalisation series
        mu_death, S_ = self._forward_model.simulate(dI, psir, beta_rw, I0, p_dead, m_mul)

        # Aggregate all series
        ag_mu_gp = np.add.reduceat(mu_death,np.arange(0,len(mu_death),2))


        # Now calculate the individual data stream log Likelihoods
        death_ppc = self.death_lik(ag_mu_gp, eta, True)
        sero_ppc = self.sero_lik(S_, ksens, kspec, True)  
        return  death_ppc, sero_ppc    
                   
    def n_parameters(self):
        return self.n_params
            
        


