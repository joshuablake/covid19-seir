import numpy as np
import scipy
import scipy.stats as stats
# import pints

class LogPosterior(object):#pints.LogPDF

    def __init__(self, forward_model, background_model, data, denoms, pop_size_N, bg_prior, hosp_time = None, NPFS = None, testing = False):

        self._forward_model = forward_model
        self._background_model = background_model
        self.gp_data = data[0]
        self.hosp_data = data[1]
        self.viro_data = data[2]
        self.sero_data = data[3]

        self.gp_denom = denoms[0]
        self.viro_denom = denoms[1]
        self.sero_denom = denoms[2]
        
        if NPFS == None:
            self._NPFS = 83
        else:
            self._NPFS = NPFS
        if hosp_time == None:
            self._hosp_time = 50
        else:
            self._hosp_time = hosp_time

        self._N = pop_size_N
        self._bg_prior = bg_prior
        self._testing = testing

        self.n_params = 49

    def gp_lik(self, ag_mu_gp, eta, bg_gp, sample = False):
        N = self._N
        N_gp = np.hstack((N[0]+N[1],N[2:]))
        NPFS = self._NPFS
      
        eta1=eta[0]
        eta2=eta[1]
        eta_NPFS = np.concatenate((np.ones(NPFS)*eta1, np.ones(len(ag_mu_gp)-NPFS)*eta2)).reshape((len(ag_mu_gp),1))
        total = ag_mu_gp + bg_gp
    
        col0 = total[:,0] + total[:,1]
        col0 = col0.reshape((len(ag_mu_gp),1))
        total = np.hstack((col0, total[:,2:]))
        gp_denom_norm = self.gp_denom/N_gp
        gp = total*gp_denom_norm
        r = gp/eta_NPFS
        prob = (r/(r+gp))
        if sample:
            return stats.nbinom(n=r, p=prob).rvs()
        else:
            return np.sum(stats.nbinom.logpmf(self.gp_data, n=r, p=prob ))

    def hosp_lik(self, ag_mu_cc, sample = False):
        hosp_time = self._hosp_time
        cc = ag_mu_cc[:hosp_time,:]
        if sample:
            return stats.poisson(mu=cc).rvs()
        else:
            return np.sum(stats.poisson.logpmf(self.hosp_data, mu=cc))    

    def sero_lik(self, S, sample = False):
        N = self._N
        ag_S = S[1::2,:]
        """
        sero=np.empty_like(ag_S)
        for i in range(len(ag_S)):
            sero[i,:] = 1 - ag_S[i,:]/N
        """
        sero = 1 - ag_S/N
        if sample:
            return stats.binom(n=np.array(self.sero_denom, dtype=np.int64), p=sero).rvs()
        else:
            return np.sum(stats.binom.logpmf(self.sero_data[74:119,:],n=self.sero_denom[74:119,:], p=sero[74:119,:]))

    def viro_lik(self, ag_mu_gp, bg_gp, sample = False):

        excess = ag_mu_gp
        total = ag_mu_gp + bg_gp
        viro = excess/total
        if sample:
            return stats.binom(n=np.array(self.viro_denom, dtype=np.int64), p=viro).rvs()
        else:
            return np.sum(stats.binom.logpmf(self.viro_data, n=self.viro_denom, p=viro))

    def __call__(self, parameters):

        p_gp_params = parameters[:9]
        m_params = parameters[9:14]
        ode_params = parameters[14:18]
        eta = np.array(parameters[18:20])
        bg_params = parameters[20:]  

        # Call the simulator to get the GP consultation, Hospitalisation series
        mu_gp, mu_cc, S, _,_,_,_ = self._forward_model.simulate(ode_params, m_params, p_gp_params)

        # Get the background series
        bg = self._background_model.evaluate(bg_params)

        # Aggregate all series
        ag_mu_gp = np.add.reduceat(mu_gp,np.arange(0,len(mu_gp),2))
        ag_mu_cc = np.add.reduceat(mu_cc,np.arange(0,len(mu_cc),2))                
        ag_bg = np.add.reduceat(bg,np.arange(0,len(bg),2))

        # Some normalisation
        ag_bg = ag_bg*self._N/(100000 * 365.25 * 2)

        # Now calculate the individual data stream log Likelihoods
        log_gp_lik = self.gp_lik(ag_mu_gp, eta, ag_bg)
        log_hosp_lik = self.hosp_lik(ag_mu_cc)
        log_viro_lik = self.viro_lik(ag_mu_gp, ag_bg)
        log_sero_lik = self.sero_lik(S)
        log_likelihood = log_gp_lik + log_hosp_lik + log_viro_lik + log_sero_lik
        if self._testing:
            print('In Testing')
            return log_likelihood, log_gp_lik, log_hosp_lik, log_viro_lik, log_sero_lik

        # Now calculate the necessary priors for ODE model params

        # negbin_overdispersion
        logPrior_eta = np.sum(stats.gamma.logpdf(eta, a = 2.0, scale = 1/0.01))# rate parameterisation
        # infectious_period 
        logPrior_dI = stats.gamma.logpdf(ode_params[0], a = 517.6839, scale = 1/357.0234)# add 
        # prop_symptomatic
        logPrior_theta = stats.beta.logpdf(ode_params[1], 32.5, 18.5)
        # exponential_growth_rate
        logPrior_psir = stats.gamma.logpdf(ode_params[2], a = 6.31775, scale = 1/56.61066)
        # log_p_lambda_0
        logPrior_nu = stats.norm.logpdf(ode_params[3], -19.1495, 4.05411)
        
        # priors for propensity params and contact matrix
        logPrior_contact = np.sum(stats.beta.logpdf(m_params, 1.0, 1.0))
        # prop_case_to_GP_consultation
        p_GP_loc = np.array([-0.1870785, -0.1870785, 0.4258796, 0.4258796, -0.3186211, -0.3186211, -0.2840577, -0.2840577])
        p_GP_scale = np.array([0.4074365, 0.4074365, 0.9640366, 0.9640366, 0.5126546, 0.5126546, 0.5134608, 0.5134608])
        logPrior_p_GP = np.sum(stats.norm.logpdf(p_gp_params[:-1], p_GP_loc, p_GP_scale))
        # prop_case_to_hosp
        logPrior_p_cc = stats.beta.logpdf(p_gp_params[-1], 1.0289, 2.6922)

        # priors for background model params
        logPrior_bg = stats.multivariate_normal.logpdf(bg_params, self._bg_prior[:,0], self._bg_prior[:,1:])

        
        log_prior = logPrior_eta + logPrior_dI + logPrior_theta + logPrior_psir \
                    + logPrior_nu + logPrior_contact + logPrior_p_GP + logPrior_p_cc + logPrior_bg

        return log_likelihood + log_prior

    def sample(self, parameters):

            p_gp_params = parameters[:9]
            m_params = parameters[9:14]
            ode_params = parameters[14:18]
            eta = np.array(parameters[18:20])
            bg_params = parameters[20:]  

            # Call the simulator to get the GP consultation, Hospitalisation series
            mu_gp, mu_cc, S, _,_,_,_ = self._forward_model.simulate(ode_params, m_params, p_gp_params)

            # Get the background series
            bg = self._background_model.evaluate(bg_params)

            # Aggregate all series
            ag_mu_gp = np.add.reduceat(mu_gp,np.arange(0,len(mu_gp),2))
            ag_mu_cc = np.add.reduceat(mu_cc,np.arange(0,len(mu_cc),2))                
            ag_bg = np.add.reduceat(bg,np.arange(0,len(bg),2))

            # Some normalisation
            ag_bg = ag_bg*self._N/(100000 * 365.25 * 2)

            # Now calculate the individual data stream log Likelihoods
            gp_sample = self.gp_lik(ag_mu_gp, eta, ag_bg, sample = True)
            hosp_sample = self.hosp_lik(ag_mu_cc, sample = True)
            viro_sample = self.viro_lik(ag_mu_gp, ag_bg, sample = True)
            sero_sample = self.sero_lik(S, sample = True)

            return [gp_sample, hosp_sample, viro_sample, sero_sample]
                   
    def n_parameters(self):
        return self.n_params
            
        


