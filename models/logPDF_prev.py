from _pytest.mark import param
import numpy as np
from numpy.core.numeric import Inf
import pandas as pd
import scipy
import scipy.stats as stats
import scipy.special as special

PARAMS = [
    "psir",
    "i0",
    "pi",
    "dL",
    "dI",
]

class LogPosteriorOneRegion(object):
    def __init__(self, forward_model, positive_num, test_num, testing = False, fixed_params = {}):
        assert len(positive_num) == len(test_num)
        self._forward_model = forward_model
        self._positives = positive_num
        self._tests = test_num
        self._fixed_params = pd.Series(fixed_params)
        model_params_to_estimate = [p for p in PARAMS if p not in fixed_params.keys()]
        beta_params = ["beta_rw_sd"] + ["beta[{}]".format(i) for i in range(forward_model._num_betas)]
        self._estimated_params = tuple(model_params_to_estimate + beta_params)
        self._n_params = len(self._estimated_params)
        self._transform = False

    def _log_lik(self, results):
        return sum(stats.binom.logpmf(self._positives, self._tests, results[1]))

    def __call__(self, parameters, joint_reg=False):
        assert len(parameters) == self._n_params
        named_params = self._fixed_params.append(pd.Series(data=parameters, index=self._estimated_params))   
        betas = parameters[-self._forward_model._num_betas:]

        ## Now the priors ####
        # exponential_growth_rate
        log_prior = stats.gamma.logpdf(named_params["psir"], a = 31.36, scale = 1/224)
        # initial infectives
        log_prior += stats.norm.logpdf(named_params["i0"], -17.5, 1.25)
        # initial infected
        log_prior += stats.beta.logpdf(named_params["pi"], 3.44, 23.0)

        
        # Call the simulator to get the expected nnis and prev
        try:
            named_params["i0"] = np.exp(named_params["i0"])
            params_for_sim = named_params[PARAMS].to_dict()
            params_for_sim["beta"] = np.exp(betas)
            result = self._forward_model.simulate(**params_for_sim)
        except ValueError:
            print("Invalid params")
            return -Inf

        # Now calculate the individual data stream log Likelihoods
        log_lik = self._log_lik(result)
                
        # Now calculate the random-walk likelihood
        log_rw_lik = np.sum(stats.norm.logpdf(betas, 0.0, named_params["beta_rw_sd"]))

        return log_lik + log_rw_lik + log_prior
    
    def sample(self, parameters):
        assert len(parameters) == self.n_params                 
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
        nnis = self._forward_model.simulate(dI, psir, beta_rw, I0, p_dead, m_mul, generate=True)
        ag_mu_gp = np.add.reduceat(nnis,np.arange(0,len(nnis),2))
        return  nnis#self.death_lik(ag_mu_gp, eta, sample = True)

    def generate_R(self, parameters, joint_reg=False): 
        assert len(parameters) == self._n_params          
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
            
        


