#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from models.logPDF_Cpp import LogPosteriorRegionWise
from models.forwardModel_Cpp import ForwardModelRegionWise
from mcmc.adaptiveMetropolis import AdaptiveMetropolis
from mcmc.block_mcmc import MCMC
from contact_m.generate_contact_matrices import breakpoint, daily_mixing_matrices
import scipy.stats as stats
import pandas as pd
import time as timer
import random
TEST_ENABLED = False

Paul_chain = pd.read_csv("./results/chain_sanmitra.csv")
dI = Paul_chain['infectious_period1'].as_matrix()[:,None]
ksens = Paul_chain['sero_test_sensitivity1'].as_matrix()[:,None]
kspec = Paul_chain['sero_test_specificity1'].as_matrix()[:,None]
eta = Paul_chain['hosp_negbin_overdispersion1'].as_matrix()[:,None]
p_dead = Paul_chain.as_matrix(columns=Paul_chain.columns[37:37+7])
beta_rw_sd = Paul_chain['log_beta_rw_sd2'].as_matrix()[:,None]
m_mul = Paul_chain.as_matrix(columns=Paul_chain.columns[2:2+21])
beta_rw = Paul_chain.as_matrix(columns=Paul_chain.columns[46:46+112])
psir = Paul_chain.as_matrix(columns=Paul_chain.columns[23:23+7])
I0 = Paul_chain.as_matrix(columns=Paul_chain.columns[30:30+7])


burnin = -96000
thin = 16
param_filename = './results/_mcmc_test_joint_real_200k.p'
trace = pickle.load( open( param_filename , "rb" ) )
trace_post_burn = trace[burnin:,:]
py_thinned_trace_amgs = trace_post_burn[::thin,:]
np.savetxt('./results/Sanmitra_chain.csv', py_thinned_trace_amgs,delimiter=",",newline='\r\n')
#pd.write_csv('./results/Sanmitra_chain.csv', py_thinned_trace_amgs)
#######################################################################################
####################      DEFINE CONSTANTS     ########################################
#######################################################################################
dt = 0.5
time = np.arange(0,167,dt)
regions = int(7)
A = int(8)
T = int(len(time))
N = np.array([74103, 318183, 804260, 704025, 1634429, 1697206, 683583, 577399, 
122401, 493480, 1123981, 1028009, 3063113, 2017884, 575433, 483780, 
118454, 505611, 1284743, 1308343, 2631847, 2708355, 1085138, 895188, 
92626, 396683, 1014827, 1056588, 2115517, 2253914, 904953, 731817, 
79977, 340962, 851539, 845215, 1786666, 1820128, 710782, 577678, 
94823, 412569, 1085466, 1022927, 2175624, 2338669, 921243, 801040, 
55450, 241405, 633169, 644122, 1304392, 1496240, 666261, 564958]).reshape((regions,A)).T

#######################################################################################
############## Create Date-time and generate Mixing Matrices    #######################
#######################################################################################
idx_w = pd.date_range('2020-02-17 00:00:00', '2020-08-01 12:00:00', freq='W-TUE')
idx_d = pd.date_range('2020-02-17 00:00:00', '2020-08-01 12:00:00', freq='12H')

ts_d = pd.Series(range(len(idx_d)), index=idx_d)
breakpoints = np.array(breakpoint(ts_d, idx_d, idx_w)[1])
list_brk_beta = [breakpoints[5],breakpoints[-3], breakpoints[-1]]
list_brk_mix = [breakpoints[5],*breakpoints[-2:]]
print(breakpoints[6], breakpoints[5])
daily_mix_matrix = daily_mixing_matrices(dt, idx_d, list_brk_mix, regions)
#######################################################################################
####################      DATA LOADING         ########################################
#######################################################################################
death_data = np.zeros((int(len(time)/2),A,regions))
days_to_load = int(len(time)/2)
sero_data = np.zeros((152,A,regions))

data = [death_data, sero_data]
regions_names = ["East_of_England",
                 "London",
                 "Midlands",
                 "North_East_and_Yorkshire",
                 "North_West",
                 "South_East",
                 "South_West"]
sero_denom_path_prefix = "20200717_"
death_path_prefix = "deaths20200807_"
sero_denom_path_postfix = "_8ag_samples"
sero_assay_path_postfix = "_8ag_positives"
death_path_postfix = "_8agCONF"
sero_denom = np.zeros((152,A,regions))
for r, region_name in enumerate(regions_names):
        sero_denom_path = sero_denom_path_prefix + \
                region_name + sero_denom_path_postfix + '.txt'
        sero_denom[:,:,r] = np.loadtxt('./model_data/SeroSanmitra/' + \
                sero_denom_path,usecols=(1,2,3,4,5,6,7,8))
        sero_assay_path = sero_denom_path_prefix + \
                region_name + sero_assay_path_postfix + '.txt'
        sero_data[:,:,r] = np.loadtxt('./model_data/SeroSanmitra/' + \
                sero_assay_path,usecols=(1,2,3,4,5,6,7,8))
        death_path = death_path_prefix + \
                region_name + death_path_postfix+'.txt'     
        death_data[:,:,r] = np.loadtxt('./data/real_data/' + \
                death_path,usecols=(1,2,3,4,5,6,7,8))[:days_to_load,:] 
death_data_int = death_data.astype(int)
sero_data_int = sero_data.astype(int)
data = [death_data_int, sero_data_int]


x_axis = pd.date_range('2020-02-17', '2020-08-01', freq='D')
region_starts = [12 + (i*21) for i in range(regions)]

PPC_REG=[]                     
for RUN_REGION in range(regions):
 
        forward_model = ForwardModelRegionWise(dt, time, A, N[:,RUN_REGION], daily_mix_matrix, \
                        breakpoints[6], breakpoints[5], list_brk_beta, idx_d)
        forward_model.real_data()
        logP = LogPosteriorRegionWise(forward_model, RUN_REGION, 
        data, sero_denom[:,:,RUN_REGION])
        psir_I0 = np.hstack((psir[:,RUN_REGION][:,None], I0[:,RUN_REGION][:,None]))
        true_param = np.hstack((dI[:1000,:],ksens[:1000,:], kspec[:1000,:], eta[:1000,:],\
        p_dead[:1000,:],beta_rw_sd[:1000,:], psir_I0[:1000,:], \
        m_mul[:1000,RUN_REGION*3:3 + RUN_REGION*3], beta_rw[:1000, \
                RUN_REGION*16:16 + RUN_REGION*16]))   

        if RUN_REGION==0:
                gbp = []
                for i in range(3):
                        gbp.append(logP.global_priors(true_param[i,:12]))
                gbp=np.array(gbp)

        ppc_paul = []
        rand_range = random.sample(range(0, 1000), 500)
        for i in rand_range:
                ppc_paul.append(logP(true_param[i,:],True))#+                        logP.global_priors(true_param[i,:12]))        
        ppc_paul_ar = np.array(ppc_paul)
        PPC_REG.append(ppc_paul_ar)

        trace_regional = py_thinned_trace_amgs[:, region_starts[RUN_REGION]:region_starts[RUN_REGION]+21]
        trace_global = py_thinned_trace_amgs[:,:12]
        trace_ppc = np.concatenate((trace_global,trace_regional),axis=1)

        ppc_sanmitra = []
        rand_range = random.sample(range(0, 1000), 500)
        for i in rand_range:
                ppc_sanmitra.append(logP(trace_ppc[i,:],True))#+                         logP.global_priors(trace_ppc[i,:12]))
        ppc_sanmitra_ar = np.array(ppc_sanmitra)
        plt.hist(ppc_sanmitra_ar,20,alpha=0.5,color='blue', label='Sanmitra')
        plt.hist(ppc_paul_ar,20,alpha=0.5,color='red', label='Paul')

        plt.legend()
        plt.xlabel('LogP', fontsize=20)
        plt.ylabel('R_'+str(RUN_REGION), fontsize=20)
        plt.savefig('./figures/joint/ppc_real/'+str(RUN_REGION)+'CPP_logp.png')
        plt.close()
        print('Done region :'+str(RUN_REGION))

    






