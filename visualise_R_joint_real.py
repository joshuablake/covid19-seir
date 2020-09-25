#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from models.logPDF_Cpp import LogPosteriorRegionWise
from models.forwardModel_vecage import ForwardModelRegionWise
from mcmc.adaptiveMetropolis import AdaptiveMetropolis
from mcmc.block_mcmc import MCMC
from contact_m.generate_contact_matrices import breakpoint, daily_mixing_matrices
import scipy.stats as stats
import pandas as pd
import time as timer
import random
TEST_ENABLED = False
#######################################################################################
####################      DEFINE CONSTANTS     ########################################
#######################################################################################
dt = 0.5
time = np.arange(0,166,dt)
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
idx_w = pd.date_range('2020-02-17 00:00:00', '2020-07-31 12:00:00', freq='W-TUE')
idx_d = pd.date_range('2020-02-17 00:00:00', '2020-07-31 12:00:00', freq='12H')

ts_d = pd.Series(range(len(idx_d)), index=idx_d)
breakpoints = np.array(breakpoint(ts_d, idx_d, idx_w)[1])
list_brk_beta = [breakpoints[5],breakpoints[-3], breakpoints[-1]]
list_brk_mix = [breakpoints[5],*breakpoints[-2:]]
print(breakpoints[6], breakpoints[5])
daily_mix_matrix = daily_mixing_matrices(dt, idx_d, list_brk_mix, regions)
#######################################################################################
####################      DATA LOADING         ########################################
#######################################################################################
Paul_chain = pd.read_csv("./results/chain_sanmitra.csv")
dI = np.mean(Paul_chain['infectious_period1'].as_matrix()[:,None],axis=0)
ksens = np.mean(Paul_chain['sero_test_sensitivity1'].as_matrix()[:,None],axis=0)
kspec = np.mean(Paul_chain['sero_test_specificity1'].as_matrix()[:,None],axis=0)
eta = np.mean(Paul_chain['hosp_negbin_overdispersion1'].as_matrix()[:,None],axis=0)
p_dead = Paul_chain.as_matrix(columns=Paul_chain.columns[37:37+7]).mean(axis=0)
beta_rw_sd = np.mean(Paul_chain['log_beta_rw_sd2'].as_matrix()[:,None],axis=0)
p_m_mul = Paul_chain.as_matrix(columns=Paul_chain.columns[2:2+21]).mean(axis=0)
p_beta_rw = Paul_chain.as_matrix(columns=Paul_chain.columns[46:46+112]).mean(axis=0)
psir = Paul_chain.as_matrix(columns=Paul_chain.columns[23:23+7]).mean(axis=0)
I0 = Paul_chain.as_matrix(columns=Paul_chain.columns[30:30+7]).mean(axis=0)

m_mul = np.array(np.split(p_m_mul,regions,axis=0)).reshape((regions,3)).T 
beta_rw = p_beta_rw

beta_rw = np.array(np.split(beta_rw,regions, axis=0)).T 

regions_names = ["East_of_England",
                "London",
                "Midlands",
                "North_East_and_Yorkshire",
                "North_West",
                "South_East",
                "South_West"]
path_prefix = "20200717_"
path_postfix = "_8ag_samples"
sero_denom = np.zeros((1,1,regions))
dummy_data = [np.zeros((1,1,regions)), np.zeros((1,1,regions))]  

burnin = -200000
thin = 200
param_filename = './results/_mcmc_test_joint_real_gamcorprhalfmil.p'
trace = pickle.load( open( param_filename , "rb" ) )
trace_post_burn = trace[burnin:,:]
py_thinned_trace_amgs = trace_post_burn[::thin,:]
x_axis = pd.date_range('2020-02-17', '2020-07-31', freq='D')
region_starts = [12 + (i*21) for i in range(regions)]
true_param = []
true_param.append(dI)
true_param.append(ksens)
true_param.append(kspec)
true_param.append(eta)
true_param.append(p_dead)
true_param.append(beta_rw_sd)
for RUN_REGION in range(regions):
    true_param.append(psir[RUN_REGION])
    true_param.append(I0[RUN_REGION])
    true_param.append(m_mul[:,RUN_REGION])
    true_param.append(beta_rw[:,RUN_REGION])
    X_true = np.hstack(true_param)
for RUN_REGION in range(regions):
    forward_model = ForwardModelRegionWise(dt, time, A, N[:,RUN_REGION], daily_mix_matrix, \
                    breakpoints[6], breakpoints[5], list_brk_beta, idx_d)
    forward_model.real_data()
    logP = LogPosteriorRegionWise(forward_model, RUN_REGION, dummy_data, sero_denom[:,:,RUN_REGION])
    X_true_g = X_true[:12]
    X_true_l = X_true[region_starts[RUN_REGION]:region_starts[RUN_REGION]+21]
    X_true_gl = np.hstack((X_true_g,X_true_l))
    R = logP.generate_R(X_true_gl, True)
    Rt = R[1::2] 

    trace_regional = py_thinned_trace_amgs[:, region_starts[RUN_REGION]:region_starts[RUN_REGION]+21]
    trace_global = py_thinned_trace_amgs[:,:12]
    trace_ppc = np.concatenate((trace_global,trace_regional),axis=1)
    ppc = []
    rand_range = random.sample(range(0, 1000), 500)
    for i in rand_range:
        R_ppc = logP.generate_R(trace_ppc[i,:],True)
        Rt_ppc = R_ppc[1::2]    
        ppc.append(Rt_ppc)

    ppc_samples = np.array(ppc)
    ppc_mean = np.percentile(ppc_samples,50,axis=0)
    ppc_uq = np.percentile(ppc_samples,97.5,axis=0)
    ppc_lq = np.percentile(ppc_samples,2.5,axis=0)
    plt.plot(x_axis, ppc_lq,'--', color='blue')
    plt.plot(x_axis, ppc_uq,'--', color='blue')
    plt.plot(x_axis, Rt, linewidth=2., color='red', label='Rt_Paul_'+regions_names[RUN_REGION])
    plt.plot(x_axis, ppc_mean, linewidth=2.5, color='blue', label='Rt_mean_'+regions_names[RUN_REGION])
    plt.legend()
    plt.ylabel(r"$R_{t}$")
    plt.savefig('./figures/joint/R_estimates_real/R_estimate_1_R'+str(RUN_REGION)+'.png')
    plt.close()
    print('Done region :'+str(RUN_REGION))
    






