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
TEST_ENABLED = False
seed=112
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
daily_mix_matrix = daily_mixing_matrices(dt, idx_d, list_brk_mix, regions)
#######################################################################################
####################      DATA LOADING         ########################################
#######################################################################################
dI = [0.472845081832237]
ksens = [0.6948124]
kspec = [0.9682104]
eta = [5.0]
psir = np.array([0.2759202, 0.2319129, 0.2385402, 0.2977007, 0.2418800, 0.2210527, 0.2229998])
I0 = np.array([-16.63574, -13.90437, -15.33340, -17.67026, -15.51458, -15.29906, -16.09737])
p_dead = [1.149820e-05, 2.444837e-05, 5.711427e-05, 3.656723e-04,
                        4.959340e-03, 3.428846e-02, 2.210078e-01]
beta_rw_sd = [0.1288072]
m_mul = np.array(np.split(np.loadtxt('./model_data/contact_parameters.txt'),regions,axis=0)).reshape((regions,3)).T 
beta_rw = np.loadtxt('./model_data/log_beta_rw.txt')
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
sero_denom = np.zeros((152,A,regions))
dummy_data = [np.zeros((1,1,regions)), np.zeros((1,1,regions))]
for r, region_name in enumerate(regions_names):
    path = path_prefix+region_name+path_postfix+'.txt'
    sero_denom[:,:,r] = np.loadtxt('./model_data/SeroSanmitra/'+path,usecols=(1,2,3,4,5,6,7,8)).astype(np.int64)

x_axis = pd.date_range('2020-02-17', '2020-08-01', freq='D')
for RUN_REGION in range(regions):
    true_param = []
    true_param.append(dI)
    true_param.append(ksens)
    true_param.append(kspec)
    true_param.append(eta)
    true_param.append(psir[RUN_REGION])
    true_param.append(I0[RUN_REGION])
    true_param.append(p_dead)
    true_param.append(beta_rw_sd)
    true_param.append(m_mul[:,RUN_REGION])
    true_param.append(beta_rw[1:,RUN_REGION])
    X_true = np.hstack(true_param)
    forward_model = ForwardModelRegionWise(dt, time, A, N[:,RUN_REGION], daily_mix_matrix, \
                    breakpoints[6], breakpoints[5], list_brk_beta, idx_d)
    
    logP = LogPosteriorRegionWise(forward_model, RUN_REGION, dummy_data, sero_denom[:,:,RUN_REGION])
    death_data, sero_data = logP.sample(X_true)
    np.savetxt('./data/deaths_2_r'+str(RUN_REGION)+'.txt', death_data)
    np.savetxt('./data/sero_assay_2_r'+str(RUN_REGION)+'.txt', sero_data)
    plt.plot(x_axis, np.sum(death_data,axis=1))
    plt.savefig('./figures/generative/generative_2_R'+str(RUN_REGION)+'.png')



