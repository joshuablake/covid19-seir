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
#idx_w = pd.date_range('2020-02-17 00:00:00', '2020-08-01 12:00:00', freq='W-TUE')
#idx_d = pd.date_range('2020-02-17 00:00:00', '2020-08-01 12:00:00', freq='12H')
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
death_data_kids = np.zeros((int(len(time)/2),A-1,regions))
sero_data = np.zeros((152,A,regions))
regions_names = ["East_of_England",
                "London",
                "Midlands",
                "North_East_and_Yorkshire",
                "North_West",
                "South_East",
                "South_West"]
sero_denom_path_prefix = "20200717_"
death_path_prefix = "deaths20200731_"
sero_denom_path_postfix = "_8ag_samples"
sero_assay_path_postfix = "_8ag_positives"
death_path_postfix = "_8agCONF"
sero_denom = np.zeros((152,A,regions))
for r, region_name in enumerate(regions_names):
        sero_denom_path = sero_denom_path_prefix + \
                region_name + sero_denom_path_postfix + '.txt'
        sero_denom[:,:,r] = np.loadtxt('./model_data/SeroSanmitra/' + \
                sero_denom_path,usecols=(1,2,3,4,5,6,7,8)).astype(np.int64)
        sero_assay_path = sero_denom_path_prefix + \
                region_name + sero_assay_path_postfix + '.txt'
        sero_data[:,:,r] = np.loadtxt('./model_data/SeroSanmitra/' + \
                sero_assay_path,usecols=(1,2,3,4,5,6,7,8)).astype(np.int64) 
        death_path = death_path_prefix + \
                region_name + death_path_postfix+'.txt'     
        death_data[:,:,r] = np.loadtxt('./data/real_data/' + \
                death_path,usecols=(1,2,3,4,5,6,7,8)).astype(np.int64) 
death_data_kids[:,0] = death_data[:,0] + death_data[:,1]
death_data_kids[:,1:] = death_data[:,2:]
data = [death_data_kids, sero_data]   

burnin = -200000
thin = 200
param_filename = './results/_mcmc_test_ind_real_halfmil.p'
traces = pickle.load( open( param_filename , "rb" ) )
trace = np.array(traces)
trace_post_burn = trace[:,burnin:,:]
py_thinned_trace_amgs = trace_post_burn[:,::thin,:]
x_axis = pd.date_range('2020-02-17', '2020-07-31', freq='D')
for RUN_REGION in range(regions):
    forward_model = ForwardModelRegionWise(dt, time, A, N[:,RUN_REGION], daily_mix_matrix, \
                            breakpoints[6], breakpoints[5], list_brk_beta, idx_d)
    forward_model.real_data()        

    logP = LogPosteriorRegionWise(forward_model, RUN_REGION, data, sero_denom[:,:,RUN_REGION])
    
    #ts_d = pd.Series(range(len(idx_d)), index=idx_d)
    ppc = []
    rand_range = random.sample(range(0, 1000), 500)
    for i in rand_range:
            ppc.append(logP.sample(py_thinned_trace_amgs[RUN_REGION,i,:])[0])

    ppc_samples = np.array(ppc)
    ppc_mean = np.percentile(ppc_samples,50,axis=0)
    ppc_uq = np.percentile(ppc_samples,97.5,axis=0)
    ppc_lq = np.percentile(ppc_samples,2.5,axis=0)
    all_age_ppc_mean = np.sum(ppc_mean,axis=1)
    all_age_ppc_lq = np.sum(ppc_lq,axis=1)
    all_age_ppc_uq = np.sum(ppc_uq,axis=1)
    all_age_death_data = np.sum(death_data[:,:,RUN_REGION],axis=1)

    plt.plot(x_axis, all_age_ppc_lq,'--', color='blue')
    plt.plot(x_axis, all_age_ppc_uq,'--', color='blue')
    plt.plot(x_axis, all_age_death_data, 'o', color='green', label=regions_names[RUN_REGION])
    plt.plot(x_axis, all_age_ppc_mean, linewidth=2.5, color='blue', label='Model_mean_'+regions_names[RUN_REGION],)
    plt.legend()
    plt.ylabel('Daily_deaths')
    plt.savefig('./figures/ind/ppc_real/'+str(RUN_REGION)+'.png')
    plt.close()
    print('Done region :'+str(RUN_REGION))
    






