#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from models.logPDF_Cpph2 import LogPosteriorRegionWise
from models.forwardModel_Cpph2 import ForwardModelRegionWise
from mcmc.adaptiveMetropolis import AdaptiveMetropolis
from mcmc.sub_block_adapt_mcmc import SubBlockRegionalAMGS
from contact_m.generate_contact_matrices import breakpoint, daily_mixing_matrices
import scipy.stats as stats
import pandas as pd
import time as timer
TEST_ENABLED = False
#######################################################################################
####################      DEFINE CONSTANTS     ########################################
#######################################################################################
idx_w = pd.date_range('2020-05-12 00:00:00', '2020-08-01 12:00:00', freq='W-TUE')
idx_d = pd.date_range('2020-05-12 00:00:00', '2020-08-01 12:00:00', freq='12H')
ts_d = pd.Series(range(len(idx_d)), index=idx_d)
dt = 0.5
end_time = 82
time = np.arange(0,end_time,dt)
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
a= breakpoint(ts_d, idx_d, idx_w)#make this spit out the breakpoints when passed in the dates
breakpoints = np.array(breakpoint(ts_d, idx_d, idx_w)[1])
list_brk_beta = [breakpoints[5],breakpoints[-3], breakpoints[-1]]
list_brk_mix = [breakpoints[5],*breakpoints[-2:]]
daily_mix_matrix = daily_mixing_matrices(dt, idx_d, list_brk_mix, regions, \
        startpoint=8, add_prelckdn=False)
#######################################################################################
####################      DATA LOADING         ########################################
#######################################################################################
death_data = np.zeros((int(len(time)/2),A,regions))
days_to_load = range(85,167)
days_to_load_sero = range(85,152)
sero_data = np.zeros((152-85,A,regions))

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
sero_denom = np.zeros((152-85,A,regions))
for r, region_name in enumerate(regions_names):
        sero_denom_path = sero_denom_path_prefix + \
                region_name + sero_denom_path_postfix + '.txt'
        sero_denom[:,:,r] = np.loadtxt('./model_data/SeroSanmitra/' + \
                sero_denom_path,usecols=(1,2,3,4,5,6,7,8))[days_to_load_sero,:]
        sero_assay_path = sero_denom_path_prefix + \
                region_name + sero_assay_path_postfix + '.txt'
        sero_data[:,:,r] = np.loadtxt('./model_data/SeroSanmitra/' + \
                sero_assay_path,usecols=(1,2,3,4,5,6,7,8))[days_to_load_sero,:]
        death_path = death_path_prefix + \
                region_name + death_path_postfix+'.txt'     
        death_data[:,:,r] = np.loadtxt('./data/real_data/' + \
                death_path,usecols=(1,2,3,4,5,6,7,8))[days_to_load,:] 
death_data_int = death_data.astype(int)
sero_data_int = sero_data.astype(int)
data = [death_data_int, sero_data_int]

#######################################################################################
########################      DEFINE REGIONWISE MODELS    #############################
#######################################################################################
param_filename = './results/last_states.p'
last_states = pickle.load( open( param_filename , "rb" ) )
regional_logPs = []
for reg_model in range(regions):
        forward_model = ForwardModelRegionWise(last_states[reg_model], dt, time, A, N[:,reg_model], daily_mix_matrix, \
                list_brk_beta, idx_d)
        forward_model.real_data()
        regional_logPs.append(LogPosteriorRegionWise(forward_model, reg_model, \
                data, sero_denom[:,:,reg_model]))
#######################################################################################
########################    DEFINE starting PARAMETERS    #############################
#######################################################################################
burnin = -100000
thin = 100
param_filename = './results/_mcmc_test_joint_real_200kh1.p'
trace = pickle.load( open( param_filename , "rb" ) )
trace_post_burn = trace[burnin:,:]
py_thinned_trace_amgs = trace_post_burn[::thin,:]
start_pars=py_thinned_trace_amgs.mean(axis=0)
beta_rw = np.loadtxt('./model_data/beta_starts.txt')
beta_rw = np.array(np.split(beta_rw,regions, axis=0)).T 

regional_size = 12
global_size = 12
region_starts = [global_size + (i*regional_size) for i in range(regions)]
print(region_starts)
start = []
start.append(np.log(start_pars[0]))
start.append(np.log(start_pars[1]))
start.append(np.log(start_pars[2]))
start.append(np.log(start_pars[3]))
start.append(np.log(start_pars[4:11]))
start.append(np.log(start_pars[11]))
for region in range(regions):
        print(start_pars[15 + (region*11):(15 + (region*11))+2])
        print(start_pars[region_starts[region]+3:region_starts[region]+5])
        start.append(start_pars[region_starts[region]+3:region_starts[region]+5])
        start.append(beta_rw[7:17,region])

X0 = np.hstack(start)
#######################################################################################
#############    RUN AMGS regionwise independent MCMC SAMPLER    ######################
#######################################################################################
t0 = timer.time()
MCMCsampler = SubBlockRegionalAMGS(regional_logPs, X0, X0, 400000)
trace = MCMCsampler.run()       
t1 = timer.time()
total = t1-t0
print('Time for SubBlock all regions AMGS is: ',total)
param_filename = './results/_mcmc_test_joint_real_400kh2_betafxd.p'
pickle.dump(trace, open(param_filename, 'wb'))

