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
#######################################################################################
####################      DATA LOADING         ########################################
#######################################################################################
dt = 0.5
time = np.arange(0,167,dt)
regions = int(7)
A = int(8)
T = int(len(time))
time = np.arange(0,167,dt)
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
death_all_age = np.sum(death_data_int,axis=1)
#######################################################################################
############## Create Date-time and generate Mixing Matrices    #######################
#######################################################################################
idx_w = pd.date_range('2020-02-17 00:00:00', '2020-08-01 12:00:00', freq='W-TUE')
idx_d = pd.date_range('2020-02-17 00:00:00', '2020-08-01 12:00:00', freq='12H')
ts_d = pd.Series(range(len(idx_d)), index=idx_d)
#x_axis = pd.date_range('2020-02-17', '2020-07-31', freq='D')
x_axis=idx_d
########### Load Paul's NNI ########################
param_filename = './results/ppc/paul_nni_mean.p'
paul_nni_mean = pickle.load( open( param_filename , "rb" ) )
param_filename = './results/ppc/paul_nni_lq.p'
paul_nni_lq = pickle.load( open( param_filename , "rb" ) )
param_filename = './results/ppc/paul_nni_uq.p'
paul_nni_uq = pickle.load( open( param_filename , "rb" ) )
########### Load Full NNI ########################
param_filename = './results/ppc/full_nni_mean.p'
full_nni_mean = pickle.load( open( param_filename , "rb" ) )
param_filename = './results/ppc/full_nni_lq.p'
full_nni_lq = pickle.load( open( param_filename , "rb" ) )
param_filename = './results/ppc/full_nni_uq.p'
full_nni_uq = pickle.load( open( param_filename , "rb" ) )
########### Load OU NNI ########################
param_filename = './results/ppc/ou_nni_mean.p'
ou_nni_mean = pickle.load( open( param_filename , "rb" ) )
param_filename = './results/ppc/ou_nni_lq.p'
ou_nni_lq = pickle.load( open( param_filename , "rb" ) )
param_filename = './results/ppc/ou_nni_uq.p'
ou_nni_uq = pickle.load( open( param_filename , "rb" ) )
########### Load h1 NNI ########################
param_filename = './results/ppc/h1_nni_mean.p'
h1_nni_mean = pickle.load( open( param_filename , "rb" ) )
param_filename = './results/ppc/h1_nni_lq.p'
h1_nni_lq = pickle.load( open( param_filename , "rb" ) )
param_filename = './results/ppc/h1_nni_uq.p'
h1_nni_uq = pickle.load( open( param_filename , "rb" ) )
########### Load h2 NNI ########################
param_filename = './results/ppc/h2Tx_nni_mean.p'
h2_nni_mean = pickle.load( open( param_filename , "rb" ) )
param_filename = './results/ppc/h2Tx_nni_lq.p'
h2_nni_lq = pickle.load( open( param_filename , "rb" ) )
param_filename = './results/ppc/h2Tx_nni_uq.p'
h2_nni_uq = pickle.load( open( param_filename , "rb" ) )

##### Plot age 0 #############################
regions_names = ["East_of_England",
                 "London",
                 "Midlands",
                 "North_East_and_Yorkshire",
                 "North_West",
                 "South_East",
                 "South_West"]
RUN_REGION = 0
h1h2_nni_mean = np.hstack((h1_nni_mean[RUN_REGION],h2_nni_mean[RUN_REGION]))
h1h2_nni_lq = np.hstack((h1_nni_lq[RUN_REGION],h2_nni_lq[RUN_REGION]))
h1h2_nni_uq = np.hstack((h1_nni_uq[RUN_REGION],h2_nni_uq[RUN_REGION]))

"""
x_axis1 = pd.date_range('2020-02-17', '2020-08-01', freq='D')

plt.figure(figsize=(15, 7.5))
plt.plot(x_axis1, full_nni_lq[RUN_REGION],'--', color='red')
plt.plot(x_axis1, full_nni_uq[RUN_REGION],'--', color='red')
plt.plot(x_axis1, ou_nni_lq[RUN_REGION],'--', color='orange')
plt.plot(x_axis1, ou_nni_uq[RUN_REGION],'--', color='orange')
#plt.plot(x_axis, paul_nni_mean[RUN_REGION], linewidth=5, \
#        color='blue', alpha=0.5, label='Full_RwMCMC_NNI_median')
plt.plot(x_axis1, death_all_age[:,RUN_REGION],'o', \
        color='green', label=regions_names[RUN_REGION])

plt.plot(x_axis1, full_nni_mean[RUN_REGION],linewidth=3, \
        color='red', label='Full_AMGS_Death_median')

plt.plot(x_axis1, ou_nni_mean[RUN_REGION],linewidth=3, \
       color='orange', label='Full_OU_AMGS_Death_median')

plt.legend()
plt.ylabel('Deaths all age summed')
plt.title('Posterior medians of NNIs '+regions_names[RUN_REGION])
plt.savefig('./figures/joint/ppc_real/report/'+str(RUN_REGION)+'GoF.png')
plt.close()
print('Done region :'+str(RUN_REGION))
"""
"""
times = np.arange(0,131,0.5)
plt.figure(figsize=(15, 7.5))
plt.plot(times, ou_nni_lq[RUN_REGION],'--', color='orange')
plt.plot(times, ou_nni_uq[RUN_REGION],'--', color='orange')
plt.plot(times, ou_nni_mean[RUN_REGION], color='orange')
#plt.plot(np.arange(0,17,1), full_nni_lq[RUN_REGION],'--', color='blue')
#plt.plot(np.arange(0,17,1), full_nni_uq[RUN_REGION],'--', color='blue')
#plt.plot(np.arange(0,17,1), full_nni_mean[RUN_REGION], color='blue')
plt.savefig('./figures/joint/ppc_real/report/'+str(RUN_REGION)+'_Betas.png')
plt.close()
print('Done region :'+str(RUN_REGION))
"""


plt.figure(figsize=(15, 7.5))
plt.plot(x_axis, full_nni_lq[RUN_REGION],'--', color='red')
plt.plot(x_axis, full_nni_uq[RUN_REGION],'--', color='red')
plt.plot(x_axis, ou_nni_lq[RUN_REGION],'--', color='orange')
plt.plot(x_axis, ou_nni_uq[RUN_REGION],'--', color='orange')

plt.plot(x_axis, full_nni_mean[RUN_REGION],linewidth=3, \
        color='red', label='Full_AMGS_NNI_median')

plt.plot(x_axis, ou_nni_mean[RUN_REGION],linewidth=3, \
       color='orange', label='Full_OU_AMGS_NNI_median')

plt.legend()
plt.ylabel('NNIs all age summed')
plt.title('Posterior medians of NNIs '+regions_names[RUN_REGION])
plt.savefig('./figures/joint/ppc_real/report/'+str(RUN_REGION)+'_NNI.png')
plt.close()
print('Done region :'+str(RUN_REGION))





