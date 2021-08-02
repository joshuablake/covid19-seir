#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from logPDF import LogPosterior
from forwardModel import ForwardModel
from sub_block_adapt3_mcmc import subBlockAMGS3
import scipy.stats as stats
import pandas as pd
import time as timer

idx_w = pd.date_range('2020-02-17 00:00:00', '2020-08-01 12:00:00', freq='W-TUE')
idx_d = pd.date_range('2020-02-17 00:00:00', '2020-08-01 12:00:00', freq='12H')
ts_d = pd.Series(range(len(idx_d)), index=idx_d)
def breakpoint(dayseries, dayindex, weekindex):
  breaks_date = []
  breaks_points = []
  for i in range(len(weekindex)):
    breaks_date.append(dayindex[dayseries.loc[weekindex[i]]])
    breaks_points.append(dayseries.loc[weekindex[i]])
  breaks_points.append(dayseries.loc[dayindex[-1]])
  return tuple((breaks_date, breaks_points))

def daily_mixing_matrices(dt, dayindex, breakpoints_list, region):
  grid_space = int(1/dt)
  pre_lckdn_mat = np.loadtxt('./model_data/contact_mats/england_8ag_contact.txt')
  mixing_mat = np.repeat(pre_lckdn_mat[:, :, np.newaxis], breakpoints_list[0], axis=2)
  for i in range(1,19):
    last_loaded_mat = np.loadtxt('./model_data/contact_mats/england_8ag_contact_ldwk'+str(i)+'_20200731.txt')
    post_lckdn_mat = np.repeat(last_loaded_mat[:, :, np.newaxis], 7*grid_space, axis=2)  
    mixing_mat = np.concatenate((mixing_mat,post_lckdn_mat),axis=2)
  
  post_lckdn_mat = np.repeat(last_loaded_mat[:, :, np.newaxis], 
                    ((breakpoints_list[2]+1) - (breakpoints_list[1])), axis=2) 
  mixing_mat = np.concatenate((mixing_mat,post_lckdn_mat),axis=2)
  mixing_mat = np.repeat(mixing_mat[:,:,:,np.newaxis],region,axis=3)
  assert len(dayindex) == mixing_mat.shape[2]
  
  return mixing_mat

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

ts_d = pd.Series(range(len(idx_d)), index=idx_d)
breakpoints = np.array(breakpoint(ts_d, idx_d, idx_w)[1])
list_brk_beta = [breakpoints[5],breakpoints[-3], breakpoints[-1]]
list_brk_mix = [breakpoints[5],*breakpoints[-2:]]
daily_mix_matrix = daily_mixing_matrices(dt, idx_d, list_brk_mix, regions)

 
forward_model = ForwardModel(dt, time, N, daily_mix_matrix, \
                        breakpoints[6], breakpoints[5], list_brk_beta, idx_d)
forward_model.test()                        

TEST_ENABLED = False
death_data = np.zeros((167,7,7))
sero_data = np.zeros((152,8,7))
for i in range(7):
        death_data[:,:,i] = np.loadtxt('./data/deaths_r'+str(i)+'.txt')
        sero_data[:,:,i] = np.loadtxt('./data/sero_assay_r'+str(i)+'.txt')

data = [death_data, sero_data]
regions_names = ["East_of_England",
                 "London",
                 "Midlands",
                 "North_East_and_Yorkshire",
                 "North_West",
                 "South_East",
                 "South_West"]
path_prefix = "20200717_"
path_postfix = "_8ag_samples"
sero_denom = np.zeros((152,8,7))
for r, region_name in enumerate(regions_names):
  path = path_prefix+region_name+path_postfix+'.txt'
  sero_denom[:,:,r] = np.loadtxt('./model_data/SeroSanmitra/'+path,usecols=(1,2,3,4,5,6,7,8)).astype(np.int64)
 
logP = LogPosterior(forward_model, data, sero_denom)

dI = [ 0.5033387]
ksens = [0.7078755]
kspec = [0.9650125]
eta = [0.0672998]
psir = [0.281224110810985,0.246300679874443, 0.230259384150778, 
        0.307383663711624, 0.249492140587071, 0.224509782739688, 0.234528728809235]
I0 = np.array([-16.7064769395683, -14.3327333035582,
                -15.0542472007424, -17.785749594464, -15.8038617705659,
                -15.3720269985272, -16.3667281951197])
p_dead = [0.00000577188478860128,0.00000957867182705255, 
                0.000045278018816958, 0.000323870211139221,
                0.00471791669192503, 0.0316645720110774, 0.202480672513791]
m_mul = np.loadtxt('./model_data/m_mul_starts.txt')
beta_rw = np.loadtxt('./model_data/beta_starts.txt')
beta_rw_sd = [0.1510573]

start = []
start.append(dI)
start.append(ksens)
start.append(kspec)
start.append(eta)
start.append(psir)
start.append(I0)
start.append(p_dead)
start.append(beta_rw_sd)
start.append(m_mul)
start.append(beta_rw)

X0 = np.hstack(start)

t0 = timer.time()
MCMCsampler = subBlockAMGS3(logP, X0, X0, 100000)
trace = MCMCsampler.run()
t1 = timer.time()
total = t1-t0
print('Time for 100K 3 blocks is: ',total)
param_filename = './results/_mcmc_test.p'
pickle.dump(trace, open(param_filename, 'wb'))

