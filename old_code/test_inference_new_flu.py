#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from logPDF import LogPosterior
from forwardModel import ForwardModel
from background import BackgroundModel
from adaptiveMetropolis import AdaptiveMetropolis
from adaptiveMetropolisPCA import AdaptiveMetropolisPCA
from block_mcmc import MCMC
from sub_block_adapt3_mcmc import subBlockAMGS3
import scipy.stats as stats

data_samples_ind =  [1951, 3321, 3052, 5244, 741, 2966, 4639, 7311, 4062, 8374, 4753, 6458, 7207, 992, 9400, 8929, 1827, 6839, 8149, 1333]
task_id = os.getenv('SLURM_ARRAY_TASK_ID')
id_number = int(task_id) - 1

file_path = '/project/pandemic_flu/Real_Time_Modelling/Sanmitra/Initial_SimData/'
data_path = '/project/pandemic_flu/Real_Time_Modelling/Sanmitra/dummy_data_sanmitra/'

gp_Denom=np.loadtxt(os.path.join(file_path,'dummy_data/GP_denoms.txt'),usecols = (1,2,3,4,5,6))
viro_Denom=np.loadtxt(os.path.join(file_path,'dummy_data/viro_sam.txt'),usecols = (1,2,3,4,5,6,7))
sero_Denom=np.loadtxt(os.path.join(file_path,'dummy_data/sero_sam.txt'),usecols = (1,2,3,4,5,6,7))

gP_DATA = np.loadtxt(data_path + 'gp_data/gp_' + str(data_samples_ind[id_number]) + '.txt')
hOSP_DATA = np.loadtxt(data_path + 'hosp_data/hosp_' + str(data_samples_ind[id_number]) + '.txt')
sERO_DATA = np.loadtxt(data_path + 'sero_data/sero_' + str(data_samples_ind[id_number]) + '.txt')
vIRO_DATA = np.loadtxt(data_path + 'viro_data/viro_' + str(data_samples_ind[id_number]) + '.txt')

bG_prior=np.loadtxt(os.path.join(data_path,'pp.txt'))
DATA =[gP_DATA, hOSP_DATA, vIRO_DATA, sERO_DATA]
DENOMS = [gp_Denom, viro_Denom, sero_Denom]

dt = 0.5
time = np.arange(0,245,dt)
pop_size_N = np.array([119967, 410926, 840309, 989473, 2747554, 1566445, 882256])


polymod_file_path = '/project/pandemic_flu/RTM.code/polymod_matrices/polymod_non_holiday.txt'
forward_model = ForwardModel(dt, time, pop_size_N, polymod_file_path)


bg_design_matrix = np.loadtxt(os.path.join(file_path,'bg_design.txt'))
background_model = BackgroundModel(bg_design_matrix)
TEST_ENABLED = False
logP = LogPosterior(forward_model, background_model, DATA, DENOMS, pop_size_N, bG_prior, testing = False)

p_gp_param_Start = [ -2.032412, -1.777473, -1.192682, -2.204748, -1.702390, -1.449284, -0.550381, -0.532177]
p_cc_param_Start = [ 0.146859]
m_params_Start = [0.449422, 0.702529, 0.053619, 0.024701, 0.435492]
ode_params_Start = [1.486955, 0.401808, 0.125082, -15.29728]
eta_params_Start = [123.0212, 75.8902]
bg_param_Start = [7.217326, 7.188790, 7.238073, 6.863742, 7.471192, 9.228371, 7.935605, \
                                        6.600467, 6.479529, 6.676478, 6.512326, 6.783735, \
                                        7.004250, 7.060164, 7.285456, 7.244336, 6.810427, 0.699570,\
                                        0.726366, 0.063490, 0.002485, -0.396529,-0.866604, 1.507578, \
                                        0.870114, 0.011742, 0.364132, 0.135676, 0.386747]


p_gp_param_Scale = [0.007213, 0.007213, 0.025846, 0.015628, 0.010218, 0.010218, 0.014426, 0.014426]
p_cc_param_Scale = [0.074305]
m_params_Scale = [0.006741, 0.063928, 0.063928, 0.063928, 0.063928]
ode_params_Scale = [0.034102, 0.093939, 0.001183, 0.032967]
eta_params_Scale = [0.018322, 0.012261]
bg_param_Scale = [0.016742,0.013618,0.021679,0.011201,0.002730,0.000701,0.000600,0.010039,
                0.027766,0.014594,0.007135,0.005441,0.002202,0.002402,0.002229,0.002296,0.004049,0.004193,0.002238,0.003244,
                0.004677,0.002678, 0.012051, 0.000724, 0.001274, 0.002523, 0.003362,2.523e-03, 3.892e-03]

len(bg_param_Scale)
start, scale = [], []
start.append(p_gp_param_Start)
start.append(p_cc_param_Start)
start.append(m_params_Start)
start.append(ode_params_Start)
start.append(eta_params_Start)
start.append(bg_param_Start)

scale.append(p_gp_param_Scale)
scale.append(p_cc_param_Scale)
scale.append(m_params_Scale)
scale.append(ode_params_Scale)
scale.append(eta_params_Scale)
scale.append(bg_param_Scale)

X0 = np.hstack(start)
proposal_cov = np.eye(len(X0))*np.hstack(scale)

ADAPT_FULL = True
COMPONENTWISE = False
if ADAPT_FULL:
        if COMPONENTWISE:
                mean = None
                cov = None
        else:
                mean = X0
                cov = proposal_cov

"""
sampler = AdaptiveMetropolis(logP, mean_est=mean, cov_est=cov, tune_interval = 1)
MCMCsampler = MCMC(sampler, logP, X0, 400000)
trace = MCMCsampler.run()
 """

"""
sampler = AdaptiveMetropolisPCA(logP, num_eigen=9, mean_est=mean, cov_est=cov, tune_interval = 1)
MCMCsampler = MCMC(sampler, logP, X0, 300000)
trace = MCMCsampler.run()
"""

MCMCsampler = subBlockAMGS3(logP, X0, X0, 300000)
trace = MCMCsampler.run()

param_filename = './results/AMGS_SUB3/' + str(data_samples_ind[id_number]) + '_mcmc_test.p'
pickle.dump(trace, open(param_filename, 'wb'))

