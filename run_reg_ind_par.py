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
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool
from util.util import ParallelEvaluator
TEST_ENABLED = False

num_cores = 1#multiprocessing.cpu_count()-1

def mcmc_runner(RUN_REGION):
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
        death_data = np.zeros((int(len(time)/2),A-1,regions))
        sero_data = np.zeros((152,A,regions))
        for i in range(regions):
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
        sero_denom = np.zeros((152,A,regions))
        for r, region_name in enumerate(regions_names):
                path = path_prefix+region_name+path_postfix+'.txt'
                sero_denom[:,:,r] = np.loadtxt('./model_data/SeroSanmitra/'+path,usecols=(1,2,3,4,5,6,7,8)).astype(np.int64)

        #######################################################################################
        ########################      DEFINE REGIONWISE MODELS    #############################
        #######################################################################################
        
        forward_model = ForwardModelRegionWise(dt, time, A, N[:,RUN_REGION], daily_mix_matrix, \
                                breakpoints[6], breakpoints[5], list_brk_beta, idx_d)
                        
        #forward_model.test()
        logP = LogPosteriorRegionWise(forward_model, RUN_REGION, data, sero_denom[:,:,RUN_REGION])
        #######################################################################################
        ########################    DEFINE starting PARAMETERS    #############################
        #######################################################################################
        dI = [ 0.5033387]
        ksens = [0.7078755]
        kspec = [0.9650125]
        eta = [0.0672998]
        psir = np.array([0.281224110810985,0.246300679874443, 0.230259384150778, 
                0.307383663711624, 0.249492140587071, 0.224509782739688, 0.234528728809235])
        I0 = np.array([-16.7064769395683, -14.3327333035582,
                        -15.0542472007424, -17.785749594464, -15.8038617705659,
                        -15.3720269985272, -16.3667281951197])
        p_dead = [0.00000577188478860128,0.00000957867182705255, 
                        0.000045278018816958, 0.000323870211139221,
                        0.00471791669192503, 0.0316645720110774, 0.202480672513791]
        beta_rw_sd = [0.1510573]
        m_mul = np.array(np.split(np.loadtxt('./model_data/m_mul_starts.txt'),regions,axis=0)).reshape((regions,3)).T 
        beta_rw = np.loadtxt('./model_data/beta_starts.txt')
        beta_rw = np.array(np.split(beta_rw,regions, axis=0)).T 

        start = []
        start.append(dI)
        start.append(ksens)
        start.append(kspec)
        start.append(eta)
        #start.append(psir[RUN_REGION])
        start.append(psir[RUN_REGION])
        start.append(I0[RUN_REGION])
        start.append(p_dead)
        start.append(beta_rw_sd)
        start.append(m_mul[:,RUN_REGION])
        start.append(beta_rw[:,RUN_REGION])
        #######################################################################################
        #############    RUN AMGS regionwise independent MCMC SAMPLER    ######################
        #######################################################################################

        X0 = np.hstack(start)
        sampler = AdaptiveMetropolis(logP, mean_est=X0, cov_est=None, tune_interval = 1)
 
        MCMCsampler = MCMC(sampler, logP, X0, 500) 
        return MCMCsampler.run()       


traces = []
for i in range(7):
        traces.append(mcmc_runner(i))
param_filename = './results/_mcmc_test_allregions_par.p'
pickle.dump(traces, open(param_filename, 'wb'))
"""
traces = Parallel(n_jobs=num_cores)(delayed(mcmc_runner)(region) for region in range(7))


p = Pool(2)
REGIONS = list(np.arange(2))
traces = p.map( mcmc_runner, (RUN_REGION for RUN_REGION in REGIONS) )

n_workers = ParallelEvaluator.cpu_count()-6
evaluator_mcmc_runner = ParallelEvaluator(mcmc_runner, n_workers=1)
traces = evaluator_mcmc_runner.evaluate(REGIONS)

log_pp_obj = log_posterior_pointwise(problem, current, sigma)
    n_workers = pints.ParallelEvaluator.cpu_count()
    evaluator_log_pp = pints.ParallelEvaluator(log_pp_obj, n_workers=n_workers)
    log_pp = evaluator_log_pp.evaluate(samples)
    log_pp = np.asarray(log_pp)
"""
