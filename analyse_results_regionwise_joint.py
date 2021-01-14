import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np
import pandas as pd
from models.logPDF_Cpp import LogPosteriorRegionWise
from models.forwardModel_Cpp import ForwardModelRegionWise
from contact_m.generate_contact_matrices import breakpoint, daily_mixing_matrices
from util.param_dict import give_param_names
def plot_function(_trace, _burnin, _thin, _T_lines, _param_names, _ind, fig_name, global_ind=None):
  trace_post_burn = _trace[_burnin:,:]
  py_thinned_trace_amgs = trace_post_burn[::_thin,:]

  plot_trace = py_thinned_trace_amgs[:,_ind]
  plot_full_trace = _trace[:,_ind]
  sns.set_context("paper", font_scale=1)
  sns.set(rc={"figure.figsize":(8,6),"font.size":10,"axes.titlesize":20,"axes.labelsize":7,
           "xtick.labelsize":6, "ytick.labelsize":6},style="white")
  for i, p in enumerate(_param_names):    
      # Add histogram subplot
      plt.subplot(7, 2, 1 + 2 * i)
      plt.ylabel('Frequency')
      sns.kdeplot(plot_trace[:, i], color='lightseagreen', legend=True)

      plt.axvline(_T_lines[i], linewidth=2.5, color='red')
      if i==0:
              plt.legend()

      # Add trace subplot
      plt.subplot(7, 2, 2 + 2 * i)
      plt.ylabel(p, fontsize=20) 
      
      plt.plot(plot_full_trace[:, i], alpha=0.5, color='lightseagreen')
      plt.axhline(_T_lines[i], linewidth=2.5, color='red')
  plt.subplots_adjust(wspace = 0.35, hspace=0.5)
  plt.savefig('./figures/joint/real/'+fig_name+'_sde.png')
  plt.close()

regions = 7
burnin = 100000
thin = 100#500
param_name_dict = give_param_names()
param_filename = './results/_mcmc_test_joint_real_sde_200k.p'
trace = pickle.load( open( param_filename , "rb" ) )

Paul_chain = pd.read_csv("./results/chain_sanmitra.csv")
p_dI = np.mean(Paul_chain['infectious_period1'].as_matrix()[:,None],axis=0)
p_ksens = np.mean(Paul_chain['sero_test_sensitivity1'].as_matrix()[:,None],axis=0)
p_kspec = np.mean(Paul_chain['sero_test_specificity1'].as_matrix()[:,None],axis=0)
p_eta = np.mean(Paul_chain['hosp_negbin_overdispersion1'].as_matrix()[:,None],axis=0)
p_p_dead = Paul_chain.as_matrix(columns=Paul_chain.columns[37:37+7]).mean(axis=0)
p_beta_rw_sd = np.mean(Paul_chain['log_beta_rw_sd2'].as_matrix()[:,None],axis=0)
p_m_mul = Paul_chain.as_matrix(columns=Paul_chain.columns[2:2+21]).mean(axis=0)
p_beta_rw = Paul_chain.as_matrix(columns=Paul_chain.columns[46:46+112]).mean(axis=0)
p_psir = Paul_chain.as_matrix(columns=Paul_chain.columns[23:23+7]).mean(axis=0)
p_I0 = Paul_chain.as_matrix(columns=Paul_chain.columns[30:30+7]).mean(axis=0)

#######################################################################################
####################      PLOT REGIONALS    #############################################
#######################################################################################

#### plot m_muls ###################
m_mul = np.array(np.split(p_m_mul,\
  regions,axis=0)).reshape((regions,3)).T
for i in range(3):
  T_lines =  m_mul[i,:]  
  param_names = param_name_dict['m_mul'+str(i+1)][:-1]
  ind = ([param_name_dict['m_mul'+str(i+1)][-1] + (r*21) for r in range(regions)])
  fig_name = 'm_mul'+str(i+1)
  plot_function(trace, burnin, thin, T_lines, param_names, ind, fig_name)

#### plot betas ###################
beta_rw = p_beta_rw
beta_rw = np.array(np.split(beta_rw,regions, axis=0)).T 
for i in range(5):
  T_lines =  beta_rw[i,:] 
  param_names = param_name_dict['beta'+str(i+1)][:-1]
  ind = ([param_name_dict['beta'+str(i+1)][-1] + (r*21) for r in range(regions)])
  fig_name = 'beta'+str(i+1)
  plot_function(trace, burnin, thin, T_lines, param_names, ind, fig_name)

#### plot exponential growth ###################
T_lines =  p_psir
param_names = param_name_dict['psir'][:-1]
ind = ([param_name_dict['psir'][-1] + (r*21) for r in range(regions)])
fig_name = 'psir'
plot_function(trace, burnin, thin, T_lines, param_names, ind, fig_name)

#### plot Initial Infectives ###################
T_lines =  p_I0
param_names = param_name_dict['I0'][:-1]
ind = ([param_name_dict['I0'][-1] + (r*21) for r in range(regions)])
fig_name = 'I0'
plot_function(trace, burnin, thin, T_lines, param_names, ind, fig_name)

#######################################################################################
####################      PLOT GLOBALS    #############################################
#######################################################################################

#### plot IFRs ###################
T_lines =  p_p_dead
 
param_names = param_name_dict['p_dead'][:-1]
ind = param_name_dict['p_dead'][-1]
fig_name = 'p_dead'
plot_function(trace, burnin, thin, T_lines, param_names, ind, fig_name)

#### plot dI, Ksens, Kspec, eta, beta_sd ###################
T_lines =[ p_dI, p_ksens, p_kspec, p_eta, p_beta_rw_sd]#0.07708538 for real

param_names = param_name_dict['global_scalars'][:-1]
ind = param_name_dict['global_scalars'][-1]
fig_name = 'global_scalars'
plot_function(trace, burnin, thin, T_lines, param_names, ind, fig_name)






