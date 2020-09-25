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
  trace_post_burn = _trace[:,_burnin:,:]
  py_thinned_trace_amgs = trace_post_burn[:,::_thin,:]
  if global_ind is not None:
    plot_trace = py_thinned_trace_amgs[global_ind,:,_ind]#.mean(axis=0).T
    plot_full_trace = trace[global_ind,:,_ind]#.mean(axis=0).T
  else:
    plot_trace = py_thinned_trace_amgs[:,:,_ind]#.mean(axis=0).T
    plot_full_trace = trace[:,:,_ind]#.mean(axis=0).T    
  sns.set_context("paper", font_scale=1)
  sns.set(rc={"figure.figsize":(8,6),"font.size":10,"axes.titlesize":20,"axes.labelsize":7,
           "xtick.labelsize":6, "ytick.labelsize":6},style="white")
  for i, p in enumerate(_param_names):    
      # Add histogram subplot
      plt.subplot(7, 2, 1 + 2 * i)
      plt.ylabel('Frequency')
      sns.kdeplot(plot_trace[i, :], color='lightseagreen', legend=True)

      plt.axvline(_T_lines[i], linewidth=2.5, color='black')
      if i==0:
              plt.legend()

      # Add trace subplot
      plt.subplot(7, 2, 2 + 2 * i)
      plt.ylabel(p, fontsize=20)  
      plt.plot(plot_full_trace[i, :], alpha=0.5, color='lightseagreen')
      plt.axhline(_T_lines[i], linewidth=2.5, color='black')
  plt.subplots_adjust(wspace = 0.35, hspace=0.5)
  plt.savefig('./figures/ind/sim/'+fig_name+'.png')
  plt.close()

regions = 7
burnin = -200000
thin = 200
param_name_dict = give_param_names(True)
param_filename = './results/_mcmc_test_ind_halfmil.p'
traces = pickle.load( open( param_filename , "rb" ) )
trace = np.array(traces)
#######################################################################################
####################      PLOT REGIONALS    #############################################
#######################################################################################

#### plot m_muls ###################
m_mul = np.array(np.split(np.loadtxt('./model_data/contact_parameters.txt'),\
  regions,axis=0)).reshape((regions,3)).T
for i in range(3):
  T_lines =  m_mul[i,:]  
  param_names = param_name_dict['m_mul'+str(i+1)][:-1]
  ind = param_name_dict['m_mul'+str(i+1)][-1]
  fig_name = 'm_mul'+str(i+1)
  plot_function(trace, burnin, thin, T_lines, param_names, ind, fig_name)

#### plot betas ###################
beta_rw = np.loadtxt('./model_data/log_beta_rw.txt')
beta_rw = np.array(np.split(beta_rw,regions, axis=0)).T 
for i in range(16):
  T_lines =  beta_rw[i,:]  
  param_names = param_name_dict['beta'+str(i+1)][:-1]
  ind = param_name_dict['beta'+str(i+1)][-1]
  fig_name = 'beta'+str(i+1)
  plot_function(trace, burnin, thin, T_lines, param_names, ind, fig_name)

#### plot exponential growth ###################
T_lines =  [0.2759202, 0.2319129, 0.2385402, 0.2977007, 0.2418800, 0.2210527, 0.2229998]
param_names = param_name_dict['psir'][:-1]
ind = param_name_dict['psir'][-1]
fig_name = 'psir'
plot_function(trace, burnin, thin, T_lines, param_names, ind, fig_name)

#### plot Initial Infectives ###################
T_lines =  [-16.63574, -13.90437, -15.33340, -17.67026, -15.51458, -15.29906, -16.09737]
param_names = param_name_dict['I0'][:-1]
ind = param_name_dict['I0'][-1]
fig_name = 'I0'
plot_function(trace, burnin, thin, T_lines, param_names, ind, fig_name)

#######################################################################################
####################      PLOT GLOBALS    #############################################
#######################################################################################

#### plot IFRs ###################
T_lines =  [1.149820e-05, 2.444837e-05, 5.711427e-05, 3.656723e-04,
                        4.959340e-03, 3.428846e-02, 2.210078e-01]
for i in range(regions):  
  param_names = param_name_dict['p_dead'][:-1]
  ind = param_name_dict['p_dead'][-1]
  fig_name = 'p_dead'+str(i+1)
  plot_function(trace, burnin, thin, T_lines, param_names, ind, fig_name, i)

#### plot dI, Ksens, Kspec, eta, beta_sd ###################
T_lines =[0.472845081832237, 0.6948124, 0.9682104, 0.07708538, 0.1288072]
for i in range(regions):  
  param_names = param_name_dict['global_scalars'][:-1]
  ind = param_name_dict['global_scalars'][-1]
  fig_name = 'global_scalars'+str(i+1)
  plot_function(trace, burnin, thin, T_lines, param_names, ind, fig_name, i)






