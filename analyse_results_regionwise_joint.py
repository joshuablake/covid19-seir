import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np
import pandas as pd
from models.logPDF_Cpp import LogPosteriorRegionWise
from models.forwardModel_Cpp import ForwardModelRegionWise
from contact_m.generate_contact_matrices import breakpoint, daily_mixing_matrices
from util.param_dict import give_param_names
regions = 7
burnin = -100000
thin = 100
param_name_dict = give_param_names()
param_filename = './results/_mcmc_test_allregions.p'
traces = pickle.load( open( param_filename , "rb" ) )
trace = np.array(traces)
trace_post_burn = trace[burnin:,:]
py_thinned_trace_amgs = trace_post_burn[::thin,:]
sns.set_context("paper", font_scale=1)
sns.set(rc={"figure.figsize":(8,6),"font.size":10,"axes.titlesize":20,"axes.labelsize":17,
           "xtick.labelsize":6, "ytick.labelsize":6},style="white")

m_mul = np.array(np.split(np.loadtxt('./model_data/contact_parameters.txt'),\
  regions,axis=0)).reshape((regions,3)).T
beta_rw = np.loadtxt('./model_data/log_beta_rw.txt')
beta_rw = np.array(np.split(beta_rw,regions, axis=0)).T 
#T_lines = [0.472845081832237,0.6948124,0.9682104,0.07708538, 0.1288072]
#T_lines = [1.149820e-05, 2.444837e-05, 5.711427e-05, 3.656723e-04,4.959340e-03, 3.428846e-02, 2.210078e-01] 
T_lines = m_mul[1, :]
loop_iterator = range(15, ((22*7) +12),22)
region_starts = np.array([i for i in loop_iterator])
param_names = param_name_dict['m_mul2'][:-1]
ind = (region_starts)#(np.arange(6,13))#param_name_dict['global_scalars'][-1]#

plot_trace = py_thinned_trace_amgs[:,ind]
plot_full_trace = trace[:,ind]

for i, p in enumerate(param_names):
    # Add histogram subplot
    plt.subplot(7, 2, 1 + 2 * i)
    plt.ylabel('Frequency')
    sns.kdeplot(plot_trace[:, i], color='lightseagreen', legend=True)

    plt.axvline(T_lines[i], linewidth=2.5, color='black')
    if i==0:
            plt.legend()

    # Add trace subplot
    plt.subplot(7, 2, 2 + 2 * i)
    plt.ylabel(p, fontsize=20)  
    plt.plot(plot_full_trace[:, i], alpha=0.5, color='lightseagreen')
    plt.axhline(T_lines[i], linewidth=2.5, color='black')
plt.subplots_adjust(hspace=0.5)
plt.savefig('./figures/scalars_joint.png')

