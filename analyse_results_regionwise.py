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
burnin = 1#50000
thin = 1#50
param_name_dict = give_param_names()
param_filename = './results/_mcmc_test_allregions_par.p'
traces = pickle.load( open( param_filename , "rb" ) )
trace = np.array(traces)
trace_post_burn = trace[:,burnin:,:]
py_thinned_trace_amgs = trace_post_burn[:,::thin,:]
sns.set_context("paper", font_scale=1)
sns.set(rc={"figure.figsize":(8,6),"font.size":10,"axes.titlesize":20,"axes.labelsize":17,
           "xtick.labelsize":6, "ytick.labelsize":6},style="white")

m_mul = np.array(np.split(np.loadtxt('./model_data/m_mul_starts.txt'),\
  regions,axis=0)).reshape((regions,3)).T
beta_rw = np.loadtxt('./model_data/log_beta_rw.txt')
beta_rw = np.array(np.split(beta_rw,regions, axis=0)).T 
T_lines = [1.149820e-05, 2.444837e-05, 5.711427e-05, 3.656723e-04,
                        4.959340e-03, 3.428846e-02, 2.210078e-01]
param_names = param_name_dict['p_dead']
ind = (np.arange(4,11))

plot_trace = py_thinned_trace_amgs[0,:,ind]
plot_full_trace = trace[0,:,ind]

for i, p in enumerate(param_names):
    
    # Add histogram subplot
    plt.subplot(7, 2, 1 + 2 * i)
    plt.ylabel('Frequency')
    sns.kdeplot(plot_trace[i, :], color='lightseagreen', legend=True)

    plt.axvline(T_lines[i], linewidth=2.5, color='black')
    if i==0:
            plt.legend()

    # Add trace subplot
    plt.subplot(7, 2, 2 + 2 * i)
    plt.ylabel(p, fontsize=20)  
    plt.plot(plot_full_trace[i, :], alpha=0.5, color='lightseagreen')
    plt.axhline(T_lines[i], linewidth=2.5, color='black')
plt.show()

"""
idx_d = pd.date_range('2020-02-17', '2020-08-01', freq='D')
ts_d = pd.Series(range(len(idx_d)), index=idx_d)
ppc = []
for i in range(py_thinned_trace_amgs.shape[0]):#py_thinned_trace_amgs.shape[0]
    ppc.append(logP.sample(py_thinned_trace_amgs[i,:])[0])

ppc_samples = np.array(ppc)
ppc_mean = np.percentile(ppc_samples,50,axis=0)
ppc_uq = np.percentile(ppc_samples,97.5,axis=0)
ppc_lq = np.percentile(ppc_samples,2.5,axis=0)
all_age_ppc_mean = np.sum(ppc_mean,axis=1)
all_age_ppc_lq = np.sum(ppc_lq,axis=1)
all_age_ppc_uq = np.sum(ppc_uq,axis=1)
all_age_death_data = np.sum(death_data[:,:,RUN_REGION],axis=1)

plt.plot(idx_d, all_age_ppc_lq,'--', color='blue')
plt.plot(idx_d, all_age_ppc_uq,'--', color='blue')
plt.plot(idx_d, all_age_death_data, 'o', color='green', label='daily_deaths_R7')
plt.plot(idx_d, all_age_ppc_mean, linewidth=2.5, color='blue', label='Model_mean_R7',)
plt.legend()
plt.ylabel('Daily_deaths')
plt.show()
a=1
"""