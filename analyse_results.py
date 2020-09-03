import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np
import pandas as pd
from logPDF import LogPosterior
from forwardModel import ForwardModel

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

burnin = 50000
thin = 10
trace = pickle.load( open( "./results/_mcmc_test1.p", "rb" ) )
trace_post_burn = trace[burnin:,:]
py_thinned_trace_amgs = trace_post_burn[::thin,:]
sns.set_context("paper", font_scale=1)
sns.set(rc={"figure.figsize":(8,6),"font.size":10,"axes.titlesize":20,"axes.labelsize":17,
           "xtick.labelsize":6, "ytick.labelsize":6},style="white")
param_names = [r"$\beta_{r0a1}$", r"$\beta_{r0a2}$", r"$\beta_{r0a3}$", r"$\beta_{r0a4}$", r"$\beta_{r0a5}$",
r"$\beta_{r0a6}$",r"$\beta_{r0a7}$"]
beta_rw_load = np.loadtxt('./model_data/log_beta_rw.txt')
T_lines = [-16.63574, -13.90437, -15.33340, -17.67026, -15.51458, -15.29906, -16.09737]
plot_trace = py_thinned_trace_amgs[:,11:18]
plot_full_trace = trace[:,11:18]

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
all_age_death_data = np.sum(death_data,axis=1)

plt.plot(idx_d, all_age_ppc_lq[:,-1],'--', color='blue')
plt.plot(idx_d, all_age_ppc_uq[:,-1],'--', color='blue')
plt.plot(idx_d, all_age_death_data[:,-1], 'o', color='green', label='daily_deaths_R7')
plt.plot(idx_d, all_age_ppc_mean[:,-1], linewidth=2.5, color='blue', label='Model_mean_R7',)
plt.legend()
plt.ylabel('Daily_deaths')
plt.show()
a=1
"""