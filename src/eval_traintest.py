#!/usr/bin/python

# Imports
import sys, os, re, time
import argparse
import pdb
import pickle
from itertools import *
# Science
import numpy as np
import scipy.stats as stats
import pandas as pd
# Plotting
import matplotlib.pyplot as plt
from matplotlib import colors

# Import mixture models
from MixtureModel_multipleDataSources import *

########## Results to evaluate ##########
q_ids_name='q_ids_njdm'
result_dir='../results'

# For each of the run models
Ks=np.array([2, 3, 4, 5])
alphas=np.array([0.1,0.01,0.001])
betas=np.array([0.1,0.01,0.001])
R=10
# Train-test split variables
traintest_ratio=0.8
traintest_split='balanced'
split_n_obs=40

# main_dir
main_dir='{}/{}/traintest_with_MixtureModel_multipleDataSources/online/{}_{}_{}/all/R_{}'.format(result_dir,q_ids_name,traintest_ratio,traintest_split,split_n_obs,R)

# Preallocate space
all_loglik_train={'mean':np.nan*np.ones(0),'std':np.nan*np.ones(0)}
all_loglik_train_allinone={'mean':np.nan*np.ones(0),'std':np.nan*np.ones(0)}
all_dataloglik_test={'mean':np.nan*np.ones(0),'std':np.nan*np.ones(0)}
all_dataloglik_test_allinone={'mean':np.nan*np.ones(0),'std':np.nan*np.ones(0)}
all_cases=[]

for K in Ks:
    for alpha in alphas:
        for beta in betas:
            all_cases += [r'$K={},\alpha={},\beta={}$'.format(K,alpha,beta)]
            
            # Per-question data source
            inf_loglik_train=np.nan*np.ones(R)
            dataloglik_test=np.nan*np.ones(R)
            for r in np.arange(R):
                this_dir='{}/r_{}/K_{}/alpha_{}/beta_{}'.format(main_dir,r,K,alpha,beta)
                if os.path.exists(this_dir+'/train/inf_loglik_train.pickle'):
                    with open(this_dir+'/train/inf_loglik_train.pickle', 'rb') as f:
                        inf_loglik_train[r]=pickle.load(f)

                        if os.path.exists(this_dir+'/test/inf_dataloglik_test.pickle'):
                            with open(this_dir+'/test/inf_dataloglik_test.pickle', 'rb') as f:
                                inf_dataloglik_test=pickle.load(f)
                                dataloglik_test[r]=inf_dataloglik_test.sum()
                        
            # All in one data source
            inf_loglik_train_allinone=np.nan*np.ones(R)
            dataloglik_test_allinone=np.nan*np.ones(R)
            for r in np.arange(R):
                this_dir='{}/r_{}/K_{}/alpha_{}/beta_{}'.format(main_dir,r,K,alpha,beta)
                if os.path.exists(this_dir+'/train/allinone/inf_loglik_train.pickle'):
                    with open(this_dir+'/train/allinone/inf_loglik_train.pickle', 'rb') as f:
                        inf_loglik_train_allinone[r]=pickle.load(f)

                        if os.path.exists(this_dir+'/test/allinone/inf_dataloglik_test.pickle'):
                            with open(this_dir+'/test/allinone/inf_dataloglik_test.pickle', 'rb') as f:
                                inf_dataloglik_test_allinone=pickle.load(f)
                                dataloglik_test_allinone[r]=inf_dataloglik_test_allinone.sum()

            # Collect per-question
            all_loglik_train['mean']=np.append(all_loglik_train['mean'],np.nanmean(inf_loglik_train))
            all_loglik_train['std']=np.append(all_loglik_train['std'],np.nanstd(inf_loglik_train))
            all_dataloglik_test['mean']=np.append(all_dataloglik_test['mean'],np.nanmean(dataloglik_test))
            all_dataloglik_test['std']=np.append(all_dataloglik_test['std'],np.nanstd(dataloglik_test))

            # Collect allinone
            all_loglik_train_allinone['mean']=np.append(all_loglik_train_allinone['mean'],np.nanmean(inf_loglik_train_allinone))
            all_loglik_train_allinone['std']=np.append(all_loglik_train_allinone['std'],np.nanstd(inf_loglik_train_allinone))
            all_dataloglik_test_allinone['mean']=np.append(all_dataloglik_test_allinone['mean'],np.nanmean(dataloglik_test_allinone))
            all_dataloglik_test_allinone['std']=np.append(all_dataloglik_test_allinone['std'],np.nanstd(dataloglik_test_allinone))

            # Plot per parameterization
            plt.plot(np.arange(R), inf_loglik_train, 'b', label='$\log p(X_{train},Z_{train}|XD_{train})$')
            plt.plot(np.arange(R), inf_loglik_train_allinone, 'b:', label='$\log p(X_{train},Z_{train}|XD_{train})$ allinone')
            plt.plot(np.arange(R), dataloglik_test, 'r', label='$\log p(X_{test}|X_{train},XD)$')
            plt.plot(np.arange(R), dataloglik_test_allinone, 'r:', label='$\log p(X_{test}|X_{train},XD)$ allinone')
            plt.xlabel('r')
            plt.ylabel('$\log p()$')
            plt.xlim([0,R-1])
            legend = plt.legend(loc='upper right', ncol=1, shadow=False)
            plt.savefig('{}/inf_logliks_K_{}_alpha_{}_beta_{}.pdf'.format(main_dir,K,alpha,beta), format='pdf', bbox_inches='tight')
            plt.close()

# Plot all
plt.errorbar(np.arange(all_loglik_train['mean'].size), all_loglik_train['mean'], yerr=all_loglik_train['std'], color='b', fmt='o', label='$\log p(X_{train},Z_{train}|XD_{train})$')
plt.errorbar(np.arange(all_loglik_train_allinone['mean'].size), all_loglik_train_allinone['mean'], yerr=all_loglik_train_allinone['std'], color='b', fmt='^', label='$\log p(X_{train},Z_{train}|XD_{train})$ allinone')
plt.errorbar(np.arange(all_dataloglik_test['mean'].size), all_dataloglik_test['mean'], yerr=all_dataloglik_test['std'], color='r', fmt='x', label='$\log p(X_{test}|X_{train},XD)$')
plt.errorbar(np.arange(all_dataloglik_test_allinone['mean'].size), all_dataloglik_test_allinone['mean'], yerr=all_dataloglik_test_allinone['std'], color='r', fmt='^', label='$\log p(X_{test}|X_{train},XD) allinone$')
plt.xticks(np.arange(all_loglik_train['mean'].size), all_cases, rotation='vertical')
plt.ylabel('$\log p()$')
legend = plt.legend(loc='upper right', ncol=1, shadow=False)
plt.savefig('{}/train_test_logliks.pdf'.format(main_dir), format='pdf', bbox_inches='tight')
plt.close()
# Plot test-set loglik
plt.errorbar(np.arange(all_dataloglik_test['mean'].size), all_dataloglik_test['mean'], yerr=all_dataloglik_test['std'], color='r', fmt='x')#, label='$\log p(X_{test}|X_{train},XD)$')
plt.errorbar(np.arange(all_dataloglik_test_allinone['mean'].size), all_dataloglik_test_allinone['mean'], yerr=all_dataloglik_test_allinone['std'], color='b', fmt='o')#, label='$\log p(X_{test}|X_{train},XD)$ allinone')
plt.xticks(np.arange(all_loglik_train['mean'].size), all_cases, rotation='vertical')
plt.ylabel('$\log p()$')
#legend = plt.legend(loc='upper right', ncol=1, shadow=False)
plt.savefig('{}/test_logliks.pdf'.format(main_dir), format='pdf', bbox_inches='tight')
plt.close()

print('Test Loglikelihoods')
for a_model in np.arange(all_dataloglik_test['mean'].size):
    print('{0} & ${1:.2f}(\pm{2:.2f})$ & ${3:.2f}(\pm{4:.2f})$ \\\\ \hline '.format(all_cases[a_model], all_dataloglik_test_allinone['mean'][a_model], all_dataloglik_test_allinone['std'][a_model], all_dataloglik_test['mean'][a_model], all_dataloglik_test['std'][a_model]))

