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

########### LOAD patients of interest
q_ids_name='q_ids_njdm'
main_result_dir='../results'
selected_participants_dir='../data/{}/selected_participants'.format(q_ids_name)

# Load list of selected participant ids
with open('{}/selected_pid'.format(selected_participants_dir),'r') as f:
    selected_pid=np.loadtxt(f,dtype=int)
    
# Load expert grouping
expert_grouping_dir='{}/expert_groupings'.format(selected_participants_dir)
expert_file='expert_1'
with open('{}/{}'.format(expert_grouping_dir,expert_file),'r') as f:
    # p_idxs to group
    expert_grouping={selected_pid[int(p_idx)]:group for (group,line) in enumerate(f.read().splitlines()) for p_idx in line.split(',')}
    
# Expert groups
groups=np.unique(np.array([val for val in expert_grouping.values()]))
# Expert "posterior"
expert_posterior=np.zeros((selected_pid.size, groups.size))
for (p_idx,p_id) in enumerate(selected_pid):
    expert_posterior[p_idx,expert_grouping[p_id]]=1.

# Plot heatmap for selected
fig, ax = plt.subplots(1)
cmap=plt.pcolormesh(expert_posterior, cmap='inferno', vmin=0., vmax=1.)
fig.colorbar(cmap)
# Put the major ticks at the middle of each cell
k=groups
ax.set_xticks(k+ 0.5, minor=False)
# X labels are phenotype number
ax.set_xticklabels(k, minor=False)
plt.xlabel('k')
plt.ylabel('p_id')
plt.savefig('../data/{}/selected_participants/expert_groupings/selected_pid_posterior_{}.pdf'.format(q_ids_name, expert_file), format='pdf', bbox_inches='tight')
plt.close()

########### Selected participant clusterings
# From what simulation
q_ids_name='q_ids_njdm'
K=4
alpha=0.001
beta=0.001
R=5
r=1

# Result dir
result_dir='../results/{}/infer_MixtureModel_multipleDataSources/online/K_{}/alpha_{}/beta_{}/R_{}/r_{}'.format(q_ids_name, K, alpha, beta, R,r)
                
if os.path.exists(result_dir+'/mixtureModel.pickle'):
    # Load model
    with open('{}/mixtureModel.pickle'.format(result_dir), 'rb') as f:
        inferredModel=pickle.load(f)
    
        # Topics per patient heatmap
        N_sk=np.zeros((inferredModel.S, inferredModel.K))
        # Iterate over sets for plotting
        for s in np.arange(inferredModel.S):
            k_Z, count_K=np.unique(inferredModel.Z[s,~np.isnan(inferredModel.Z[s,:])], return_counts=True)
            N_sk[s,k_Z.astype(int)]=count_K

        # Empirical posterior
        emp_posterior=N_sk/N_sk.sum(axis=1, keepdims=True)    
        
        ############## K Vs K ###################        
        # Hard assignments
        hard_cluster_assignment=emp_posterior.argmax(axis=1)
        # Confusion matrix
        conf_matrix=np.zeros((K,groups.size))
        for p_id in selected_pid:
            conf_matrix[hard_cluster_assignment[p_id],expert_grouping[p_id]]+=1
            
        # Purity
        purity=np.max(conf_matrix, axis=1).sum()/selected_pid.size
        # Normalized mutual information
        mi_tmp=conf_matrix/selected_pid.size*np.log((conf_matrix*selected_pid.size)/(conf_matrix.sum(axis=0,keepdims=True)*conf_matrix.sum(axis=1,keepdims=True)))
        mi_tmp[conf_matrix==0]=0.
        mi=np.sum(mi_tmp)
        h_c=-np.sum(conf_matrix.sum(axis=0,keepdims=True)/selected_pid.size*np.log(conf_matrix.sum(axis=0,keepdims=True)/selected_pid.size))
        h_g=-np.sum(conf_matrix.sum(axis=1,keepdims=True)/selected_pid.size*np.log(conf_matrix.sum(axis=1,keepdims=True)/selected_pid.size))
        nmi=mi/(h_c+h_g)/2
        
        ############## Severity yes/no ###################      
        # Confusion matrix
        sev_conf_matrix=np.zeros((2,2))
        sev_conf_matrix[0,0]=conf_matrix[0,0].sum()
        sev_conf_matrix[0,1]=conf_matrix[0,1:].sum()
        sev_conf_matrix[1,0]=conf_matrix[1:K,0].sum()
        sev_conf_matrix[1,1]=conf_matrix[1:K,1:].sum()
        
        # Purity
        sev_purity=np.max(sev_conf_matrix, axis=1).sum()/selected_pid.size
        # Normalized mutual information
        mi_tmp=sev_conf_matrix/selected_pid.size*np.log((sev_conf_matrix*selected_pid.size)/(sev_conf_matrix.sum(axis=0,keepdims=True)*sev_conf_matrix.sum(axis=1,keepdims=True)))
        mi_tmp[sev_conf_matrix==0]=0.
        sev_mi=np.sum(mi_tmp)
        h_c=-np.sum(sev_conf_matrix.sum(axis=0,keepdims=True)/selected_pid.size*np.log(sev_conf_matrix.sum(axis=0,keepdims=True)/selected_pid.size))
        h_g=-np.sum(sev_conf_matrix.sum(axis=1,keepdims=True)/selected_pid.size*np.log(sev_conf_matrix.sum(axis=1,keepdims=True)/selected_pid.size))
        sev_nmi=sev_mi/(h_c+h_g)/2
        ############## Mild yes/no ###################      
        # Confusion matrix
        not_mild_index_K=np.setdiff1d(np.arange(K),[1])
        not_mild_index_groups=np.setdiff1d(np.arange(groups.size),[1])
        mild_conf_matrix=np.zeros((2,2))
        mild_conf_matrix[0,0]=conf_matrix[1,1].sum()
        mild_conf_matrix[0,1]=conf_matrix[1,not_mild_index_groups].sum()
        mild_conf_matrix[1,0]=conf_matrix[not_mild_index_K,1].sum()
        mild_conf_matrix[1,1]=conf_matrix[np.ix_(not_mild_index_K,not_mild_index_groups)].sum()
            
        # Purity
        mild_purity=np.max(mild_conf_matrix, axis=1).sum()/selected_pid.size
        # Normalized mutual information
        mi_tmp=mild_conf_matrix/selected_pid.size*np.log((mild_conf_matrix*selected_pid.size)/(mild_conf_matrix.sum(axis=0,keepdims=True)*mild_conf_matrix.sum(axis=1,keepdims=True)))
        mi_tmp[mild_conf_matrix==0]=0.
        mild_mi=np.sum(mi_tmp)
        h_c=-np.sum(mild_conf_matrix.sum(axis=0,keepdims=True)/selected_pid.size*np.log(mild_conf_matrix.sum(axis=0,keepdims=True)/selected_pid.size))
        h_g=-np.sum(mild_conf_matrix.sum(axis=1,keepdims=True)/selected_pid.size*np.log(mild_conf_matrix.sum(axis=1,keepdims=True)/selected_pid.size))
        mild_nmi=mild_mi/(h_c+h_g)/2
        print('##############################')
        print('K={}, alpha={}, beta={}, r={}:'.format(K,alpha,beta,r))
        print('##############################')
        print('Confusion matrix')
        print('{}'.format(conf_matrix))
        print('purity={}/nmi={}'.format(purity,nmi))
        print('##############################')
        print('Severity Confusion matrix')
        print('{}'.format(sev_conf_matrix))
        print('sev_purity={}/sev_nmi={}'.format(sev_purity,sev_nmi))
        print('##############################')
        print('##############################')
        print('Mild Confusion matrix')
        print('{}'.format(mild_conf_matrix))
        print('mild_purity={}/mild_nmi={}'.format(mild_purity,mild_nmi))
        print('##############################')
        print('')

