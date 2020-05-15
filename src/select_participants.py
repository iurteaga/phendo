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

# Import mixture models
from MixtureModel_multipleDataSources import *

########### SELECT patients of interest
# Number of observations to have 
n_obs=100
n_days=30
# Number of participants per type
n_per_type=8
# Cluster assignment threshold
assignment_threshold=0.95

# From what simulation
# For NJDM
q_ids_name='q_ids_njdm'
K=4
alpha=0.001
beta=0.001
R=5
r=1

# File
result_dir='../results/{}/infer_MixtureModel_multipleDataSources/online/K_{}/alpha_{}/beta_{}/R_{}/r_{}'.format(q_ids_name, K, alpha, beta, R,r)

# Load vocab
vocab_dir='../data/{}/vocab'.format(q_ids_name)
vocab=[]
for d in np.arange(len(os.listdir(vocab_dir))):
    # Load data source vocabularies
    with open('{}/d_{}'.format(vocab_dir,d)) as f:
            vocab.append(f.read().splitlines())

# Load model of interest
with open('{}/mixtureModel.pickle'.format(result_dir), 'rb') as f:
    inferredModel=pickle.load(f)
    
assert len(vocab)==inferredModel.D

# Compute empirical posterior for model of interest
N_sk=np.zeros((inferredModel.S, inferredModel.K))
# Iterate over sets for plotting
for s in np.arange(inferredModel.S):
    k_Z, count_K=np.unique(inferredModel.Z[s,~np.isnan(inferredModel.Z[s,:])], return_counts=True)
    N_sk[s,k_Z.astype(int)]=count_K

emp_posterior=N_sk/N_sk.sum(axis=1, keepdims=True)

# Number of observations and days per cluster
with open('../data/{}/n_observations_matrix.pickle'.format(q_ids_name), 'rb') as f:
    n_observations_matrix=pickle.load(f)
with open('../data/{}/n_days_matrix.pickle'.format(q_ids_name), 'rb') as f:
    n_days_matrix=pickle.load(f)

for k in np.arange(K):
    print('#### Cluster {} ####'.format(k))
    print('Average {} and max {} observations'.format(n_observations_matrix.sum(axis=1, dtype=int)[emp_posterior[:,k]>assignment_threshold].mean(axis=0), n_observations_matrix.sum(axis=1, dtype=int)[emp_posterior[:,k]>assignment_threshold].max(axis=0)))
    print('Average {} and max {} days'.format(n_days_matrix.sum(axis=1, dtype=int)[emp_posterior[:,k]>assignment_threshold].mean(axis=0), n_days_matrix.sum(axis=1, dtype=int)[emp_posterior[:,k]>assignment_threshold].max(axis=0)))

# Select participants, by id
selected_pid=np.zeros((K+1,n_per_type), dtype=int)
# participant id in my matrix
all_p=np.arange(inferredModel.S)
# Load true participant_ids (for mapping)
with open('../data/{}/data_participant_ids'.format(q_ids_name)) as f:
    true_p_ids=np.loadtxt(f,dtype='str')

# Participants clearly assigned and with "enough" observations
p_some=N_sk.sum(axis=1)>n_obs
p_some=n_days_matrix.sum(axis=1, dtype=int)>n_days

for k in np.arange(K):
    this_k=np.where(emp_posterior[p_some,k]>assignment_threshold)[0]
    selected_pid[k,:]=all_p[p_some][this_k[np.random.permutation(this_k.size)][:n_per_type].astype(int)]

# Doubtful participants
#unsure_k=np.where((emp_posterior[p_some,0]<0.55).astype(bool) & (emp_posterior[p_some,1]<0.55).astype(bool) & (emp_posterior[p_some,2]<0.55).astype(bool))[0]
unsure_k=np.where((emp_posterior[p_some]>0.4).sum(axis=1)>1)[0]
selected_pid[-1,:]=all_p[p_some][unsure_k[np.random.permutation(unsure_k.size)][:n_per_type].astype(int)]

# Get directory ready and save list of ids
os.makedirs('../data/{}/selected_participants'.format(q_ids_name), exist_ok=True)
# My numbers
with open('../data/{}/selected_participants/selected_pid'.format(q_ids_name),'w') as f:
    np.savetxt(f, selected_pid.flatten(), fmt='%s')

# True participant_ids
with open('../data/{}/selected_participants/selected_true_pid'.format(q_ids_name),'w') as f:
    np.savetxt(f, true_p_ids[selected_pid.flatten()], fmt='%s')

# And their assignment posteriors
with open('../data/{}/selected_participants/selected_pid_posterior'.format(q_ids_name),'w') as f:
    np.savetxt(f, emp_posterior[selected_pid.flatten(),:], fmt='%s')

# Plot heatmap for selected
fig, ax = plt.subplots(1)
cmap=plt.pcolormesh(emp_posterior[selected_pid.flatten(),:], cmap='inferno', vmin=0., vmax=1.)
fig.colorbar(cmap)
# Put the major ticks at the middle of each cell
k=np.arange(inferredModel.K)
ax.set_xticks(k+ 0.5, minor=False)
# X labels are phenotype number
ax.set_xticklabels(k, minor=False)
plt.xlabel('k')
plt.ylabel('p_id')
plt.savefig('../data/{}/selected_participants/selected_pid_posterior.pdf'.format(q_ids_name), format='pdf', bbox_inches='tight')
plt.close()

########### PRINT selected patients data
# Load data source to q_id and type mapping
with open('../data/{}/d_q_id_type'.format(q_ids_name), 'rb') as f:
    d_q_ids=np.loadtxt(f,dtype='str')

for (p_idx,p_id) in enumerate(selected_pid.flatten()):
    print('##############################')
    print('PATIENT #{}'.format(p_idx))
    for d in np.arange(inferredModel.D):
        # Find words and counts within this vocab
        v,count_v=np.unique(inferredModel.X[p_id,inferredModel.XD[p_id,:]==d], return_counts=True)
        #n_v_d=np.zeros(len(vocab[d]), dtype=int)
        #n_v_d[v.astype(int)]=count_v
        responses=''
        for (v_idx, v_d) in enumerate(v.astype(int)):
            responses+=' {}({})'.format(vocab[d][v_d], count_v[v_idx])
        print('{}:{}'.format(d_q_ids[d,2],responses))
    print('##############################')
    print('')
    

