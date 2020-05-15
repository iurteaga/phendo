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

###########        
# From what simulation
# For NJDM
q_ids_name='q_ids_njdm'
K=4
alpha=0.001
beta=0.001
R=5
r=1

# Participant ids
with open('../data/{}/data_participant_ids'.format(q_ids_name), 'rb') as f:
    data_participant_ids=np.loadtxt(f)

# Load model of interest
result_dir='../results/{}/infer_MixtureModel_multipleDataSources/online/K_{}/alpha_{}/beta_{}/R_{}/r_{}'.format(q_ids_name, K, alpha, beta, R,r)
with open('{}/mixtureModel.pickle'.format(result_dir), 'rb') as f:
    inferredModel=pickle.load(f)

# Compute empirical posterior for model of interest
N_sk=np.zeros((inferredModel.S, inferredModel.K))
# Iterate over sets
for s in np.arange(inferredModel.S):
    k_Z, count_K=np.unique(inferredModel.Z[s,~np.isnan(inferredModel.Z[s,:])], return_counts=True)
    N_sk[s,k_Z.astype(int)]=count_K

# Empirical posterior
emp_posterior=N_sk/N_sk.sum(axis=1, keepdims=True)

# Plot heatmap ordered by true_pid
fig, ax = plt.subplots(1)
cmap=plt.pcolormesh(emp_posterior, cmap='inferno', vmin=0., vmax=1.)
fig.colorbar(cmap)
# Put the major ticks at the middle of each cell
k=np.arange(inferredModel.K)
ax.set_xticks(k+ 0.5, minor=False)
# X labels are phenotype number
ax.set_xticklabels(k, minor=False)
plt.xlabel('k')
plt.ylabel('p_id')
plt.savefig('../data/{}/selected_participants/all_pid_emp_posterior.pdf'.format(q_ids_name), format='pdf', bbox_inches='tight')
plt.close()

# Get directory ready and save participant assignments
os.makedirs('../data/{}/selected_participants'.format(q_ids_name), exist_ok=True)

#### HARD cluster assignments
hard_assignments=np.argmax(emp_posterior, axis=1)

with open('../data/{}/selected_participants/participant_all_assignments'.format(q_ids_name),'w') as f:
    np.savetxt(f, np.array([data_participant_ids, hard_assignments], dtype=int).T, fmt='%s')

# Hard posterior
max_posterior=np.zeros(emp_posterior.shape)
max_posterior[np.arange(emp_posterior.shape[0]),hard_assignments]=1
# Plot heatmap for hard_posterior
fig, ax = plt.subplots(1)
cmap=plt.pcolormesh(max_posterior, cmap='inferno', vmin=0., vmax=1.)
fig.colorbar(cmap)
# Put the major ticks at the middle of each cell
k=np.arange(inferredModel.K)
ax.set_xticks(k+ 0.5, minor=False)
# X labels are phenotype number
ax.set_xticklabels(k, minor=False)
plt.xlabel('k')
plt.ylabel('p_id')
plt.savefig('../data/{}/selected_participants/all_pid_max_posterior.pdf'.format(q_ids_name), format='pdf', bbox_inches='tight')
plt.close()

#### Clear cluster assignments
# Cluster assignment threshold
assignment_threshold=0.95
threshold_assignments=(emp_posterior>assignment_threshold).sum(axis=1)
clear_assignments=np.argmax(emp_posterior[threshold_assignments==1,:], axis=1)

with open('../data/{}/selected_participants/participant_clear_assignments'.format(q_ids_name),'w') as f:
    np.savetxt(f, np.array([data_participant_ids[threshold_assignments==1], clear_assignments], dtype=int).T, fmt='%s')

# Clear posterior
clear_posterior=np.zeros(emp_posterior[threshold_assignments==1,:].shape)
clear_posterior[np.arange(emp_posterior[threshold_assignments==1,:].shape[0]),clear_assignments]=1
# Plot heatmap for hard_posterior
fig, ax = plt.subplots(1)
cmap=plt.pcolormesh(clear_posterior, cmap='inferno', vmin=0., vmax=1.)
fig.colorbar(cmap)
# Put the major ticks at the middle of each cell
k=np.arange(inferredModel.K)
ax.set_xticks(k+ 0.5, minor=False)
# X labels are phenotype number
ax.set_xticklabels(k, minor=False)
plt.xlabel('k')
plt.ylabel('p_id')
plt.savefig('../data/{}/selected_participants/pid_clear_posterior.pdf'.format(q_ids_name), format='pdf', bbox_inches='tight')
plt.close()

