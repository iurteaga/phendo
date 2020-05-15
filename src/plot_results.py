#!/usr/bin/python

# Imports
import sys, os, re, time
import argparse
import pdb
import pickle
from itertools import *
# Science
import numpy as np

########## FOR ALL Possibilities ##########
# For each of the run models: should match phenotype_MixtureModel_with_MultipleDataSources.py
Ks=np.array([2, 3, 4, 5])
alphas=np.array([0.1, 0.01, 0.001])
betas=np.array([0.1,0.01,0.001])
R=5
q_ids_names=['q_ids_njdm']

########## PLOTTING ALL ##########
plot_types=['perd_cond_mostprob','all_perd_cond_mostprob']
plot_types=['perd_cond_mostprob']
for plot_type in plot_types:
    for q_ids_name in q_ids_names:
        ### Vocab dir
        vocab_dir='../data/{}/vocab'.format(q_ids_name)

        ### Result dir
        result_dir='../results/{}/infer_MixtureModel_multipleDataSources/online'.format(q_ids_name)
        
        # Figure out data vocabulary list
        data_vocabs=''
        for d in np.arange(len(os.listdir(vocab_dir))):
            data_vocabs+='{}/d_{} '.format(vocab_dir,d)

        # For all parameters
        for K in Ks:
            for alpha in alphas:
                for beta in betas:
                    # Dir
                    this_result_dir='{}/K_{}/alpha_{}/beta_{}/R_{}'.format(result_dir, K, alpha, beta, R)
                    
                    if os.path.exists(this_result_dir+'/inf_time.pickle') and os.path.exists(this_result_dir+'/inf_loglik.pickle'):
                        # Inference loglikelihood and time for all R
                        with open(this_result_dir+'/inf_time.pickle', 'rb') as f:
                            inf_time=pickle.load(f)
                        with open(this_result_dir+'/inf_loglik.pickle', 'rb') as f:
                            inf_loglik=pickle.load(f)
                            
                        print('SUMMARY for {} with K={},alpha={},beta={},R={}'.format(q_ids_name, K, alpha, beta, R))
                        for r in np.arange(R):
                            print('r={}: log p(X,Z,|XD)={} in {}s'.format(r,inf_loglik[r],inf_time[r]))
                    
                    # Plotting
                    for r in np.arange(R):
                        python_script='plot_phendo_inferred_mixtures_multipleDataSources.py -inferred_model {}/r_{}/mixtureModel.pickle -data_source_vocabs {} -theta_cloud {} -plot_save {}/r_{}/plots '.format(this_result_dir, r, data_vocabs, plot_type, this_result_dir,r)
                        os.system('nohup python3 -u {} > {}/plotting_{}_K{}_alpha{}_beta{}_r{}.out 2>&1 &'.format(python_script, this_result_dir, q_ids_name, K, str(alpha).replace('.', ''), str(beta).replace('.', ''), r))

