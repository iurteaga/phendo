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

# Main code
def main(exec_machine, data_dir, result_dir, post_comp, R, traintest_ratio, traintest_split, split_n_obs):
    print('Inference of {} with mixture models: {} realizations with {} posterior computation'.format(data_dir, R, post_comp))
        
    # Directory configuration
    main_dir='{}/{}'.format(result_dir, data_dir.split('/')[-1])
    os.makedirs(main_dir, exist_ok=True)
    
    # Double-checking data dir
    assert os.path.isdir(data_dir), 'Could not find data dir {}'.format(data_dir)

    ########## Possibilities ##########
    # For each of the possible models
    python_scripts=[]
    # Parameters
    Ks=np.array([2,3,4,5])
    alphas=np.array([0.1,0.01, 0.001])
    betas=np.array([0.1,0.01,0.001])
    for K in Ks:
        for alpha in alphas:
            for beta in betas:
                # Mixture model with multiple data sources: full training
                python_scripts+=['./infer_MixtureModel_multipleDataSources.py -data_dir {} -result_dir {} -post_comp {} -R {} -Ks {} -alphas {} -betas {}'.format(data_dir, result_dir, post_comp, R, K, alpha, beta)]    

    ########## Execute inference ##########
    # Python script
    for (idx, python_script) in enumerate(python_scripts):
        job_name='run_{}_{}_{}'.format(idx, python_script.split()[0].split('/')[-1].split('.')[0], data_dir.split('/')[-1])
        
        # Execute
        print('Executing {} with out_file={}'.format(python_script,job_name))

        if exec_machine=='laptop':
            os.system('python3 {}'.format(python_script))
        else:
            raise ValueError('exec_machine={} unknown'.format(exec_machine))
        
    # In case we want to debug
    #pdb.set_trace()           


# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    parser = argparse.ArgumentParser(description='Infer mixture models given some observations with different data sources')
    parser.add_argument('-exec_machine', type=str, default='laptop', help='Where to run the simulation')
    parser.add_argument('-data_dir', type=str, help='Path to observed data')
    parser.add_argument('-result_dir', type=str, default='../results', help='Path for results to be saved at')
    parser.add_argument('-post_comp', type=str, default='online', help='Posterior computation type: general or online')
    parser.add_argument('-R', type=int, default=10, help='Number of realizations to run')
    parser.add_argument('-traintest_ratio', type=float, default=0.8, help='Train-test ratio')
    parser.add_argument('-traintest_split', type=str, default='balanced', help='Whether balanced or unbalanced split')
    parser.add_argument('-split_n_obs', type=int, default=50, help='Number of observations to use for splitting groups')
    # Get arguments
    args = parser.parse_args()
        
    # Call main function
    main(args.exec_machine, args.data_dir, args.result_dir, args.post_comp, args.R, args.traintest_ratio, args.traintest_split, args.split_n_obs)
