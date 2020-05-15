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

# Helpful for WERF processing
from werf_phendo_questionnaire_preprocessing_functions import *

# Main code
def main(exec_machine, data_dir, data_sources, result_dir, post_comp, R, traintest_ratio, traintest_split, split_n_obs):
    print('Inference of {} with mixture models: {} realizations with {} posterior computation'.format(data_dir, R, post_comp))

    # Double-checking data dir
    assert os.path.isdir(data_dir), 'Could not find data dir {}'.format(data_dir)
    
    # Directory configuration
    if 'WERF_based' in traintest_split:
        # Figure out once what is the train-test split based on whether user's have WERF data:
        # All participant ids
        with open('{}/data_participant_ids'.format(data_dir), 'rb') as f:
            data_participant_ids=np.loadtxt(f).astype(int)
        # Paths to files in servers
        # WERF related info files
        werf_file='../data/werf_and_profile_data/werf_survey.csv'
        werf_to_phendo_file='../data/werf_and_profile_data/phendoid_email_pid.csv'
        # Profile info file
        profile_file='../data/werf_and_profile_data/profile.csv'
        # Age info file
        age_file='../data/werf_and_profile_data/participant_dob.csv'
        # WERF AND PHENDO DATAFRAME
        # Combine Werf and Phendo
        werf_data_to_phendo=load_werf_and_phendo_data(werf_file, werf_to_phendo_file)
        # Clean Werf and phendo dataframe
        werf_data_to_phendo=clean_werf_data_to_phendo(werf_data_to_phendo)
        # PROFILE DATAFRAME
        # Combine profile info
        participant_profile=create_participant_profile_dataframe(profile_file, age_file)
        # Merge werf_data with phendo participant info
        werf_and_profile_data=pd.merge(werf_data_to_phendo, participant_profile, how='inner', on=['participant_id']).astype({'participant_id':pd.api.types.CategoricalDtype()})
        # Identify those with WERF data
        participants_with_werf=(data_participant_ids[:,None]==np.unique(werf_and_profile_data['participant_id'].values)[None,:]).sum(axis=1)
        
    # Directory configuration based on traintest split parameters
    main_dir='{}/{}/traintest_with_MixtureModel_multipleDataSources/{}/{}_{}_{}/{}/R_{}'.format(result_dir, data_dir.split('/')[-1], post_comp, traintest_ratio, traintest_split, split_n_obs, data_sources, R)
    # Create dir
    os.makedirs(main_dir, exist_ok=True)

    if data_sources != 'allinone':
        ### Data sources per question
        # Load true data
        with open('{}/X.pickle'.format(data_dir), 'rb') as f:
            X = pickle.load(f)
        # Data sources info
        with open('{}/XD.pickle'.format(data_dir), 'rb') as f:
            XD = pickle.load(f)
        # And Emission distribution
        with open('{}/f_emission.pickle'.format(data_dir), 'rb') as f:
            f_emission = pickle.load(f)
    
        # Data sources
        assert np.unique(XD[~np.isnan(XD)]).size == f_emission.size, 'Data sources for XD D={} and f_emission D={} mismatch'.format(np.unique(XD[~np.isnan(XD)]).size, f_emission.size)
        D=f_emission.size
    
    if data_sources != 'per_question':
        ### All in one data source
        # Load true data
        with open('{}/allinone/X.pickle'.format(data_dir), 'rb') as f:
            X_allinone = pickle.load(f)
        # Data sources info
        with open('{}/allinone/XD.pickle'.format(data_dir), 'rb') as f:
            XD_allinone = pickle.load(f)
        # And Emission distribution
        with open('{}/allinone/f_emission.pickle'.format(data_dir), 'rb') as f:
            f_emission_allinone = pickle.load(f)
        
        # Data sources
        assert np.unique(XD_allinone[~np.isnan(XD_allinone)]).size == f_emission_allinone.size, 'Allinone data sources for XD D={} and f_emission D={} mismatch'.format(np.unique(XD_allinone[~np.isnan(XD)]).size, f_emission_allinone.size)
        D=f_emission.size
    
    # Make sure sizes match
    if data_sources == 'all':
        assert X.shape[0]==X_allinone.shape[0]
        N_s=X.shape[0]
        assert N_s==XD.shape[0], 'N_s in X ={} and and in XD={} mismatch'.format(N_s, XD.shape[0])
    elif data_sources == 'per_question':
        N_s=X.shape[0]
        assert N_s==XD.shape[0], 'N_s in X ={} and and in XD={} mismatch'.format(N_s, XD.shape[0])
    elif data_sources == 'allinone':
        N_s=X_allinone.shape[0]
        assert N_s==XD_allinone.shape[0], 'N_s in X ={} and and in XD={} mismatch'.format(N_s, XD.shape[0])

    # All scripts        
    python_scripts=[]
    for r in np.arange(R):
        print('Splitting r={}/{}'.format(r, R))

        # Train-test set splitting        
        if traintest_split=='unbalanced':
            # Randomly pick indexes across all patients
            s_idx=np.random.permutation(N_s)
            N_strain=np.floor(N_s*traintest_ratio).astype(int)
            train_idx=s_idx[:N_strain]
            test_idx=s_idx[N_strain:]
        elif traintest_split=='balanced':
            # Randomly pick indexes across patients in two groups by number of observations
            if data_sources != 'allinone':
                few_obs_idx=np.where((~np.isnan(X)).sum(axis=1)<split_n_obs)[0]
                plenty_obs_idx=np.where((~np.isnan(X)).sum(axis=1)>=split_n_obs)[0]
            elif data_sources == 'per_question':
                few_obs_idx=np.where((~np.isnan(X_allinone)).sum(axis=1)<split_n_obs)[0]
                plenty_obs_idx=np.where((~np.isnan(X_allinone)).sum(axis=1)>=split_n_obs)[0]
            
            few_s_idx=few_obs_idx[np.random.permutation(few_obs_idx.size)]
            few_N_strain=np.floor(few_obs_idx.size*traintest_ratio).astype(int)
            plenty_s_idx=plenty_obs_idx[np.random.permutation(plenty_obs_idx.size)]
            plenty_N_strain=np.floor(plenty_obs_idx.size*traintest_ratio).astype(int)
            train_idx=np.concatenate((few_s_idx[:few_N_strain],plenty_s_idx[:plenty_N_strain]))
            test_idx=np.concatenate((few_s_idx[few_N_strain:],plenty_s_idx[plenty_N_strain:]))
        elif traintest_split=='WERF_based':
            # Split train-test set based on whether user's have WERF data:
            # Test set contains participants WITH WERF data
            test_idx=np.arange(N_s)[participants_with_werf==1]
            # If not all WERF should be included in test set
            if traintest_ratio != 0:
                # Randomly split WERF indexes across train and test
                s_idx=np.random.permutation(test_idx)
                N_strain=np.floor(test_idx.size*traintest_ratio).astype(int)
                test_idx=s_idx[N_strain:]
            
            # Training set contains participants not in test set
            train_idx=np.setdiff1d(np.arange(N_s), test_idx)
            
        elif traintest_split=='WERF_based_balanced':
            # Split train-test set based on whether user's have WERF data
            # AND across patients in two groups by number of observations
            # Test set contains participants WITH WERF data
            test_idx=np.arange(N_s)[participants_with_werf==1]
            # Randomly pick indexes across patients in two groups by number of observations
            if data_sources != 'allinone':
                few_obs_test_idx=np.where((~np.isnan(X[test_idx])).sum(axis=1)<split_n_obs)[0]
                plenty_obs_test_idx=np.where((~np.isnan(X[test_idx])).sum(axis=1)>=split_n_obs)[0]
            elif data_sources == 'per_question':
                few_obs_test_idx=np.where((~np.isnan(X_allinone[test_idx])).sum(axis=1)<split_n_obs)[0]
                plenty_obs_test_idx=np.where((~np.isnan(X_allinone[test_idx])).sum(axis=1)>=split_n_obs)[0]
            
            # If not all WERF should be included in test set
            if traintest_ratio != 0:
                # Randomly split WERF indexes across train and test
                few_test_s_idx=test_idx[few_obs_test_idx[np.random.permutation(few_obs_test_idx.size)]]
                few_N_strain=np.floor(few_obs_test_idx.size*traintest_ratio).astype(int)
                plenty_test_s_idx=test_idx[plenty_obs_test_idx[np.random.permutation(plenty_obs_test_idx.size)]]
                plenty_N_strain=np.floor(plenty_obs_test_idx.size*traintest_ratio).astype(int)
                test_idx=np.concatenate((few_test_s_idx[few_N_strain:],plenty_test_s_idx[plenty_N_strain:]))
            
            # Training set contains participants not in test set
            train_idx=np.setdiff1d(np.arange(N_s), test_idx)
            
        # Make sure split makes sense
        assert np.setdiff1d(np.arange(N_s), np.concatenate((train_idx,test_idx))).size==0
        
        # Split data    
        if data_sources != 'allinone':
            # Train set
            X_train=X[train_idx]
            XD_train=XD[train_idx]
            # Test set
            X_test=X[test_idx]
            XD_test=XD[test_idx]
        if data_sources != 'per_question':
            # Train set
            X_allinone_train=X_allinone[train_idx]
            XD_allinone_train=XD_allinone[train_idx]
            # Test set
            X_allinone_test=X_allinone[test_idx]
            XD_allinone_test=XD_allinone[test_idx]
        
        # Per split directories
        r_string=main_dir+'/r_{}'.format(r)
        os.makedirs(r_string, exist_ok=True)

        # Save data        
        if data_sources != 'allinone':
            r_train=r_string+'/train'
            os.makedirs(r_train, exist_ok=True)
            r_test=r_string+'/test'
            os.makedirs(r_test, exist_ok=True)

            # Train set indexes
            with open('{}/train_idx.pickle'.format(r_train), 'wb') as f:
                pickle.dump(train_idx, f)
            # Train set X
            with open('{}/X_train.pickle'.format(r_train), 'wb') as f:
                pickle.dump(X_train, f)
            # Train set XD
            with open('{}/XD_train.pickle'.format(r_train), 'wb') as f:
                pickle.dump(XD_train, f)
            
            # Test set indexes
            with open('{}/test_idx.pickle'.format(r_test), 'wb') as f:
                pickle.dump(test_idx, f)
            # Test set X
            with open('{}/X_test.pickle'.format(r_test), 'wb') as f:
                pickle.dump(X_test, f)
            # Train set XD
            with open('{}/XD_test.pickle'.format(r_test), 'wb') as f:
                pickle.dump(XD_test, f)

        if data_sources != 'per_question':
            r_train=r_string+'/train/allinone'
            os.makedirs(r_train, exist_ok=True)
            r_test=r_string+'/test/allinone'
            os.makedirs(r_test, exist_ok=True)

            # Train set indexes
            with open('{}/train_idx.pickle'.format(r_train), 'wb') as f:
                pickle.dump(train_idx, f)
            # Train set X
            with open('{}/X_train.pickle'.format(r_train), 'wb') as f:
                pickle.dump(X_allinone_train, f)
            # Train set XD
            with open('{}/XD_train.pickle'.format(r_train), 'wb') as f:
                pickle.dump(XD_allinone_train, f)
            
            # Test set indexes
            with open('{}/test_idx.pickle'.format(r_test), 'wb') as f:
                pickle.dump(test_idx, f)
            # Test set X
            with open('{}/X_test.pickle'.format(r_test), 'wb') as f:
                pickle.dump(X_allinone_test, f)
            # Train set XD
            with open('{}/XD_test.pickle'.format(r_test), 'wb') as f:
                pickle.dump(XD_allinone_test, f)
        
        ########## Possibilities ##########
        # For each of the possible models
        # Parameters
        Ks=np.array([2,3,4,5])
        alphas=np.array([0.1,0.01, 0.001])
        betas=np.array([0.1,0.01,0.001])
        for K in Ks:
            for alpha in alphas:
                for beta in betas:
                    # Mixture model with multiple data sources: train/test
                    python_scripts+=['./traintest_with_MixtureModel_multipleDataSources.py -data_dir {} -data_sources {} -result_dir {} -post_comp {} -R {} -K {} -alpha {} -beta {} -traintest_ratio {} -traintest_split {} -split_n_obs {} -this_r {}'.format(data_dir, data_sources, result_dir, post_comp, R, K, alpha, beta, traintest_ratio, traintest_split, split_n_obs, r)]
    

    ########## Execute inference ##########
    # Python script
    for (idx, python_script) in enumerate(python_scripts):
        job_name='run_{}_{}_{}_{}_{}_{}'.format(idx, python_script.split()[0].split('/')[-1].split('.')[0], data_dir.split('/')[-1], traintest_ratio, traintest_split, split_n_obs)
        
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
    parser.add_argument('-data_sources', type=str, default='per_question', help='Per_question or allinone data sources (all runs both)')
    parser.add_argument('-result_dir', type=str, default='../results', help='Path for results to be saved at')
    parser.add_argument('-post_comp', type=str, default='online', help='Posterior computation type: general or online')
    parser.add_argument('-R', type=int, default=10, help='Number of realizations to run')
    parser.add_argument('-traintest_ratio', type=float, default=0.8, help='Train-test ratio')
    parser.add_argument('-traintest_split', type=str, default='balanced', help='Whether balanced or unbalanced split')
    parser.add_argument('-split_n_obs', type=int, default=50, help='Number of observations to use for splitting groups')
    # Get arguments
    args = parser.parse_args()
        
    # Call main function
    main(args.exec_machine, args.data_dir, args.data_sources, args.result_dir, args.post_comp, args.R, args.traintest_ratio, args.traintest_split, args.split_n_obs)
