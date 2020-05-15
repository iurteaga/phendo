#!/usr/bin/python

# Imports
import pdb
import sys, os
import argparse
import time
import numpy as np
import scipy.stats as stats
from itertools import *
from matplotlib import colors
import pickle

# Import mixture models
from MixtureModel_multipleDataSources import *

# Globals
my_colors=[colors.cnames['black'], colors.cnames['cyan'], colors.cnames['blue'], colors.cnames['lime'], colors.cnames['green'], colors.cnames['yellow'], colors.cnames['orange'], colors.cnames['red'], colors.cnames['fuchsia'], colors.cnames['purple'], colors.cnames['pink']]

# Main code
def main(data_dir, data_sources, result_dir, post_comp, R, K, alpha, beta, traintest_ratio, traintest_split, split_n_obs, r):
    print('Inference of {} with mixture model: {} realizations with {} posterior computation'.format(data_dir, R, post_comp))

    # Directory configuration based on traintest split parameters
    main_dir='{}/{}/{}/{}/{}_{}_{}/{}/R_{}/r_{}'.format(result_dir, data_dir.split('/')[-1], os.path.basename(__file__).split('.')[0], post_comp, traintest_ratio, traintest_split, split_n_obs, data_sources, R,r)
    print(main_dir)
    assert os.path.isdir(main_dir)
    
    # Gibbs config
    gibbs={'init':'posterior_max', 'max_iter':1000, 'loglik_eps':0.000001, 'burn_in':10, 'lag':10}
    
    ########## INFERENCE ##########
    # Decide on execution types
    if data_sources == 'per_question':
        dir_data_sources=['']
    elif data_sources == 'allinone':
        dir_data_sources=['/allinone']
    elif data_sources == 'all':
        dir_data_sources=['', '/allinone']

    # For each type of execution
    for dir_data_source in dir_data_sources:
        # Set up directories for this execution
        dir_string='{}/K_{}/alpha_{}/beta_{}/'.format(main_dir,K,alpha,beta)
        os.makedirs(dir_string, exist_ok=True)
        r_train='{}/train{}'.format(dir_string, dir_data_source)
        r_test='{}/test{}'.format(dir_string, dir_data_source)
        os.makedirs(r_train, exist_ok=True)
        os.makedirs(r_test, exist_ok=True)
        
        # Emission distribution
        with open('{}{}/f_emission.pickle'.format(data_dir,dir_data_source), 'rb') as f:
            f_emission = pickle.load(f)

        D=f_emission.size
        # Emission hyperparameters
        g_prior=np.zeros(D, dtype='object')
        for d in np.arange(D):
            if 'multinomial' in f_emission[d]['dist'].__str__():
                g_prior[d]={'dist':stats.dirichlet, 'beta': beta*np.ones(f_emission[d]['d_x'])}
            else:
                raise ValueError('f_emission distribution {} not implemented yet'.format(f_emission[d]['dist']))    

        # Mixture prior configuration
        m_prior={'dist':stats.dirichlet, 'alpha':alpha*np.ones(K)}

        # Train set X
        with open('{}/train{}/X_train.pickle'.format(main_dir, dir_data_source), 'rb') as f:
            X_train=pickle.load(f)
        # Train set XD
        with open('{}/train{}/XD_train.pickle'.format(main_dir, dir_data_source), 'rb') as f:
            XD_train=pickle.load(f)

        # Test set X
        with open('{}/test{}/X_test.pickle'.format(main_dir, dir_data_source), 'rb') as f:
            X_test=pickle.load(f)
        # Train set XD
        with open('{}/test{}/XD_test.pickle'.format(main_dir, dir_data_source), 'rb') as f:
            XD_test=pickle.load(f)

        ### TRAIN
        print('TRAINING r={}/{}: Mixture model with {} data sources with K={}, alpha={}, beta={}'.format(r, R, D, K, alpha,beta))
        # Train mixture model
        mixtureModel=MixtureModel_multipleDataSources(K, m_prior, None, D, g_prior, f_emission)
        r_start_time = time.time()
        mixtureModel.run_gibbs_inference(X_train, XD_train, gibbs, r_train, p_computation=post_comp)
        inf_train_time=time.time() - r_start_time
        
        # Save realization data
        XcondXDZ_loglik, Z_loglik=mixtureModel.compute_loglikelihood()
        inf_loglik_train=XcondXDZ_loglik+Z_loglik
           
        # Save trained model
        with open(r_train+'/mixtureModel.pickle', 'wb') as f:
            pickle.dump(mixtureModel, f)
            
        # Plotting
        plot_save=r_train+'/plots'
        os.makedirs(plot_save, exist_ok=True)
        # Likelihood evolution
        mixtureModel.plot_loglikelihoods(r_train, plot_save)
        
        ### TEST
        print('TEST r={}/{}: Mixture model with {} data sources with K={}, alpha={}, beta={}'.format(r, R, D, K, alpha,beta))
        # Test set data assignments
        r_start_time = time.time()
        mixtureModel.run_test_gibbs_inference(X_test, XD_test, gibbs, r_test, p_computation=post_comp)
        inf_test_time=time.time() - r_start_time
        # Save train/test model
        with open(r_test+'/mixtureModel.pickle', 'wb') as f:
            pickle.dump(mixtureModel, f)
        
        ### Data likelihood
        estimation={'type':'lefttoright', 'samples_M':1000}
        # Train set data likelihood
        r_start_time = time.time()
        #inf_dataloglik_train=mixtureModel.estimate_test_datalikelihood(X_train, XD_train, estimation, r_train)
        #inf_dataloglik_time_train=time.time() - r_start_time
        # Test set data likelihood
        r_start_time = time.time()
        inf_dataloglik_test=mixtureModel.estimate_test_datalikelihood(X_test, XD_test, estimation, r_test)
        inf_dataloglik_time_test=time.time() - r_start_time
        
        # Save generated realization data
        with open(r_train+'/inf_train_time.pickle', 'wb') as f:
            pickle.dump(inf_train_time, f)
        with open(r_train+'/inf_loglik_train.pickle', 'wb') as f:
            pickle.dump(inf_loglik_train, f)
        
        #with open(r_train+'/inf_dataloglik_train.pickle', 'wb') as f:
        #    pickle.dump(inf_dataloglik_train, f)
        #with open(r_train+'/inf_dataloglik_time_train.pickle', 'wb') as f:
        #    pickle.dump(inf_dataloglik_time_train, f)
        
        with open(r_test+'/inf_test_time.pickle', 'wb') as f:
            pickle.dump(inf_test_time, f)
        with open(r_test+'/inf_dataloglik_test.pickle', 'wb') as f:
            pickle.dump(inf_dataloglik_test, f)
        with open(r_test+'/inf_dataloglik_time_test.pickle', 'wb') as f:
            pickle.dump(inf_dataloglik_time_test, f)
        print('FINISHED R={}: Mixture model with {} data sources with K={} topics, prior alpha={}, beta={}'.format(R, D, K, alpha,beta))
                
    # In case we want to debug
    #pdb.set_trace()           


# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    parser = argparse.ArgumentParser(description='Infer a Mixture Model given some observations')
    parser.add_argument('-data_dir', type=str, help='Path to observed data')
    parser.add_argument('-data_sources', type=str, default='per_question', help='Per_question or allinone data sources (all runs both)')
    parser.add_argument('-result_dir', type=str, default='../results', help='Path for results to be saved at')
    parser.add_argument('-post_comp', type=str, default='online', help='Posterior computation type: general or online')
    parser.add_argument('-R', type=int, default=10, help='Which split/realizations to run')
    parser.add_argument('-K', type=int, default=10, help='Number of mixtures K to consider')
    parser.add_argument('-alpha', type=float, default=0.1, help='Dirichlet prior alpha to consider')
    parser.add_argument('-beta', type=float, default=0.1, help='Dirichlet prior beta_0 to consider for emission distribution')
    parser.add_argument('-traintest_ratio', type=float, default=0.8, help='Train-test ratio')
    parser.add_argument('-traintest_split', type=str, default='balanced', help='Whether balanced or unbalanced split')
    parser.add_argument('-split_n_obs', type=int, default=50, help='Number of observations to use for splitting groups')
    parser.add_argument('-this_r', type=int, default=10, help='Which split/realizations to run')
    # Get arguments
    args = parser.parse_args()
       
    # Call main function
    main(args.data_dir, args.data_sources, args.result_dir, args.post_comp, args.R, args.K, args.alpha, args.beta, args.traintest_ratio, args.traintest_split, args.split_n_obs, args.this_r)
