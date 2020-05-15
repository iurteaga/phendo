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
def main(data_dir, result_dir, post_comp, R, Ks, alphas, betas):
    print('Inference of {} with mixture model: {} realizations with {} posterior computation'.format(data_dir, R, post_comp))

    # Directory configuration
    main_dir='{}/{}/{}/{}'.format(result_dir, data_dir.split('/')[-1], os.path.basename(__file__).split('.')[0], post_comp)
    os.makedirs(main_dir, exist_ok=True)
    
    # Gibbs config
    gibbs={'max_iter':1000, 'loglik_eps':0.000001, 'burn_in':10, 'lag':10}
        
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
    
    ########## INFERENCE ##########
    for K in Ks:
        for alpha in alphas:
            for beta in betas:
                # Emission hyperparameters
                g_prior=np.zeros(D, dtype='object')
                for d in np.arange(D):
                    if 'multinomial' in f_emission[d]['dist'].__str__():
                        g_prior[d]={'dist':stats.dirichlet, 'beta': beta*np.ones(f_emission[d]['d_x'])}
                    else:
                        raise ValueError('f_emission distribution {} not implemented yet'.format(f_emission[d]['dist']))
            
                print('START R={}: Mixture model with {} data sources with K={} topics, prior alpha={}, beta={}'.format(R, D, K, alpha,beta))
                
                # Mixture prior configuration
                m_prior={'dist':stats.dirichlet, 'alpha':alpha*np.ones(K)}

                # Per realization data
                inf_time=np.zeros(R)
                inf_loglik=np.zeros(R)

                dir_string=main_dir+'/K_{}/alpha_{}/beta_{}/R_{}'.format(K,alpha,beta,R)
                os.makedirs(dir_string, exist_ok=True)
                for idx_r,r in enumerate(np.arange(R)):
                    print('RUNNING r={}/{}: Mixture model with {} data sources with K={} topics, prior alpha={}, beta={}'.format(r, R, D, K, alpha,beta))
                    r_string=dir_string+'/r_{}'.format(r)
                    os.makedirs(r_string, exist_ok=True)
                    # Create object
                    mixtureModel=MixtureModel_multipleDataSources(K, m_prior, None, D, g_prior, f_emission)
                    r_start_time = time.time()
                    mixtureModel.run_gibbs_inference(X, XD, gibbs, r_string, p_computation=post_comp)
                    inf_time[idx_r]=time.time() - r_start_time
                    
                    # Save realization data
                    XcondXDZ_loglik, Z_loglik=mixtureModel.compute_loglikelihood()
                    inf_loglik[idx_r]=XcondXDZ_loglik+Z_loglik
                       
                    # Save generated data
                    with open(r_string+'/mixtureModel.pickle', 'wb') as f:
                        pickle.dump(mixtureModel, f)
                        
                    # Plotting
                    plot_save=None
                    plot_save=r_string+'/plots'
                    os.makedirs(plot_save, exist_ok=True)
                    # Likelihood evolution
                    mixtureModel.plot_loglikelihoods(r_string, plot_save)
                
                # Save generated realization data
                with open(dir_string+'/inf_time.pickle', 'wb') as f:
                    pickle.dump(inf_time, f)
                with open(dir_string+'/inf_loglik.pickle', 'wb') as f:
                    pickle.dump(inf_loglik, f)
                print('FINISHED R={}: Mixture model with {} data sources with K={} topics, prior alpha={}, beta={}'.format(R, D, K, alpha,beta))
    # In case we want to debug
    #pdb.set_trace()           


# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    # Example: python3 -m pdb infer_MixtureModel.py -data_dir ../data/X_ap_vpointer.pickle -post_comp online -R 1 -alphas 1 -Ks 10
    parser = argparse.ArgumentParser(description='Infer a Mixture Model given some observations')
    parser.add_argument('-data_dir', type=str, help='Path to observed data')
    parser.add_argument('-result_dir', type=str, default='../results', help='Path for results to be saved at')
    parser.add_argument('-post_comp', type=str, default='online', help='Posterior computation type: general or online')
    parser.add_argument('-R', type=int, default=10, help='Number of realizations to run')
    parser.add_argument('-Ks', nargs='+', type=int, default=np.array([10]), help='Number of mixtures K to consider')
    parser.add_argument('-alphas', nargs='+', type=float, default=np.array([0.1]), help='Dirichlet prior alphas to consider')
    parser.add_argument('-betas', nargs='+', type=float, default=np.array([0.1]), help='Dirichlet prior beta_0 to consider for emission distribution')

    # Get arguments
    args = parser.parse_args()
       
    # Call main function
    main(args.data_dir, args.result_dir, args.post_comp, args.R, np.array(args.Ks), np.array(args.alphas), np.array(args.betas))
