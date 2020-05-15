#!/usr/bin/python

# Imports
import pdb
import sys, os, time
import argparse
import pandas as pd
import numpy as np
import scipy.stats as stats
from itertools import *
import pickle
# Plotting
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D
# Sklearn helpers
from functools import partial
from collections import OrderedDict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import manifold, datasets
from sklearn.metrics import confusion_matrix

# For associations
from compute_associations import *

############################################################
def model_assignments(original_result_dir, data_participant_ids, assignments='hard', assignment_threshold=0.75):
    #Load original model
    with open('{}/mixtureModel.pickle'.format(original_result_dir), 'rb') as f:
        original_inferredModel=pickle.load(f)
    
    # Compute empirical posterior for original model
    N_sk=np.zeros((original_inferredModel.S, original_inferredModel.K))
    # Iterate over sets
    for s in np.arange(original_inferredModel.S):
        k_Z, count_K=np.unique(original_inferredModel.Z[s,~np.isnan(original_inferredModel.Z[s,:])], return_counts=True)
        N_sk[s,k_Z.astype(int)]=count_K

    # Empirical posterior
    original_emp_posterior=N_sk/N_sk.sum(axis=1, keepdims=True)

    if assignments == 'clear':
        #### Clear cluster assignments
        # Original model
        original_threshold_assignments=(original_emp_posterior>assignment_threshold).sum(axis=1)
        original_clear_assignments=np.argmax(original_emp_posterior[original_threshold_assignments==1,:], axis=1)
        original_participant_assignments=pd.DataFrame(data={'participant_id': data_participant_ids[original_threshold_assignments.astype(bool)], 'assigned_cluster': original_clear_assignments}, dtype=pd.api.types.CategoricalDtype())

    elif assignments == 'hard':
        #### HARD cluster assignments
        # Original model
        original_hard_assignments=np.argmax(original_emp_posterior, axis=1)
        original_participant_assignments=pd.DataFrame(data={'participant_id': data_participant_ids, 'assigned_cluster': original_hard_assignments}, dtype=pd.api.types.CategoricalDtype())

    else:
        raise ValueError('Unknown {} assignment'.format(assignments))

    return original_participant_assignments
    
def kmeans_assignments(X, data_participant_ids, n_clusters=4):
    # Run k-means
    my_kmeans = KMeans(n_clusters=n_clusters).fit(X)
    # HARD cluster
    kmeans_participant_assignments=pd.DataFrame(data={'participant_id': data_participant_ids, 'assigned_cluster': my_kmeans.labels_}, dtype=pd.api.types.CategoricalDtype())
    return kmeans_participant_assignments

def add_low_dimensional_cluster_plot(fig, fig_pos, Y, cluster_labels, title):
    # Plot depends on number of components
    if Y.shape[1]==2:
        ax = fig.add_subplot(*fig_pos)
        ax.scatter(Y[:,0], Y[:,1], c=cluster_labels, cmap=plt.cm.Spectral)
    elif Y.shape[1]==3:
        ax = fig.add_subplot(*fig_pos, projection='3d')
        ax.scatter(Y[:,0], Y[:,1], Y[:,2], c=cluster_labels, cmap=plt.cm.Spectral)
    else:
        raise ValueError('Too many components')
    ax.set_title('{}'.format(title))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')
    
############################################################

def main():
    #####################################################################
    # Load all data
    print('Loading data')
    # Original model of interest
    # parameters
    q_ids_name='q_ids_njdm'
    K=4
    alpha=0.001
    beta=0.001
    # Selected realization
    R=5
    r=1
    
    # Original result_dir
    original_result_dir='../results/{}/infer_MixtureModel_multipleDataSources/online/K_{}/alpha_{}/beta_{}/R_{}/r_{}'.format(q_ids_name, K, alpha, beta, R,r)

    # Participant ids
    with open('../data/{}/data_participant_ids'.format(q_ids_name), 'rb') as f:
        data_participant_ids=np.loadtxt(f).astype(int)
    # Emission details
    with open('../data/{}/allinone/f_emission.pickle'.format(q_ids_name), 'rb') as f:
        f_emission = pickle.load(f)
    # Data all in one
    with open('../data/{}/allinone/X.pickle'.format(q_ids_name), 'rb') as f:
        X=pickle.load(f)

    print('Pre-processing data')
    vocab=np.arange(f_emission[0]['d_x'])
    my_bins=np.arange(f_emission[0]['d_x']+1)
    S=X.shape[0]
    # Counts per vocab item
    X_count=np.zeros((S,f_emission[0]['d_x']))
    for s in np.arange(S):
        X_count[s],_=np.histogram(X[s,~np.isnan(X[s,:])], bins=my_bins)

    # Normalize
    X_density=X_count/X_count.sum(axis=1, keepdims=True)
    
    # Clustering alternatives
    # Parameters
    n_components = 2
    n_neighbors = 10
    n_clusters=4
    n_clusters=2

    # Candidate manifold methods
    LLE = partial(manifold.LocallyLinearEmbedding,
                  n_neighbors,
                  n_components,
                  eigen_solver='auto')
    methods = OrderedDict()
    methods['PCA'] = PCA(n_components=n_components)
    methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca')

    print('Clustering data')
    # Original model
    original_assignments=model_assignments(original_result_dir, data_participant_ids, 'hard')

    # K means in original space
    t0 = time.time()
    original_k_means_assignments=kmeans_assignments(X_density, data_participant_ids, n_clusters)
    t1 = time.time()
    print('K-means in original space: {:.3g}s'.format(t1-t0))
    
    # K means after manifold learning
    manifold_k_means_assignments = OrderedDict()
    # Create figure
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle('Clustering alternatives', fontsize=14)
    for i, (label, method) in enumerate(methods.items()):
        # Manifold learning
        t0 = time.time()
        Y = method.fit_transform(X_density)
        t1 = time.time()
        print('{} manifold learning: {:.3g}s'.format(label, t1-t0))

        # Low-dimensional k-means
        t0 = time.time()
        manifold_k_means_assignments[label]=kmeans_assignments(Y, data_participant_ids, n_clusters)
        t1 = time.time()
        print('K-means with learned {} manifold: {:.3g}s'.format(label, t1-t0))

        # Plotting
        # Just k-means
        add_low_dimensional_cluster_plot(fig, (2,2,(i+1)), Y, original_k_means_assignments['assigned_cluster'],
                'K-means in original space, shown via {}'.format(label))
        add_low_dimensional_cluster_plot(fig, (2,2,2+(i+1)), Y, manifold_k_means_assignments[label]['assigned_cluster'],
                'K-means in learned {} manifold'.format(label))

    # Show all!
    plot_save=True
    if plot_save:
        os.makedirs('../results/{}/clustering_alternatives'.format(q_ids_name), exist_ok=True)
        plt.savefig('../results/{}/clustering_alternatives/clustering_{}d.pdf'.format(q_ids_name,n_components), format='pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    #####################################################################    
    # Associations
    show_explanations=False
    #### WERF AND PHENDO DATAFRAME
    # WERF info file
    werf_file='../data/werf_and_profile_data/werf_survey.csv'
    # WERF to PHENDO MAPPING file
    werf_to_phendo_file='../data/werf_and_profile_data/phendoid_email_pid.csv'
    # Combine Werf and Phendo
    werf_data_to_phendo=load_werf_and_phendo_data(werf_file, werf_to_phendo_file)
    # Clean Werf and phendo dataframe
    werf_data_to_phendo=clean_werf_data_to_phendo(werf_data_to_phendo)
    
    #### PROFILE DATAFRAME
    # Profile info file
    profile_file='../data/werf_and_profile_data/profile.csv'
    # Age info file
    age_file='../data/werf_and_profile_data/participant_dob.csv'
    participant_profile=create_participant_profile_dataframe(profile_file, age_file)
    
    #########################################
    # ORIGINAL MODEL
    # Merge assignments with werf_data
    original_participant_assignment_werf=pd.merge(original_assignments, werf_data_to_phendo, how='inner', on='participant_id').astype({'participant_id':pd.api.types.CategoricalDtype()})
    # Merge assignments with partipant profile info
    original_participant_assignment_profile=pd.merge(original_assignments, participant_profile, how='inner', on='participant_id').astype({'participant_id':pd.api.types.CategoricalDtype()})
    # Merge assignments with werf_data
    original_participant_assignment_werf=pd.merge(original_assignments, werf_data_to_phendo, how='inner', on='participant_id').astype({'participant_id':pd.api.types.CategoricalDtype()})
    # Merge werf_data with phendo participant info
    original_werf_phendo_data=pd.merge(original_participant_assignment_werf, original_participant_assignment_profile, how='outer', on=['participant_id', 'assigned_cluster']).astype({'participant_id':pd.api.types.CategoricalDtype()})
    # Drop participant_id for associations
    original_werf_phendo_data.drop(labels=['participant_id'], axis='columns', inplace=True)
    
    #########################################
    # ALTERNATIVE CLUSTERING
    # K-means in original space
    # Merge assignments with werf_data
    original_k_means_participant_assignment_werf=pd.merge(original_k_means_assignments, werf_data_to_phendo, how='inner', on='participant_id').astype({'participant_id':pd.api.types.CategoricalDtype()})
    # Merge assignments with partipant profile info
    original_k_means_participant_assignment_profile=pd.merge(original_k_means_assignments, participant_profile, how='inner', on='participant_id').astype({'participant_id':pd.api.types.CategoricalDtype()})
    # Merge assignments with werf_data
    original_k_means_participant_assignment_werf=pd.merge(original_k_means_assignments, werf_data_to_phendo, how='inner', on='participant_id').astype({'participant_id':pd.api.types.CategoricalDtype()})
    # Merge werf_data with phendo participant info
    original_k_means_werf_phendo_data=pd.merge(original_k_means_participant_assignment_werf, original_k_means_participant_assignment_profile, how='outer', on=['participant_id', 'assigned_cluster']).astype({'participant_id':pd.api.types.CategoricalDtype()})
    # Drop participant_id for associations
    original_k_means_werf_phendo_data.drop(labels=['participant_id'], axis='columns', inplace=True)

    # Evaluate model and associations
    p_threshold=0.05
    print('###########################################################')
    print('#################### K-means in original space #######################')
    # Confusion matrix
    original_kmeans_merged=pd.merge(original_participant_assignment_werf,original_k_means_participant_assignment_werf, how='inner', on='participant_id').astype({'participant_id':pd.api.types.CategoricalDtype()})
    conf_matrix=confusion_matrix(original_kmeans_merged['assigned_cluster_x'],original_kmeans_merged['assigned_cluster_y'], labels=np.arange(K))
    purity=np.max(conf_matrix, axis=1).sum()/conf_matrix.sum()
    print('######## Original Vs learned confusion matrix with purity={} ############'.format(purity))
    print(conf_matrix)
    # Associations
    original_k_means_associations, original_k_means_explanations=compute_associations(original_k_means_werf_phendo_data)
    print_associations(original_k_means_werf_phendo_data, original_k_means_associations, original_k_means_explanations, p_threshold, show_explanations)
    original_k_means_associations_per_cluster, original_k_means_explanations_per_cluster=compute_associations_per_cluster(original_k_means_werf_phendo_data)
    print_associations_per_cluster(original_k_means_werf_phendo_data, original_k_means_associations_per_cluster, original_k_means_explanations_per_cluster, p_threshold, show_explanations)
    print('###########################################################')
    
    for i, (label, method) in enumerate(methods.items()):
        # K-means in learned manifold
        # Merge assignments with werf_data
        manifold_k_means_participant_assignment_werf=pd.merge(manifold_k_means_assignments[label], werf_data_to_phendo, how='inner', on='participant_id').astype({'participant_id':pd.api.types.CategoricalDtype()})
        # Merge assignments with partipant profile info
        manifold_k_means_participant_assignment_profile=pd.merge(manifold_k_means_assignments[label], participant_profile, how='inner', on='participant_id').astype({'participant_id':pd.api.types.CategoricalDtype()})
        # Merge assignments with werf_data
        manifold_k_means_participant_assignment_werf=pd.merge(manifold_k_means_assignments[label], werf_data_to_phendo, how='inner', on='participant_id').astype({'participant_id':pd.api.types.CategoricalDtype()})
        # Merge werf_data with phendo participant info
        manifold_k_means_werf_phendo_data=pd.merge(manifold_k_means_participant_assignment_werf, manifold_k_means_participant_assignment_profile, how='outer', on=['participant_id', 'assigned_cluster']).astype({'participant_id':pd.api.types.CategoricalDtype()})
        # Drop participant_id for associations
        manifold_k_means_werf_phendo_data.drop(labels=['participant_id'], axis='columns', inplace=True)

        # Evaluate model and associations
        print('###########################################################')
        print('########## K-means in learned {} manifold #################'.format(label))
        # Confusion matrix
        original_kmeans_manifold_merged=pd.merge(original_participant_assignment_werf,manifold_k_means_participant_assignment_werf, how='inner', on='participant_id').astype({'participant_id':pd.api.types.CategoricalDtype()})
        conf_matrix=confusion_matrix(original_kmeans_manifold_merged['assigned_cluster_x'],original_kmeans_manifold_merged['assigned_cluster_y'], labels=np.arange(K))
        purity=np.max(conf_matrix, axis=1).sum()/conf_matrix.sum()
        print('######## Original Vs learned confusion matrix with purity={} ############'.format(purity))
        print(conf_matrix)
        # Associations
        manifold_k_means_associations, manifold_k_means_explanations=compute_associations(manifold_k_means_werf_phendo_data)
        print_associations(manifold_k_means_werf_phendo_data, manifold_k_means_associations, manifold_k_means_explanations, p_threshold, show_explanations)
        manifold_k_means_associations_per_cluster, manifold_k_means_explanations_per_cluster=compute_associations_per_cluster(manifold_k_means_werf_phendo_data)
        print_associations_per_cluster(manifold_k_means_werf_phendo_data, manifold_k_means_associations_per_cluster, manifold_k_means_explanations_per_cluster, p_threshold, show_explanations)
        print('###########################################################')
    
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
