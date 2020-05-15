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

# Add useful functions
from werf_phendo_questionnaire_preprocessing_functions import *
# Import mixture models
from MixtureModel_multipleDataSources import *

############################################################
def compute_associations(werf_phendo_data):
    # Assocation results
    associations=np.zeros((werf_phendo_data.columns.size, 2)) # statistic and p_value
    # Explanation results
    explanations=[]
    # Cluster size
    cluster_size=werf_phendo_data['assigned_cluster'].cat.categories.size
    # iterating over columns (of interest)
    for (column_idx, column) in enumerate(werf_phendo_data.columns):
        #print(column_idx, column, werf_phendo_data[column].dtype)
        # Categorical?
        if werf_phendo_data[column].dtype.str=='|O08':  #pd.api.types.CategoricalDtype() in string
            contingency_table=pd.crosstab(werf_phendo_data['assigned_cluster'],werf_phendo_data[column], dropna=False)
            if contingency_table.size >0:
                if cluster_size==2 and werf_phendo_data[column].cat.categories.size == 2:
                    # Exact Fisher
                    # 2 by 2 contingency table, ok with small samples
                    associations[column_idx, 0], associations[column_idx, 1] = stats.fisher_exact(contingency_table)
                    explanations.append(contingency_table)
                else:
                    # Chi-square for at least two groups
                    # any contingency table: for the validity of this calculation is that the test should be used only if the observed and expected frequencies in each cell are at least 5.
                    # As it is, it computes Pearson’s chi-squared statistic
                    #chi2_stat, p_value, dof, e_freq = stats.chi2_contingency(contingency_table)
                    try:
                        (associations[column_idx, 0], associations[column_idx, 1], _, expected_numbers)=stats.chi2_contingency(contingency_table)
                        explanations.append(contingency_table)
                    except:
                        associations[column_idx, :]=np.nan
                        explanations.append((np.nan))
            else:
                associations[column_idx, :]=np.nan
                explanations.append((np.nan))
        # Continous?
        elif werf_phendo_data[column].dtype.str=='<f8': #float64 in string
            #print('Processing dtype={} in {}:{}'.format(werf_phendo_data[column].dtype, column_idx, column))
            # Get the arrays of interest (without nans) per group
            per_group=werf_phendo_data[['assigned_cluster',column]].dropna(subset=[column]).groupby('assigned_cluster')[column].apply(np.array)
            
            if cluster_size==2:
                # Welch's t-test, two-sample test which does not assume equal population variance
                #t_stat, p_value = stats.ttest_ind(per_group[0], per_group[1], equal_var = False)
                associations[column_idx, 0], associations[column_idx, 1] = stats.ttest_ind(per_group[0], per_group[1], equal_var = False)
                explanations.append(pd.DataFrame(data={'mean':per_group.apply(np.mean), 'std':per_group.apply(np.std)}, index=per_group.index))
            else:
                # Kruskal-Wallis H-test:
                # tests the null hypothesis that the population median of all of the groups are equal
                # It is a non-parametric version of ANOVA.
                # The test works on 2 or more independent samples, which may have different sizes. 
                # kruskal_stat, p_value = stats.kruskal(*per_group.tolist(), nan_policy = 'raise') # or 'omit'?
                associations[column_idx, 0], associations[column_idx, 1] = stats.kruskal(*per_group.tolist(), nan_policy = 'raise') # or 'omit'?
                explanations.append(pd.DataFrame(data={'mean':per_group.apply(np.mean), 'std':per_group.apply(np.std)}, index=per_group.index))
        else:
            print('Unexpected dtype={} ({}) in {}:{}'.format(werf_phendo_data[column].dtype.str, werf_phendo_data[column].dtype, column_idx, column))

    return associations, explanations

def compute_associations_per_cluster(werf_phendo_data):
    # Assocations per cluster
    cluster_size=werf_phendo_data['assigned_cluster'].cat.categories.size
    associations_per_cluster=np.zeros((cluster_size, werf_phendo_data.columns.size, 2)) # statistic and p_value
    explanations_per_cluster=[]

    for cluster in werf_phendo_data['assigned_cluster'].cat.categories.values:
        this_cluster_explanations=[]
        # Separate in cluster and out of cluster data
        in_cluster=werf_phendo_data[werf_phendo_data['assigned_cluster']==cluster]
        out_cluster=werf_phendo_data[werf_phendo_data['assigned_cluster']!=cluster]
        # iterating over columns (of interest)
        for (column_idx, column) in enumerate(werf_phendo_data.columns):
            # Categorical?
            if werf_phendo_data[column].dtype.str=='|O08':  #pd.api.types.CategoricalDtype() in string
                contingency_table=np.zeros((2,werf_phendo_data[column].cat.categories.values.size))
                contingency_table[0,:]=in_cluster[column].value_counts()[in_cluster[column].cat.categories.values]
                contingency_table[1,:]=out_cluster[column].value_counts()[out_cluster[column].cat.categories.values]
                if contingency_table.size>0 and np.all(contingency_table.sum(axis=0)>0):
                    if werf_phendo_data[column].cat.categories.size == 2:
                        # Exact Fisher
                        # 2 by 2 contingency table, ok with small samples
                        # prior_oddsratio, p_value = stats.fisher_exact(contingency_table)
                        associations_per_cluster[cluster,column_idx, 0], associations_per_cluster[cluster,column_idx, 1] = stats.fisher_exact(contingency_table)
                        this_cluster_explanations.append(pd.DataFrame(contingency_table, columns=werf_phendo_data[column].cat.categories.values, index=['in_cluster','out_cluster']))
                    else:
                        # Chi-square for at least two groups
                        # any contingency table: for the validity of this calculation is that the test should be used only if the observed and expected frequencies in each cell are at least 5.
                        # As it is, it computes Pearson’s chi-squared statistic
                        # chi2_stat, p_value, dof, e_freq = stats.chi2_contingency(contingency_table)
                        try:
                            (associations_per_cluster[cluster,column_idx, 0], associations_per_cluster[cluster,column_idx, 1], _, expected_numbers )=stats.chi2_contingency(contingency_table)
                            this_cluster_explanations.append((np.nan) if np.isnan(expected_numbers).all() else pd.DataFrame(contingency_table, columns=werf_phendo_data[column].cat.categories.values, index=['in_cluster','out_cluster']))
                        except:
                            associations_per_cluster[cluster,column_idx, :]=np.nan
                            this_cluster_explanations.append((np.nan))
                else:
                    associations_per_cluster[cluster,column_idx, :]=np.nan
                    this_cluster_explanations.append((np.nan))
            # Continous?
            elif werf_phendo_data[column].dtype.str=='<f8': #float64 in string
                # Get the arrays of interest (without nans) per group
                per_group=[np.array(in_cluster[column].dropna()), np.array(out_cluster[column].dropna())]
                # Welch's t-test, two-sample test which does not assume equal population variance
                # t_stat, p_value = stats.ttest_ind(per_group[0], per_group[1], equal_var = False)
                associations_per_cluster[cluster,column_idx, 0], associations_per_cluster[cluster,column_idx, 1] = stats.ttest_ind(per_group[0], per_group[1], equal_var = False)
                this_cluster_explanations.append(pd.DataFrame(data={'mean':[per_group[0].mean(), per_group[1].mean()], 'std':[per_group[0].std(), per_group[1].std()]}, index=['in_cluster', 'out_cluster']))
            else:
                print('Unexpected dtype={} ({}) in {}:{}'.format(werf_phendo_data[column].dtype.str, werf_phendo_data[column].dtype, column_idx, column))
        
        # Add explanations
        explanations_per_cluster.append(this_cluster_explanations)
        
    return associations_per_cluster, explanations_per_cluster  

def print_associations(werf_phendo_data, associations, explanations, p_threshold=0.05, show_explanations=True):
    # Ordered associations
    order_of_associations=associations[:,1].argsort()
    print('############ Ordered associations for {} participants ################'.format(werf_phendo_data.shape[0]))
    #for value in werf_phendo_data.columns[associations[:,1]<p_threshold].values: print(value)
    for idx in order_of_associations:
        if not np.isnan(associations[idx,1]):
            if associations[idx,1]<p_threshold:
                print('### {} with p-value {} ###'.format(werf_phendo_data.columns.values[idx], associations[idx,1]))
                if show_explanations:
                    print(explanations[idx])
                print('##########')
                
def print_associations_per_cluster(werf_phendo_data, associations_per_cluster, explanations_per_cluster, p_threshold=0.05, show_explanations=True):
    for cluster in werf_phendo_data['assigned_cluster'].cat.categories.values:
        # Ordered associations
        order_of_associations_per_cluster=associations_per_cluster[cluster,:,1].argsort()
        print('####### Ordered associations for cluster {} #######'.format(cluster))
        for idx in order_of_associations_per_cluster:
            if not np.isnan(associations_per_cluster[cluster,idx,1]):
                if associations_per_cluster[cluster,idx,1]<p_threshold:
                    print('### {} with p-value {} ###'.format(werf_phendo_data.columns.values[idx], associations_per_cluster[cluster,idx,1]))
                    if show_explanations:
                        print(explanations_per_cluster[cluster][idx])
                    print('##########')


#####################################################################
def main():
    #### WERF AND PHENDO DATAFRAME
    # WERF info file
    werf_file='../data/werf_and_profile_data/werf_survey.csv'
    # WERF to PHENDO MAPPING file
    werf_to_phendo_file='../data/werf_and_profile_data/phendoid_email_pid.csv'
    # Combine Werf and Phendo
    werf_data_to_phendo=load_werf_and_phendo_data(werf_file, werf_to_phendo_file)
    # Clean Werf and phendo dataframe
    werf_data_to_phendo=clean_werf_data_to_phendo(werf_data_to_phendo)

    #####################################################################
    #### PROFILE DATAFRAME
    # Profile info file
    profile_file='../data/werf_and_profile_data/profile.csv'
    # Age info file
    age_file='../data/werf_and_profile_data/participant_dob.csv'
    participant_profile=create_participant_profile_dataframe(profile_file, age_file)
    
    #####################################################################
    ### MODEL ASSIGNMENTS
    # From what simulation
    q_ids_name='q_ids_njdm'
    K=4
    alpha=0.001
    beta=0.001
    R=5
    r=1
    
    # Participant ids
    with open('../data/{}/data_participant_ids'.format(q_ids_name), 'rb') as f:
        data_participant_ids=np.loadtxt(f).astype(int)

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

    # Assignment type
    assignments='clear'
    assignments='hard'

    if assignments == 'clear':
        #### Clear cluster assignments
        # Cluster assignment threshold
        assignment_threshold=0.9
        threshold_assignments=(emp_posterior>assignment_threshold).sum(axis=1)
        clear_assignments=np.argmax(emp_posterior[threshold_assignments==1,:], axis=1)

        participant_assignments=pd.DataFrame(data={'participant_id': data_participant_ids[threshold_assignments.astype(bool)], 'assigned_cluster': clear_assignments}, dtype=pd.api.types.CategoricalDtype())

    elif assignments == 'hard':
        #### HARD cluster assignments
        hard_assignments=np.argmax(emp_posterior, axis=1)
        participant_assignments=pd.DataFrame(data={'participant_id': data_participant_ids, 'assigned_cluster': hard_assignments}, dtype=pd.api.types.CategoricalDtype())

    else:
        raise ValueError('Unknown {} assignment'.format(assignments))

    #####################################################################
    os.makedirs('../data/{}/associations/{}_assignments'.format(q_ids_name,assignments), exist_ok=True)
    #### MERGING AND SELECTING COHORT
    # Merge assignments with partipant profile info
    participant_assignment_profile=pd.merge(participant_assignments, participant_profile, how='inner', on='participant_id').astype({'participant_id':pd.api.types.CategoricalDtype()})
    # Merge assignments with werf_data
    participant_assignment_werf=pd.merge(participant_assignments, werf_data_to_phendo, how='inner', on='participant_id').astype({'participant_id':pd.api.types.CategoricalDtype()})
    # Merge werf_data with phendo participant info
    werf_phendo_data=pd.merge(participant_assignment_werf, participant_assignment_profile, how='outer', on=['participant_id', 'assigned_cluster']).astype({'participant_id':pd.api.types.CategoricalDtype()})
    # Drop participant_id for associations
    werf_phendo_data.drop(labels=['participant_id'], axis='columns', inplace=True)
    # Dimensionalities
    print('Total Profile responses: {}'.format(participant_profile.shape[0]))
    print('Total WERF responses: {}'.format(werf_data_to_phendo.shape[0]))
    print('Total Participants with assignments: {}'.format(participant_assignments.shape[0]))
    print('Total Participants with completed WERF and {} assignments: {}'.format(assignments, participant_assignment_werf.shape[0]))
    print('Total Participants with completed Profile and {} assignments: {}'.format(assignments, participant_assignment_profile.shape[0]))
    print('Total Participants with completed WERF and/or Profile and {} assignments: {}'.format(assignments, werf_phendo_data.shape[0]))
    
    #####################################################################
    # Whether to show association explanations
    show_explanations=False
    # Compute associations
    p_threshold=0.05
    print('###########################################################')
    print('#################### ORIGINAL model #######################')
    associations, explanations=compute_associations(werf_phendo_data)
    print_associations(werf_phendo_data, associations, explanations, p_threshold, show_explanations)
    associations_per_cluster, explanations_per_cluster=compute_associations_per_cluster(werf_phendo_data)
    print_associations_per_cluster(werf_phendo_data, associations_per_cluster, explanations_per_cluster, p_threshold, show_explanations)
    print('###########################################################')
    #####################################################################
    
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
