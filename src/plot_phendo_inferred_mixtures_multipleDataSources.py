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
from matplotlib.lines import Line2D
# WordCloud: https://github.com/amueller/word_cloud
from wordcloud import WordCloud
from PIL import ImageColor
import colorsys

# Import mixture model
from MixtureModel_multipleDataSources import *

# Globals for NJDM
data_dir='../data/q_ids_njdm'
q_ids_type=['pain_location', 'pain_description', 'pain_severity', 'symptoms','symptoms_severity','period_flow','bleeding','GI','GI_severity','sex','activities','day','med_hormones']
q_ids_type_full=['Where is the pain?', 'Describe the pain.', 'How severe is the pain?', 'What are you experiencing?','How severe is the symptom?','Describe your period flow.','What kind of bleeding.','Describe GI/GU system.','How severe is it (GI/GU)?','Describe sex.','Activities difficult to perform.','How was your day?','Medications/hormones taken.']
q_ids_selected=np.arange(13)    # All
my_colors=[colors.cnames['blue'], colors.cnames['darkblue'], colors.cnames['purple'], colors.cnames['darkgoldenrod'], colors.cnames['pink'], colors.cnames['red'], colors.cnames['deeppink'], colors.cnames['darkorange'], colors.cnames['saddlebrown'], colors.cnames['crimson'], colors.cnames['slategrey'], colors.cnames['turquoise'], colors.cnames['green'], colors.cnames['black']]
k_names=['A','B','C','D']
posterior_type='full'
posterior_type='top_vocab'
posterior_top_n=10
#posterior_type='top_med_vocab'

class DataSourceColorFunc(object):
    """ Create a color function object which assigns a specified color to words in data source
    
    Attributes:
        color_func_to_words: function mapping a color to the list of data source words.
        default_color_func : function to assign a color to a word that's not a member of the data source (safety check)
    """
    
    def get_my_color_func(self, color, color_value):
        # RGB of provided color
        r, g, b = ImageColor.getrgb(color)
        # if color_value=1
        if color_value==1:
            # Function returns always the same RGB color
            def single_color_func(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
                return 'rgb({:.0f}, {:.0f}, {:.0f})'.format(r, g, b)
        else:
            rgb_max = 255.
            #HSV of original RGB
            h, s, v = colorsys.rgb_to_hsv(r/rgb_max, g/rgb_max, b/rgb_max)
            #RGB with modified v
            r, g, b = colorsys.hsv_to_rgb(h, s, color_value)
            # Function returns the RGB color with color_value brightness
            def single_color_func(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
                return 'rgb({:.0f}, {:.0f}, {:.0f})'.format(rgb_max*r, rgb_max*g, rgb_max*b)
        return single_color_func

    def __init__(self, color_to_words, color_value, default_color):
        """ Init class attributes
        
        Args:
            color_to_words : dict(str -> list(str)), dictionary mapping a color (string) to the vocabulary (list) in data source.
            color_value: float within [0,1] indicating brightness/value of color as in https://en.wikipedia.org/wiki/HSL_and_HSV
            default_color : a color (string) assigned to a word that's not a member of the data source (safety check)
        """
        # Initialize functions
        self.color_func_to_words = [ (self.get_my_color_func(color,color_value), set(words)) for (color, words) in color_to_words.items()]
        self.default_color_func = self.get_my_color_func(default_color,color_value)

    def get_color_func(self, word):
        """ Returns a color_func associated with the word
        """
        try:
            # Find color_func for this word
            color_func = next( color_func for (color_func, words) in self.color_func_to_words if word in words )
        except StopIteration:
            # Resort to default
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)


# Main code
def main(exec_machine, inferred_model, data_source_vocabs, theta_cloud, plot_save):
    print('Inferred mixture for {}'.format(inferred_model))
    
    # Double-checking model
    assert os.path.isfile(inferred_model), 'Could not find mixture object {}'.format(inferred_model)
    
    # Load inferred object
    with open(inferred_model, 'rb') as f:
        inferredModel=pickle.load(f)
    
    # If plotting
    if plot_save != None:
        # Directory
        os.makedirs(plot_save, exist_ok=True)
    
    # Data sources and vocabularies
    assert inferredModel.D == len(data_source_vocabs), 'Number of data sources used for inference D={} does not match length of data source vocabularies={}'.format(inferredModel.D, len(data_source_vocabs))
    
    ### Word Cloud config
    if 'all' not in theta_cloud:
        wc_width=500
        wc_height=500
        wc_scale=1
        wc_max_words=10    # Most frequent words
        wc_prefer_horizontal=1.0    # Always fit horizontally
        wc_background_color=colors.cnames['white']    # Background color
        wc_relative_scaling=1.0    # Scaling between frequency and font
        wc_max_font=wc_height/1.0   # Max font
        wc_interpolation='bilinear'
    else:
        wc_width=1000
        wc_height=500
        wc_scale=1
        wc_max_words=100    # Most frequent words
        wc_prefer_horizontal=1.0    # Always fit horizontally
        wc_background_color=colors.cnames['white']    # Background color
        wc_relative_scaling=1.0    # Scaling between frequency and font
        wc_max_font=wc_height/4.0   # Max font
        wc_interpolation='bilinear'
        
    # Color to words for each data source vocab
    color_to_words={}
    vocab=[]
    ordered_vocab=[]
    vocab_order_mapping=[]
    fullvocab=[]
    #for d in np.arange(inferredModel.D):
    for (d_idx,d) in enumerate(q_ids_selected):
        # Load SELECTED data source vocabularies
        with open(data_source_vocabs[d]) as f:
            vocab.append(f.read().splitlines())
        
        # As well as ORDERED data source vocabularies
        with open(data_source_vocabs[d].replace('vocab','vocab_ordered')) as f:
            ordered_vocab.append(f.read().splitlines())
        
        # And figure out ordering map
        vocab_order_mapping.append(np.where(np.array(vocab[d_idx])[None,:]==np.array(ordered_vocab[d_idx])[:,None])[1])
        assert np.all(np.array(vocab[d_idx])[vocab_order_mapping[d_idx]]==np.array(ordered_vocab[d_idx]))
        
        # Color for words in this vocab
        color_to_words[my_colors[d_idx]]= vocab[d_idx]
        # Full flattened vocab 
        fullvocab+=vocab[d_idx]

    # Default color (safe check)
    default_color=colors.cnames['black']
    
    # Max brightness for most frequent: Colorfunc object with untouched color_value
    data_source_color_func = DataSourceColorFunc(color_to_words, 1., default_color)

    # Inferred mixture emission distribution parameters
    theta_kv={0:np.zeros((inferredModel.K, len(vocab)))}
    kl_theta=np.zeros((inferredModel.D, inferredModel.K))
    if isinstance(inferredModel, MixtureModel_multipleDataSources):
        print('Mixture Model inferred with K={} mixtures'.format(inferredModel.K))
        # Per data source
        for d in np.arange(inferredModel.D):
            theta_kv[d]=inferredModel.g_posterior[d]['beta']/(inferredModel.g_posterior[d]['beta'].sum(axis=1, keepdims=True))
            kl_theta[d]=(theta_kv[d] * np.log(theta_kv[d] * theta_kv[d].shape[1])).sum(axis=1)
    else:
        raise ValueError('Unknown inferred model type {}'.format(inferredModel))
    
    #for d in np.arange(inferredModel.D):        
    for (d_idx,d) in enumerate(q_ids_selected):
        # Topic vocabulary heatmap per SELECTED data source
        fig, ax = plt.subplots(1)
        if posterior_type=='top_vocab' and len(vocab[d_idx])>posterior_top_n:
            # Special posterior plotting limit vocabulary size
            sorted_vocab_probs=np.argsort(np.max(theta_kv[d].T,axis=1))
            top_prob_idx=sorted_vocab_probs[-posterior_top_n:]
            # Just top
            ordered_vocab_idx=top_prob_idx
            ordered_vocab=np.array(vocab[d_idx])[ordered_vocab_idx]
            # plot with new order
            cmap=plt.pcolormesh(theta_kv[d].T[ordered_vocab_idx], cmap='inferno', vmin=0., vmax=1.)
            fig.colorbar(cmap)
            # Put the major ticks at the middle of each cell
            k=np.arange(inferredModel.K)
            ax.set_xticks(k+ 0.5, minor=False)
            ax.set_yticks(np.arange(len(ordered_vocab)) + 0.5, minor=False)
            # X labels are phenotype number
            ax.set_xticklabels(k_names, minor=False)
            plt.xlabel('Phenotype')
            # Reduced vocab
            ax.set_yticklabels(ordered_vocab, minor=False)
        elif posterior_type=='top_med_vocab' and q_ids_type[d]=='med_hormones':
            # Special plotting for med_hormones, due to big vocabulary size
            # Top and low probability vocab indexes
            n_remove_irrelevant_meds=np.floor(len(vocab[d_idx])/2).astype(int)
            # Just threshold in top 30
            n_remove_irrelevant_meds=36
            sorted_vocab_probs=np.argsort(np.max(theta_kv[d].T,axis=1))
            top_prob_idx=sorted_vocab_probs[n_remove_irrelevant_meds:]
            low_prob_idx=sorted_vocab_probs[:n_remove_irrelevant_meds]
            # Just top
            ordered_vocab_idx=top_prob_idx
            ordered_vocab=np.array(vocab[d_idx])[ordered_vocab_idx]
            
            # plot with new order
            cmap=plt.pcolormesh(theta_kv[d].T[ordered_vocab_idx], cmap='inferno', vmin=0., vmax=1.)
            fig.colorbar(cmap)
            # Put the major ticks at the middle of each cell
            k=np.arange(inferredModel.K)
            ax.set_xticks(k+ 0.5, minor=False)
            ax.set_yticks(np.arange(len(ordered_vocab)) + 0.5, minor=False)
            # X labels are phenotype number
            ax.set_xticklabels(k_names, minor=False)
            plt.xlabel('Phenotype')
            # Reduced vocab
            ax.set_yticklabels(ordered_vocab, minor=False)
        else:
            # Topic vocabulary heatmap per SELECTED data source
            fig, ax = plt.subplots(1)
            cmap=plt.pcolormesh(theta_kv[d].T[vocab_order_mapping[d_idx]], cmap='inferno', vmin=0., vmax=1.)
            fig.colorbar(cmap)
            # Put the major ticks at the middle of each cell
            k=np.arange(inferredModel.K)
            ax.set_xticks(k+ 0.5, minor=False)
            ax.set_yticks(np.arange(len(vocab[d_idx])) + 0.5, minor=False)
            # X labels are phenotype number
            ax.set_xticklabels(k_names, minor=False)
            plt.xlabel('Phenotype')
            # Y labels are full vocab
            ax.set_yticklabels(np.array(vocab[d_idx])[vocab_order_mapping[d_idx]], minor=False)
        
        #plt.ylabel('v')
        #plt.title('Per-Topic vocabulary posterior heatmap')
        if plot_save is None: 
            plt.show()
        else:
            plt.savefig('{}/{}_{}_posterior.eps'.format(plot_save, q_ids_type[d], posterior_type), format='eps', bbox_inches='tight')
            plt.close()
    
    # Plot per mixture emission distributions
    for k in np.arange(inferredModel.K):
        # For all data sources
        all_theta_kv=np.zeros(0)
        all_cond_vocab=np.zeros(0)
        
        # If plotting wordclouds per data source
        if plot_save is None and 'all' not in theta_cloud:
            # Plot word could for each data source
            f, axarr = plt.subplots(inferredModel.D,1)

        # Per data source
        for (d_idx,d) in enumerate(q_ids_selected):
            # Max number of words, per data source
            wc_max_words=len(vocab[d_idx])
            # Wordcloud type
            if theta_cloud=='perd':
                # Max font for most frequent within d (for all k)
                wc_max_font_size=int(wc_max_font*(theta_kv[d][k].max())/(theta_kv[d].max()))
                # Per-topic theta dictionary
                topic_theta_dict=dict(zip(vocab[d_idx], theta_kv[d][k]))                
            elif theta_cloud=='perd_cond_mostprob':
                # Max font for most frequent within d (for all k)
                wc_max_font_size=int(wc_max_font*(theta_kv[d][k].max())/(theta_kv[d].max()))
                sorted_theta_idx=theta_kv[d][k].argsort()[::-1]
                most_theta_idx=sorted_theta_idx[theta_kv[d][k][sorted_theta_idx].cumsum()<=0.8]
                if most_theta_idx.size == 0:
                    most_theta_idx=sorted_theta_idx[theta_kv[d][k][sorted_theta_idx]>0.8]
                cond_theta=theta_kv[d][k][most_theta_idx]/theta_kv[d][k][most_theta_idx].sum()
                # Per-topic theta dictionary
                wc_max_words=cond_theta.size    # Words that have most mass
                topic_theta_dict=dict(zip(np.array(vocab[d_idx])[most_theta_idx], cond_theta/cond_theta.max()))
            elif theta_cloud=='all_perd':
                all_theta_kv=np.concatenate((all_theta_kv, theta_kv[d][k]/theta_kv[d].max()))
            elif theta_cloud=='all_perd_cond_mostprob':
                sorted_theta_idx=theta_kv[d][k].argsort()[::-1]
                most_theta_idx=sorted_theta_idx[theta_kv[d][k][sorted_theta_idx].cumsum()<=0.8]
                if most_theta_idx.size == 0:
                    most_theta_idx=sorted_theta_idx[theta_kv[d][k][sorted_theta_idx]>0.8]
                cond_theta=theta_kv[d][k][most_theta_idx]/theta_kv[d][k][most_theta_idx].sum()
                all_theta_kv=np.concatenate((all_theta_kv, cond_theta))
                all_cond_vocab=np.concatenate((all_cond_vocab, np.array(vocab[d_idx])[most_theta_idx]))
            else:
                raise ValueError('theta_cloud type {} not implemented yet'.format(theta_cloud))
                
            # If wordclouds per data source
            if 'all' not in theta_cloud:
                # Generate a word cloud per topic from thetas
                topic_theta_wc = WordCloud(width=wc_width, height=wc_height, scale=wc_scale, max_words=wc_max_words, prefer_horizontal=wc_prefer_horizontal, relative_scaling=wc_relative_scaling, background_color=wc_background_color, max_font_size=wc_max_font_size).generate_from_frequencies(topic_theta_dict, max_font_size=wc_max_font_size)
                # Apply per vocabulary color function
                topic_theta_wc.recolor(color_func=data_source_color_func)
                
                # Show or save
                if plot_save is None:
                    # Plot word could
                    if inferredModel.D>1:
                        axarr[d].imshow(topic_theta_wc, interpolation=wc_interpolation)
                        axarr[d].axis("off")
                    else:
                        axarr.imshow(topic_theta_wc, interpolation=wc_interpolation)
                        axarr.axis("off")
                else:
                    # Save wordcloud directly to file
                    topic_theta_wc.to_file(plot_save+'/wc_{}_k{}_{}.eps'.format(q_ids_type[d], k,theta_cloud))

        # If wordclouds per data source
        if 'all' not in theta_cloud:
            # Show or save
            if plot_save is None: 
                plt.show()
            #else:
                # Need to combine per d, done with latex for now
                
        
        # If overall wordclouds
        if 'all' in theta_cloud:
            # Change some WC properties
            wc_max_font_size=int(wc_max_font)   # Max size over number of data sources
            wc_max_words_all=40
            if all_cond_vocab.size>0:
                wc_max_words_all=all_cond_vocab.size
                fullvocab=all_cond_vocab
            # Generate word cloud per topic for all data sources
            topic_all_theta_dict=dict(zip(fullvocab, all_theta_kv))
            topic_all_theta_wc = WordCloud(width=wc_width, height=wc_height, scale=wc_scale, max_words=wc_max_words_all, relative_scaling=wc_relative_scaling, background_color=wc_background_color, max_font_size=wc_max_font_size).generate_from_frequencies(topic_all_theta_dict, max_font_size=wc_max_font_size)
            # Apply data source color function
            topic_all_theta_wc.recolor(color_func=data_source_color_func)

            # Plot or save
            if plot_save is None: 
                plt.figure()
                plt.imshow(topic_all_theta_wc, interpolation=wc_interpolation)
                plt.axis("off")
                plt.show()
            else:
                # Save wordcloud directly to file
                topic_all_theta_wc.to_file(plot_save+'/wc_k{}_{}.eps'.format(k,theta_cloud))
                # Corresponding custom legend
                custom_lines = [ Line2D([0], [0], color=my_colors[q], lw=2) for q in np.arange(len(q_ids_type_full))]
                plt.figlegend(custom_lines, q_ids_type_full, loc='center', ncol=1, shadow=False)
                plt.axis('tight')
                plt.axis('off')
                plt.savefig(plot_save+'/wc_k{}_legend.eps'.format(k), format='eps', bbox_inches='tight', pad_inches=-0.5)
                plt.close()

    # Topics per patient heatmap
    N_sk=np.zeros((inferredModel.S, inferredModel.K))
    # Iterate over sets for plotting
    for s in np.arange(inferredModel.S):
        k_Z, count_K=np.unique(inferredModel.Z[s,~np.isnan(inferredModel.Z[s,:])], return_counts=True)
        N_sk[s,k_Z.astype(int)]=count_K

    # Observation counts per participant and question
    with open('{}/n_observations_matrix.pickle'.format(data_dir), 'rb') as f:
        n_observations_matrix=pickle.load(f)
    # Day counts per participant and question
    with open('{}/n_days_matrix.pickle'.format(data_dir), 'rb') as f:
        n_days_matrix=pickle.load(f)

    # Compute summary counts
    n_observations_overall=n_observations_matrix.sum(axis=1)
    n_days_overall=n_days_matrix.sum(axis=1)
    # Reviewer suggested metrics
    n_observations_times_days_overall=n_observations_overall*n_days_overall
    n_observations_per_day_overall=n_observations_overall/n_days_overall
    
    # Empirical posterior
    emp_posterior=N_sk/N_sk.sum(axis=1, keepdims=True)
    # Plot heatmap
    fig, ax = plt.subplots(1)
    # With mixtures ordered by topic assignment
    order_by_topic_assignment=np.lexsort(emp_posterior.T.tolist())
    cmap=plt.pcolormesh(emp_posterior[order_by_topic_assignment], cmap='inferno', vmin=0., vmax=1.)
    fig.colorbar(cmap)
    # Put the major ticks at the middle of each cell
    k=np.arange(inferredModel.K)
    ax.set_xticks(k+ 0.5, minor=False)
    # X labels are phenotype number
    ax.set_xticklabels(k_names, minor=False)
    plt.xlabel('Phenotype')
    plt.ylabel('p_id')
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/participant_posterior_order_by_topic_assignment.eps', format='eps', bbox_inches='tight')
        plt.close()

    # Plot n_observations heatmap
    fig, ax = plt.subplots(1)
    # With mixtures ordered by topic assignment
    cmap=plt.pcolormesh(n_observations_overall[order_by_topic_assignment][:,None], cmap='inferno')
    fig.colorbar(cmap)
    ax.set_xticks(np.array([0.5]), minor=False)
    ax.set_xticklabels('n_obs', minor=False)
    plt.ylabel('p_id')
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/participant_n_observations_order_by_topic_assignment.eps', format='eps', bbox_inches='tight')
        plt.close()
    
    # Plot n_days heatmap
    fig, ax = plt.subplots(1)
    # With mixtures ordered by topic assignment
    cmap=plt.pcolormesh(n_days_overall[order_by_topic_assignment][:,None], cmap='inferno')
    fig.colorbar(cmap)
    ax.set_xticks(np.array([0.5]), minor=False)
    ax.set_xticklabels('n_obs', minor=False)
    plt.ylabel('p_id')
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/participant_n_days_order_by_topic_assignment.eps', format='eps', bbox_inches='tight')
        plt.close()
        
    # Plot n_observations times days heatmap
    fig, ax = plt.subplots(1)
    # With mixtures ordered by topic assignment
    cmap=plt.pcolormesh(n_observations_times_days_overall[order_by_topic_assignment][:,None], cmap='inferno')
    fig.colorbar(cmap)
    ax.set_xticks(np.array([0.5]), minor=False)
    ax.set_xticklabels(r'n_obs $\times$ n_days', minor=False)
    plt.ylabel('p_id')
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/participant_n_observations_times_days_order_by_topic_assignment.eps', format='eps', bbox_inches='tight')
        plt.close()
    
    # Plot n_observations per day heatmap
    fig, ax = plt.subplots(1)
    # With mixtures ordered by topic assignment
    cmap=plt.pcolormesh(n_observations_per_day_overall[order_by_topic_assignment][:,None], cmap='inferno')
    fig.colorbar(cmap)
    ax.set_xticks(np.array([0.5]), minor=False)
    ax.set_xticklabels('n_obs per day', minor=False)
    plt.ylabel('p_id')
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/participant_n_observations_per_day_order_by_topic_assignment.eps', format='eps', bbox_inches='tight')
        plt.close()
    
    # Ordered by number of observations
    order_by_n_observations=np.lexsort(n_observations_overall[:,None].T.tolist())
    # Plot Empirical posterior
    fig, ax = plt.subplots(1)
    cmap=plt.pcolormesh(emp_posterior[order_by_n_observations], cmap='inferno', vmin=0., vmax=1.)
    fig.colorbar(cmap)
    # Put the major ticks at the middle of each cell
    k=np.arange(inferredModel.K)
    ax.set_xticks(k+ 0.5, minor=False)
    # X labels are phenotype number
    ax.set_xticklabels(k_names, minor=False)
    plt.xlabel('Phenotype')
    plt.ylabel('p_id')
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/participant_posterior_order_by_n_observations.eps', format='eps', bbox_inches='tight')
        plt.close()
    
    # Ordered by number of days
    order_by_n_days=np.lexsort(n_days_overall[:,None].T.tolist())
    # Plot Empirical posterior
    fig, ax = plt.subplots(1)
    cmap=plt.pcolormesh(emp_posterior[order_by_n_days], cmap='inferno', vmin=0., vmax=1.)
    fig.colorbar(cmap)
    # Put the major ticks at the middle of each cell
    k=np.arange(inferredModel.K)
    ax.set_xticks(k+ 0.5, minor=False)
    # X labels are phenotype number
    ax.set_xticklabels(k_names, minor=False)
    plt.xlabel('Phenotype')
    plt.ylabel('p_id')
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/participant_posterior_order_by_n_days.eps', format='eps', bbox_inches='tight')
        plt.close()
    
    # Ordered by number of observations times days
    order_by_n_observations_times_days=np.lexsort(n_observations_times_days_overall[:,None].T.tolist())
    # Plot Empirical posterior
    fig, ax = plt.subplots(1)
    cmap=plt.pcolormesh(emp_posterior[order_by_n_observations_times_days], cmap='inferno', vmin=0., vmax=1.)
    fig.colorbar(cmap)
    # Put the major ticks at the middle of each cell
    k=np.arange(inferredModel.K)
    ax.set_xticks(k+ 0.5, minor=False)
    # X labels are phenotype number
    ax.set_xticklabels(k_names, minor=False)
    plt.xlabel('Phenotype')
    plt.ylabel('p_id')
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/participant_posterior_order_by_n_observations_times_days.eps', format='eps', bbox_inches='tight')
        plt.close()
    
    # Ordered by number of observations per day
    order_by_n_observations_per_day=np.lexsort(n_observations_per_day_overall[:,None].T.tolist())
    # Plot Empirical posterior
    fig, ax = plt.subplots(1)
    cmap=plt.pcolormesh(emp_posterior[order_by_n_observations_per_day], cmap='inferno', vmin=0., vmax=1.)
    fig.colorbar(cmap)
    # Put the major ticks at the middle of each cell
    k=np.arange(inferredModel.K)
    ax.set_xticks(k+ 0.5, minor=False)
    # X labels are phenotype number
    ax.set_xticklabels(k_names, minor=False)
    plt.xlabel('Phenotype')
    plt.ylabel('p_id')
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/participant_posterior_order_by_n_observations_per_day.eps', format='eps', bbox_inches='tight')
        plt.close()
          
    print('DONE plotting')
    
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    parser = argparse.ArgumentParser(description='Plot inferred mixtures for some observations with multiple data sources and a mixture model')
    parser.add_argument('-exec_machine', type=str, default='laptop', help='Where to run the simulation') 
    parser.add_argument('-inferred_model', type=str, help='Path to inferred mixture object')
    parser.add_argument('-data_source_vocabs', nargs='+', type=str, help='Path to data sources vocabulary files')
    parser.add_argument('-theta_cloud', type=str, default='overall', help='Whether theta word cloud is drawn perkd, perd or overall') 
    parser.add_argument('-plot_save', type=str, default=None, help='None or Save')

    # Get arguments
    args = parser.parse_args()
        
    # Call main function
    main(args.exec_machine, args.inferred_model, args.data_source_vocabs, args.theta_cloud, args.plot_save)
