#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
import scipy.special as special
from scipy.special import factorial
import matplotlib.pyplot as plt
import pickle
import pdb
import time

# Class definitions
class MixtureModel_multipleDataSources(object):
    """General Class for Mixture models with multiple data sources
        Following section 2.5 in report /phenotyping/doc/phenotyping.pdf

    Attributes:
        K: number of mixture components
        m_prior: prior distribution of the mixture components
            m_prior['dist']: mixture prior distribution
            m_prior['alpha']: hyperparameter of the prior distribution on mixture components
        m_distribution: distribution of the mixture components
            m_distribution['dist']: mixture distribution (Categorical by default)
            m_distribution['phi']: parameters of the mixture distribution
        D: number of data sources
        g_prior: prior distribution of the emission parameters (array of dictionaries)
            g_prior[d]['dist']: prior distribution density for data source d
            g_prior[d]['Theta']: prior distribution hyperparameters for data source d
        f_emission: emission distribution of the observations (array of dictionaries)
            f_emission[d]['dist']: emission density for data source d
            f_emission[d]['d_x']: dimensionality of the data source d observations 
            f_emission[d]['theta']: parameters of the emission density for data source d (provided or sampled from prior)
    """
    
    def __init__(self, K, m_prior, m_distribution, D, g_prior, f_emission):
        """ Initialize the Mixture Model object and its attributes
        
        Args:
            K: number of mixture components
            m_prior: prior distribution of the mixture components
                m_prior['dist']: mixture prior distribution
                m_prior['alpha']: hyperparameter of the prior distribution on mixture components
            m_distribution: distribution of the mixture components
                m_distribution['dist']: mixture distribution (Categorical by default)
                m_distribution['phi']: parameters of the mixture distribution (sampled from m_prior by default)
            D: number of data sources
            g_prior: prior distribution of the emission parameters (array of dictionaries)
                g_prior[d]['dist']: prior distribution density for data source d
                g_prior[d]['Theta']: prior distribution hyperparameters for data source d
            f_emission: emission distribution of the observations (array of dictionaries)
                f_emission[d]['dist']: emission density for data source d
                f_emission[d]['d_x']: dimensionality of the data source d observations 
                f_emission[d]['theta']: parameters of the emission density for data source d (provided or sampled from prior)
        """
        
        # Number of mixtures
        self.K=K
        
        # Prior mixture distribution
        assert 'dirichlet' in m_prior['dist'].__str__(), 'Mixture prior density must be dirichlet'
        self.m_prior=m_prior
        # Mixture posterior
        self.m_posterior=None
        
        # Mixture distribution
        if m_distribution is None:
            # Default is categorical
            self.m_distribution={'dist': stats.multinomial}
        else:
            assert 'multinomial' in m_distribution['dist'].__str__(), 'Mixture distribution must be categorical'
            self.m_distribution=m_distribution       
               
        # Number of data sources
        self.D=D
        
        # Prior emission parameter distribution
        assert g_prior.size==self.D, 'Prior information array size {} does not match number of data sources {}'.format(g_prior.size, self.D) 
        self.g_prior=g_prior
        for d in np.arange(self.D):
            if 'dirichlet' in self.g_prior[d]['dist'].__str__():
                # Precompute
                self.g_prior[d]['beta_sum']=self.g_prior[d]['beta'].sum()
            else:
                raise ValueError('g_prior distribution {} not implemented yet'.format(self.g_prior[d]['dist']))

        # Emission distribution
        assert f_emission.size==self.D, 'Emission distribution array size {} does not match number of data sources {}'.format(f_emission.size, self.D)
        self.f_emission=f_emission
                
    def generate_data(self, N_sd):
        """ Run the generative mixture model and create observations X for D data sources and mixture assignments Z
        
        Args:
            N_sd: array with number of obsevations per set s of type d
        """
        
        # Group/set dimensions
        self.N_sd=N_sd
        self.N_s=self.N_sd.sum(axis=1)
        self.S=self.N_s.size
        
        # Allocate X, XD, and Z with NaNs
        self.X=np.NaN*np.ones((self.S, np.max(self.N_s)))   # X[s,n]=value (real or vocabulary number)
        self.XD=np.NaN*np.ones((self.S, np.max(self.N_s)))  # XD[s,n]=data type d for X[s,n]
        self.Z=np.NaN*np.ones((self.S, np.max(self.N_s)))   # Z[s,n]=mixture k
        
        # Share emission parameters across sets, for each data type and mixture component
        for d in np.arange(self.D):
            if 'dirichlet' in self.g_prior[d]['dist'].__str__():
                self.f_emission[d]['theta']=self.g_prior[d]['dist'].rvs(self.g_prior[d]['beta'], size=self.K)
            else:
                raise ValueError('g_prior distribution {} not implemented yet'.format(self.g_prior[d]['dist']))
                    
        # Allocate mixture parameter space if needed
        if not('phi' is self.m_distribution.keys()):
            self.m_distribution['phi']=np.zeros((self.S,self.K))
        
        # For each set of observations
        for s in np.arange(self.S):
            # Sample mixture proportions with shared hyperparameters, if not provided
            if self.m_distribution['phi'][s].sum() == 0:
                self.m_distribution['phi'][s]=self.m_prior['dist'].rvs(self.m_prior['alpha'])
                
            # Generate z_{n,s}
            n_s,v_s=np.where(stats.multinomial.rvs(1,self.m_distribution['phi'][s], size=self.N_s[s]))
            self.Z[s,n_s]=v_s
                      
            # And x_{n,s} given z_{n,s} and d_{n,s}
            n_s=0
            for d in np.arange(self.D):
                for n_d in np.arange(self.N_sd[s,d]):
                    # Generate d_{n,s} (in simple order)
                    self.XD[s,n_s]=d
                    # Selected mixture Z[s,n]=k and data source d
                    if 'multinomial' in self.f_emission[d]['dist'].__str__():
                        self.X[s,n_s]=np.where(self.f_emission[d]['dist'].rvs(1,self.f_emission[d]['theta'][int(self.Z[s,n_s])]))[0][0]
                    else:
                        raise ValueError('f_emission distribution {} not implemented yet'.format(self.f_emission[d]['dist']))

                    # Ready for next observation
                    n_s+=1
        
        # True posterior
        self.__compute_posterior()
        # True likelihood
        print('True: log p(X|Z)={}, log p(Z)={}'.format(*(self.compute_loglikelihood())))

    def initialize_with_data(self, X, XD):
        """ Initialize the model with data

        Args:
            X: S by N_s array of observations
            XD: S by N_s array of observations data source
        """

        # Data to work with with
        assert np.all(~np.isnan(XD) == ~np.isnan(X)), 'NaN discrepancy between XD and X'
        
        ############## INITIALIZATION ################
        if not hasattr(self, 'S') and not hasattr(self, 'N_s') and not hasattr(self, 'N_sd') and not hasattr(self, 'X') and not hasattr(self, 'XD') and not hasattr(self, 'Z') :            
            # Figure out dimensions
            self.S=X.shape[0]
            self.N_s=(~np.isnan(X)).sum(axis=1)
            self.N_sd=np.zeros((self.S, self.D))
            for d in np.arange(self.D):
                self.N_sd[:,d]=(XD==d).sum(axis=1)
            
            # Data to work with with
            self.X=X[:self.S, :np.max(self.N_s)]
            self.XD=XD[:self.S, :np.max(self.N_s)]
             
            # Random mixture assignments
            self.Z=np.NaN*np.ones((self.S, np.max(self.N_s)))
            for s in np.arange(self.S):
                # By sampling from prior with hyperparameters
                n_s,v_s=np.where(stats.multinomial.rvs(1,self.m_prior['alpha']/(self.m_prior['alpha'].sum()), size=self.N_s[s]))
                self.Z[s,n_s]=v_s

        else:
            # Already initialized, just doublecheck
            # Correct dimensionalities
            assert self.S == self.X.shape[0]
            assert np.all(self.N_s == (~np.isnan(self.X)).sum(axis=1))
            assert self.N_sd.shape == (self.S, self.D)
            for d in np.arange(self.D):
                assert np.all(self.N_sd[:,d] == (self.XD==d).sum(axis=1))
            
            # We are working with same data
            assert np.all(~np.isnan(self.X) == ~np.isnan(X))
            assert np.all(~np.isnan(self.XD) == ~np.isnan(XD))
            assert np.all(~np.isnan(self.X) == ~np.isnan(self.XD))
            assert np.all(~np.isnan(self.X) == ~np.isnan(self.Z))
            np.testing.assert_equal(self.X, X, 'Not same data!')
            np.testing.assert_equal(self.XD, XD, 'Not same data-sources info!')
        
        
    def run_gibbs_inference(self, X, XD, gibbs, save_dir, p_computation='general'):
        """ Run a Gibbs sampler (with collapsed distributions) for inference of assignments Z, given observations X and their data sources XD
        
        Args:
            X: S by N_s array of observations
            XD: S by N_s array of observations data source
            gibbs: parameters for general assignment Gibbs sampler
                gibbs['max_iter']: maximum number of Gibbs sampling iterations to run
                gibbs['loglik_eps']: minimum relative difference on loglikelihood between iterations
                gibbs['burn_in']: number of Gibbs samples to skip before estimation
                gibbs['lag']: number of Gibbs samples to skip for estimation
            save_dir: directory where to place temporary results
            p_computation: whether to compute posteriors with general form or as online computation
        """
        
        # Make sure model is initizalized
        self.initialize_with_data(X, XD)
        ##############  GIBBS SAMPLING ################
        t_init=time.process_time()
        # Parameters
        self.gibbs=gibbs
        # Summary statistics
        self.gibbs_data={'XcondXDZ_loglik':-np.inf*np.ones(self.gibbs['max_iter']+1), 'Z_loglik':-np.inf*np.ones(self.gibbs['max_iter']+1), 'delta_time':-np.inf*np.ones(self.gibbs['max_iter']+1)}
        XZ_loglik=-np.inf
        # Initial statistics
        n_iter=1
        # Compute posterior
        self.__compute_posterior()
        
        # Save and print statistics
        # likelihood
        (self.gibbs_data['XcondXDZ_loglik'][n_iter], self.gibbs_data['Z_loglik'][n_iter])=self.compute_loglikelihood()
        # Execution time
        self.gibbs_data['delta_time'][n_iter]=time.process_time()-t_init
        print('n_iter={} in {} with log p(X,Z)={}, log p(X|XD,Z)={}, log p(Z)={}'.format(n_iter, self.gibbs_data['delta_time'][n_iter], self.gibbs_data['XcondXDZ_loglik'][n_iter]+self.gibbs_data['Z_loglik'][n_iter], self.gibbs_data['XcondXDZ_loglik'][n_iter], self.gibbs_data['Z_loglik'][n_iter]))

        # iterations with max_iter hard limit or no likelihood improvement bigger than epsilon
        while (n_iter < self.gibbs['max_iter'] and abs(self.gibbs_data['XcondXDZ_loglik'][n_iter]+self.gibbs_data['Z_loglik'][n_iter] - XZ_loglik) >= (self.gibbs['loglik_eps']*abs(XZ_loglik))):
            # Update variables
            XZ_loglik=self.gibbs_data['XcondXDZ_loglik'][n_iter]+self.gibbs_data['Z_loglik'][n_iter]
            n_iter+=1
            t_init=time.process_time()
            
            # iterate over s,n
            for s in np.random.permutation(self.S):
                for n in np.random.permutation(self.N_s[s]):
                    if p_computation == 'general':
                        # General approach
                        # "Unsee" z_{sn}
                        self.Z[s,n]=np.NaN
                        # Recompute posterior
                        self.__compute_posterior()
                        # Sample z_{s,n} from conditional
                        x_lik=self.__compute_xlikelihood_per_mixture(self.X[s,n], int(self.XD[s,n]))
                        p_znew=self.m_posterior['alpha'][s]*x_lik
                        # Probability normalization by avoiding numerical errors
                        p_znew=(p_znew/p_znew.max())/((p_znew/p_znew.max()).sum())
                        # New mixture assignment
                        self.Z[s,n]=np.where(stats.multinomial.rvs(1,p_znew))[0][0]
                        # Recompute posterior
                        self.__compute_posterior()
                        
                    elif p_computation == 'online':           
                        # Update posterior: delete z_{s,n}, x_{s,n}
                        self.__update_posterior('del', s, int(self.Z[s,n]), self.X[s,n],int(self.XD[s,n]))
                        # Sample z_{s,n} from conditional
                        x_lik=self.__compute_xlikelihood_per_mixture(self.X[s,n],int(self.XD[s,n]))
                        p_znew=self.m_posterior['alpha'][s]*x_lik
                        # Probability normalization by avoiding numerical errors
                        p_znew=(p_znew/p_znew.max())/((p_znew/p_znew.max()).sum())
                        # New mixture assignment
                        self.Z[s,n]=np.where(stats.multinomial.rvs(1,p_znew))[0][0]
                        # Update posterior: add z_{s,n}, x_{s,n}
                        self.__update_posterior('add', s, int(self.Z[s,n]), self.X[s,n],int(self.XD[s,n]))
                    else:
                        raise ValueError('Unknown posterior computation type={}'.format(p_computation))
                
            # Save and print statistics
            # likelihood
            (self.gibbs_data['XcondXDZ_loglik'][n_iter], self.gibbs_data['Z_loglik'][n_iter])=self.compute_loglikelihood()
            # Execution time
            self.gibbs_data['delta_time'][n_iter]=time.process_time()-t_init
            print('n_iter={} in {} with log p(X,Z)={}, log p(X|XD,Z)={}, log p(Z)={}'.format(n_iter, self.gibbs_data['delta_time'][n_iter], self.gibbs_data['XcondXDZ_loglik'][n_iter]+self.gibbs_data['Z_loglik'][n_iter], self.gibbs_data['XcondXDZ_loglik'][n_iter], self.gibbs_data['Z_loglik'][n_iter]))

        # Save Gibbs data
        with open(save_dir+'/gibbs_data.pickle', 'wb') as f:
            pickle.dump(self.gibbs_data, f)

    def __compute_posterior(self):
        """ Compute the parameter posteriors based on atributes Z, X and XD
        
        Args:
            None
        """
        
        # Mixture posterior (for each set)
        if self.m_posterior == None:
            self.m_posterior={'dist':self.m_prior['dist'], 'alpha':self.m_prior['alpha']*np.ones((self.S,1))}
        
        # Data posterior and sufficient statistics for each data source
        if not hasattr(self, 'g_posterior'):
            # Allocate and initialize
            self.g_posterior=np.zeros(self.D, dtype='object')
            for d in np.arange(self.D):
                if 'multinomial' in self.f_emission[d]['dist'].__str__():
                    self.g_posterior[d]={'dist':self.g_prior[d]['dist'], 'beta':self.g_prior[d]['beta']*np.ones((self.K,1)), 'beta_sum':self.g_prior[d]['beta_sum']*np.ones((self.K,1))}
                else:
                    raise ValueError('f_emission distribution {} not implemented yet'.format(self.f_emission[d]['dist']))

        # For each mixture component k
        for k in np.arange(self.K):
            k_idx=(self.Z==k)
            # Update mixture posterior: prior + counts of observations with mixture k
            self.m_posterior['alpha'][:,k]=self.m_prior['alpha'][k]+(k_idx).sum(axis=1)
            
            # For each data source d
            for d in np.arange(self.D):
                # Data assigned to mixture k and data source d
                kd_component_data=self.X[(k_idx) & (self.XD==d)]
                # Data sufficient statistics
                if 'multinomial' in self.f_emission[d]['dist'].__str__():
                    # Identify observations vocabulary items
                    v_idx, v_count=np.unique(kd_component_data, return_counts=True)                       
                    # Data posterior: prior plus seen counts
                    self.g_posterior[d]['beta'][k,v_idx.astype(int)]=self.g_prior[d]['beta'][v_idx.astype(int)]+v_count
                    self.g_posterior[d]['beta_sum'][k]=self.g_posterior[d]['beta'][k].sum(keepdims=True)
                else:
                    raise ValueError('f_emission distribution {} not implemented yet'.format(self.f_emission[d]['dist']))

    def __update_posterior(self, how, s_update, k_update, x, d):
        """ Update the parameter posteriors based on provided info
        
        Args:
            how: either to add or delete provided info
            s_update: set of observations to update
            k_change: what mixture information to change
            x: observation to update 
            d: data source of observation to update 
        """
        
        if how == 'del':
            # Delete assignment mixture component k in set s
            self.m_posterior['alpha'][s_update,k_update]-=1
            # Data sufficient statistics
            if 'multinomial' in self.f_emission[d]['dist'].__str__():
                # Delete word v of data source d in mixture k
                self.g_posterior[d]['beta'][k_update, int(x)]-=1
                self.g_posterior[d]['beta_sum'][k_update]-=1
            else:
                raise ValueError('f_emission distribution {} not implemented yet'.format(self.f_emission[d]['dist']))  
        elif how == 'add':
            # Add assignment mixture component k in set s
            self.m_posterior['alpha'][s_update,k_update]+=1
            # Data sufficient statistics
            if 'multinomial' in self.f_emission[d]['dist'].__str__():
                # Add word v of data source d in mixture k
                self.g_posterior[d]['beta'][k_update, int(x)]+=1
                self.g_posterior[d]['beta_sum'][k_update]+=1
            else:
                raise ValueError('f_emission distribution {} not implemented yet'.format(self.f_emission[d]['dist']))
        else:
            raise ValueError('Unknown posterior update type={}'.format(how))

    def __compute_xlikelihood_per_mixture(self, x, d):
        """ Compute the likelihood of observation x of data source d per mixture k using latest posterior
        
        Args:
            x: observation
            d: observation data source
        """
        
        f_x_k=np.zeros(self.K)
        if 'multinomial' in self.f_emission[d]['dist'].__str__():
            f_x_k=self.g_posterior[d]['beta'][:,int(x)]/self.g_posterior[d]['beta_sum'][:,0]
        else:
            raise ValueError('f_emission distribution {} not implemented yet'.format(self.f_emission[d]['dist']))
        
        # Doublechecking
        # Zeros
        if np.any(f_x_k==0):
            f_x_k[f_x_k==0]=np.finfo(float).eps
        # Infs
        if np.any(np.isinf(f_x_k)):
            f_x_k[np.isinf(f_x_k)]=1.0
        # NaNs
        assert not np.any(np.isnan(f_x_k)), 'Nan in xlikelihood_per_mixture f_x_k={}'.format(f_x_k)
        
        # Return
        return f_x_k

    def compute_loglikelihood(self):
        """ Compute the log-likelihoods
            X given XD and Z
            Z
        
        Args:
        
        Returns:
            (XcondZ, Z) = Tuple with loglikelihood of X given XD and Z, and Z
        """
        
        # Compute logp(X|XD,Z)
        XcondXDZ_loglik=self.__compute_loglikelihood_XcondXDZ()
        
        # Compute logp(Z)
        Z_loglik=self.__compute_loglikelihood_Z()
        
        return XcondXDZ_loglik, Z_loglik
        
    def __compute_loglikelihood_XcondXDZ(self):
        """ Compute the log-likelihood of X, given data sources XD and assignments Z

        Args:
        """

        XcondXDZ_loglik_perK=np.zeros((self.K,1))
        
        # For each data source
        for d in np.arange(self.D):
            if 'multinomial' in self.f_emission[d]['dist'].__str__():
                XcondXDZ_loglik_perK+=special.gammaln(self.g_prior[d]['beta_sum'][None])-special.gammaln(self.g_posterior[d]['beta_sum'])+special.gammaln(self.g_posterior[d]['beta']).sum(axis=1,keepdims=True)-special.gammaln(self.g_prior[d]['beta']).sum(keepdims=True)
            else:
                raise ValueError('f_emission distribution {} not implemented yet'.format(self.f_emission[d]['dist']))

        return XcondXDZ_loglik_perK.sum()

    def __compute_loglikelihood_Z(self):
        """ Compute the log-likelihood of Z

        Args:
        """

        if 'dirichlet' in self.m_prior['dist'].__str__():
            Z_loglik_perS=special.gammaln(self.m_prior['alpha'].sum())-special.gammaln(self.m_posterior['alpha'].sum(axis=1))+special.gammaln(self.m_posterior['alpha']).sum(axis=1)-special.gammaln(self.m_prior['alpha']).sum()
            Z_loglik=Z_loglik_perS.sum(axis=0)
        else:
            raise ValueError('m_prior distribution {} not implemented yet'.format(self.m_prior['dist']))

        return Z_loglik

    def estimate_train_datalikelihood(self, estimation, save_dir):
        """ Compute likelihood of train set
                Chib style and left to right approaches are implemented
        Args:
            estimation: what type of estimation to compute and their parameters
                estimation['type']: 'chib' or 'lefttoright' style estimator                
                If estimation['type']='chib'
                    estimation['gibbs_iter']: number of initial Gibbs iterations to run
                    estimation['chib_M']: number of transition steps
                If estimation['type']='lefttoright'
                    estimation['samples_M']: number of samples to use
            save_dir: directory where to place temporary results
        """
                
        # data loglikelihood per set
        loglik_trainX=np.zeros(self.S)

        ############## CHIB-type estimator ################        
        if estimation['type']=='chib':
            # Train mixture z_star assignments for all sets
            # Use training mixture assignments and posteriors (i.e. latest Z)
            Z_star=np.copy(self.Z)
            m_posterior_alpha_Z_star=np.copy(self.m_posterior['alpha'])

            # Each set is processed independently
            # Per-topic emission distribution posteriors already learned in training
            for s in np.arange(self.S):
                print('Data for train set {}/{}'.format(s,self.S))               
                # Pick some observation order
                n_order=np.random.permutation(self.N_s[s])
                # Preallocate likelihood of observations (to avoid recomputation)
                x_lik=np.zeros((self.N_s[s],self.K))
                    
                # Max probability mixture assignment
                for n in n_order:
                    # Compute likelihood of observations once here
                    x_lik[n,:]=self.__compute_xlikelihood_per_mixture(self.X[s,n], int(self.XD[s,n]))
                    
                    # Update posterior: delete z_{s,n}
                    m_posterior_alpha_Z_star[s,int(Z_star[s,n])]-=1
                    p_znew=m_posterior_alpha_Z_star[s]*x_lik[n,:]
                    # New mixture assignment
                    Z_star[s,n]=p_znew.argmax()
                    # Update posterior: add z_{s,n}
                    m_posterior_alpha_Z_star[s,int(Z_star[s,n])]+=1

                # Variables for Chib-style estimator (per set)
                Z_samples=np.zeros((estimation['chib_M'],self.N_s[s]))
                m_posterior_alpha_Z_samples=np.zeros((estimation['chib_M'],self.K))
                # Initial z(m)
                m_init=np.random.randint(estimation['chib_M'])
                Z_samples[m_init]=Z_star[s,:self.N_s[s]]
                m_posterior_alpha_Z_samples[m_init]=m_posterior_alpha_Z_star[s]
                
                # Find intermediate probability state z(m_init), backwards
                for n in n_order[::-1]:
                    # Update posterior: delete z_{s,n}
                    m_posterior_alpha_Z_samples[m_init,int(Z_samples[m_init,n])]-=1
                    p_znew=m_posterior_alpha_Z_samples[m_init]*x_lik[n,:]
                    p_znew=(p_znew/p_znew.max())/((p_znew/p_znew.max()).sum())
                    # New mixture assignment
                    Z_samples[m_init,n]=np.where(stats.multinomial.rvs(1,p_znew))[0][0]
                    # Update posterior: add z_{s,n}
                    m_posterior_alpha_Z_samples[m_init,int(Z_samples[m_init,n])]+=1
                
                # Find probability states z(0:m_init-1) backwards
                for m in np.arange(m_init)[::-1]:
                    Z_samples[m]=Z_samples[m+1]
                    m_posterior_alpha_Z_samples[m]=m_posterior_alpha_Z_samples[m+1]
                    # Reverse order
                    for n in n_order[::-1]:
                        # Update posterior: delete z_{s,n}
                        m_posterior_alpha_Z_samples[m,int(Z_samples[m,n])]-=1
                        p_znew=m_posterior_alpha_Z_samples[m]*x_lik[n,:]
                        p_znew=(p_znew/p_znew.max())/((p_znew/p_znew.max()).sum())
                        # New mixture assignment
                        Z_samples[m,n]=np.where(stats.multinomial.rvs(1,p_znew))[0][0]
                        # Update posterior: add z_{s,n}
                        m_posterior_alpha_Z_samples[m,int(Z_samples[m,n])]+=1
                    
                # Find probability states z(m_init+1:M) forwards
                for m in np.arange(m_init+1, estimation['chib_M']):
                    Z_samples[m]=Z_samples[m-1]
                    m_posterior_alpha_Z_samples[m]=m_posterior_alpha_Z_samples[m-1]
                    for n in n_order:
                        # Update posterior: delete z_{s,n}
                        m_posterior_alpha_Z_samples[m,int(Z_samples[m,n])]-=1
                        p_znew=m_posterior_alpha_Z_samples[m]*x_lik[n,:]
                        p_znew=(p_znew/p_znew.max())/((p_znew/p_znew.max()).sum())
                        # New mixture assignment
                        Z_samples[m,n]=np.where(stats.multinomial.rvs(1,p_znew))[0][0]
                        # Update posterior: add z_{s,n}
                        m_posterior_alpha_Z_samples[m,int(Z_samples[m,n])]+=1

                # Compute forward probabilities T(z(m) to z*)
                T_logprob=np.zeros((estimation['chib_M'], self.N_s[s]))
                for n in n_order:
                    # Update posterior: delete z_{s,n}
                    m_posterior_alpha_Z_samples[np.arange(estimation['chib_M']),Z_samples[:,n].astype(int)]-=1
                    p_znew=m_posterior_alpha_Z_samples*x_lik[n,:][None,:]
                    # Probability normalization by avoiding numerical errors
                    p_znew=(p_znew/p_znew.max(axis=1, keepdims=True))/((p_znew/p_znew.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True))
                    # Probability of transitioning from z(m) to z*
                    T_logprob[:,n]=np.log(p_znew[:,int(Z_star[s,n])])
                    # Update posterior: add z_{s,n}
                    m_posterior_alpha_Z_samples[:,int(Z_star[s,n])]+=1
                
                # Compute log(XcondZ*)
                loglikXcondZ_star=np.log(x_lik[np.arange(self.N_s[s]),Z_star[s,:self.N_s[s]].astype(int)]).sum()
                # Compute logP(Z*)
                loglikZ_star=special.gammaln(self.m_prior['alpha'].sum())-special.gammaln(m_posterior_alpha_Z_star[s].sum())+special.gammaln(m_posterior_alpha_Z_star[s]).sum()-special.gammaln(self.m_prior['alpha']).sum()
                # Compute log(X) with Chib
                loglik_trainX[s]=loglikXcondZ_star+loglikZ_star+np.log(estimation['chib_M'])-special.logsumexp(T_logprob.sum(axis=1))
                
        ############## Left-to-right estimator ################
        elif estimation['type']=='lefttoright':
            # Each set is processed independently
            # per-topic emission distribution posteriors already learned)
            for s in np.arange(self.S):
                print('Data for train set {}/{}'.format(s,self.S))
                # Pick some observation order
                n_order=np.random.permutation(self.N_s[s])
                # And preallocate likelihood of observations (to avoid recomputation)
                x_lik=np.zeros((self.N_s[s],self.K))
                # Preallocate mixture assignment samples and probabilities
                z_samples=np.zeros((estimation['samples_M'],self.N_s[s]))
                p_n=np.zeros((estimation['samples_M'],self.N_s[s]))
                # Initialize posterior alpha for each particle m
                m_posterior_alpha=self.m_prior['alpha']*np.ones((estimation['samples_M'],1))
                
                # Iterate for observations
                for (n_i, n) in enumerate(n_order):
                    # Precompute likelihood of this observation
                    x_lik[n,:]=self.__compute_xlikelihood_per_mixture(self.X[s,n], int(self.XD[s,n]))
                    # Up to n-1
                    for n_less in n_order[:n_i]:
                        # Forget about observation n_less
                        m_posterior_alpha[np.arange(estimation['samples_M']),z_samples[:,n_less].astype(int)]-=1
                        # Compute probabilities
                        p_znew=m_posterior_alpha*x_lik[n_less,:][None,:]
                        # Normalize for conditional
                        p_znew=(p_znew/p_znew.max(axis=1, keepdims=True))/((p_znew/p_znew.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True))
                        # New mixture assignment
                        z_samples[:,n_less]=(np.random.rand(estimation['samples_M'])[:,None]>p_znew.cumsum(axis=1)).sum(axis=1)
                        # Update posterior
                        m_posterior_alpha[np.arange(estimation['samples_M']),z_samples[:,n_less].astype(int)]+=1
                    # For n
                    p_znew=m_posterior_alpha*x_lik[n,:][None,:]
                    # Normalize for conditional
                    p_znew=(p_znew/p_znew.max(axis=1, keepdims=True))/((p_znew/p_znew.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True))
                    # New mixture assignment
                    z_samples[:,n]=(np.random.rand(estimation['samples_M'])[:,None]>p_znew.cumsum(axis=1)).sum(axis=1)
                    # Compute p_n from joint p(z=k)p(w|z=k) and sum over k
                    p_n[:,n]=(m_posterior_alpha/m_posterior_alpha.sum(axis=1,keepdims=True)*x_lik[n,:][None,:]).sum(axis=1)
                    # Update posterior
                    m_posterior_alpha[np.arange(estimation['samples_M']),z_samples[:,n].astype(int)]+=1
                    
                # Compute log(X)
                loglik_trainX[s]=np.log(p_n.mean(axis=0)).sum()

        else:
            raise ValueError('Data loglikelihood estimatino type {} not implemented yet'.format(estimation['type']))

        ############## SAVE AND RETURN ################
        with open(save_dir+'/loglik_trainX_{}.pickle'.format(estimation['type']), 'wb') as f:
            pickle.dump(loglik_trainX, f)

        return loglik_trainX

    def run_test_gibbs_inference(self, testX, testXD, gibbs, save_dir, p_computation='general'):
        """ Run a Gibbs sampler (with collapsed distributions) for inference of the assignments of a new test set testX and testXD, after learning from training set
                Data matrix and posteriors are increased to contain both train and test data
        Args:
            testX: testS by testN_s array of observations
            testXD: testS by testN_s array of observations data source
            gibbs: parameters for general assignment Gibbs sampler
                gibbs['init']: How to initialize the Gibbs assignment matrix for the test set
                gibbs['max_iter']: maximum number of Gibbs sampling iterations to run
                gibbs['loglik_eps']: minimum relative difference on loglikelihood between iterations
                gibbs['burn_in']: number of Gibbs samples to skip before estimation
                gibbs['lag']: number of Gibbs samples to skip for estimation
            save_dir: directory where to place temporary results
            p_computation: whether to compute posteriors with general form or as online computation
        """
        
        ############## INITIALIZATION ################
        t_init=time.process_time()
        # Data to work with with
        assert np.all(~np.isnan(testXD) == ~np.isnan(testX)), 'NaN discrepancy between testXD and testX'
        
        # Figure out dimensions
        trainS=self.S
        testS=testX.shape[0]
        self.S+=testS
        testN_s=(~np.isnan(testX)).sum(axis=1)
        self.N_s=np.concatenate((self.N_s, testN_s), axis=0)
        self.N_sd=np.concatenate((self.N_sd, np.zeros((testS, self.D))), axis=0)
        for d in np.arange(self.D):
            self.N_sd[trainS:,d]=(testXD==d).sum(axis=1)

        # Append data
        trainX=self.X
        trainXD=self.XD
        self.X=np.nan*np.ones((self.S, np.max(self.N_s)))
        self.XD=np.nan*np.ones((self.S, np.max(self.N_s)))
        self.X[:trainS,:trainX.shape[1]]=trainX
        self.XD[:trainS,:trainXD.shape[1]]=trainXD
        self.X[trainS:,:testX.shape[1]]=testX
        self.XD[trainS:,:testXD.shape[1]]=testXD
        del trainX, trainXD 
        
        # Allocate space: use NaNs to indicate missing/nonapplicable
        trainZ=self.Z
        self.Z=np.NaN*np.ones((self.S, np.max(self.N_s))) # Z[s,n]=k
        self.Z[:trainS,:trainZ.shape[1]]=trainZ
        del trainZ
        
        # Allocate posterior
        train_m_posterior=self.m_posterior
        self.m_posterior={'dist':self.m_prior['dist'], 'alpha':self.m_prior['alpha']*np.ones((self.S,1))}
        self.m_posterior['alpha'][:trainS]=train_m_posterior['alpha']
        del train_m_posterior
        
        # Initialize test set assignments, with per-topic emission distribution posteriors already learned in training
        for s in trainS+np.random.permutation(testS):
            print('Initialize data for test set {}/{}'.format(s-trainS,testS))
            if gibbs['init']=='prior':
                # Randomly sample mixture assignments from hyperparameters
                n_s,k_s=np.where(stats.multinomial.rvs(1,self.m_prior['alpha']/(self.m_prior['alpha'].sum()), size=testN_s[s]))
                self.Z[s,n_s]=k_s
            else:
                for n in np.arange(self.N_s[s]):
                    # Sample z_{s,n} from conditional
                    x_lik=self.__compute_xlikelihood_per_mixture(self.X[s,n], int(self.XD[s,n]))
                    p_znew=self.m_posterior['alpha'][s]*x_lik
                    # Probability normalization by avoiding numerical errors
                    p_znew=(p_znew/p_znew.max())/((p_znew/p_znew.max()).sum())
                    if gibbs['init']=='posterior_random':
                        # New mixture assignment: random
                        self.Z[s,n]=np.where(stats.multinomial.rvs(1,p_znew))[0][0]
                    elif gibbs['init']=='posterior_max':
                        # New mixture assignment: mle
                        self.Z[s,n]=p_znew.argmax()
            
            # Update test set's mixture posterior
            for k in np.arange(self.K):
                # Prior + counts of observations assigned to mixture k
                self.m_posterior['alpha'][s,k]=self.m_prior['alpha'][k]+(self.Z[s,:testN_s[s-trainS]]==k).sum()
        
        ##############  GIBBS SAMPLING ################
        t_init=time.process_time()
        # Parameters
        self.test_gibbs=gibbs
        # Summary statistics
        self.test_gibbs_data={'XcondXDZ_loglik':-np.inf*np.ones(self.test_gibbs['max_iter']+1), 'Z_loglik':-np.inf*np.ones(self.test_gibbs['max_iter']+1), 'delta_time':-np.inf*np.ones(self.test_gibbs['max_iter']+1)}
        XZ_loglik=-np.inf
        # Initial statistics
        n_iter=1        
        # Save and print statistics
        # likelihood
        (self.test_gibbs_data['XcondXDZ_loglik'][n_iter], self.test_gibbs_data['Z_loglik'][n_iter])=self.compute_loglikelihood()
        # Execution time
        self.test_gibbs_data['delta_time'][n_iter]=time.process_time()-t_init
        print('Test n_iter={} in {} with log p(X,Z)={}, log p(X|XD,Z)={}, log p(Z)={}'.format(n_iter, self.test_gibbs_data['delta_time'][n_iter], self.test_gibbs_data['XcondXDZ_loglik'][n_iter]+self.test_gibbs_data['Z_loglik'][n_iter], self.test_gibbs_data['XcondXDZ_loglik'][n_iter], self.test_gibbs_data['Z_loglik'][n_iter]))

        # iterations with max_iter hard limit or no likelihood improvement bigger than epsilon
        while (n_iter < self.test_gibbs['max_iter'] and abs(self.test_gibbs_data['XcondXDZ_loglik'][n_iter]+self.test_gibbs_data['Z_loglik'][n_iter] - XZ_loglik) >= (self.test_gibbs['loglik_eps']*abs(XZ_loglik))):
            # Update variables
            XZ_loglik=self.test_gibbs_data['XcondXDZ_loglik'][n_iter]+self.test_gibbs_data['Z_loglik'][n_iter]
            n_iter+=1
            t_init=time.process_time()
            
            # iterate over ONLY TEST s,n
            for s in trainS+np.random.permutation(testS):
                for n in np.random.permutation(self.N_s[s]):
                    # Since posteriors are fixed otherwise, it does not matter computation type, so simply
                    # Update posterior: delete z_{s,n}
                    self.m_posterior['alpha'][s,int(self.Z[s,n])]-=1
                    # Sample z_{s,n} from conditional
                    x_lik=self.__compute_xlikelihood_per_mixture(self.X[s,n], int(self.XD[s,n]))
                    p_znew=self.m_posterior['alpha'][s]*x_lik
                    # Probability normalization by avoiding numerical errors
                    p_znew=(p_znew/p_znew.max())/((p_znew/p_znew.max()).sum())
                    # New mixture assignment
                    self.Z[s,n]=np.where(stats.multinomial.rvs(1,p_znew))[0][0]
                    # Update posterior: add z_{s,n}
                    self.m_posterior['alpha'][s,int(self.Z[s,n])]+=1
                
            # Save and print statistics
            # likelihood
            (self.test_gibbs_data['XcondXDZ_loglik'][n_iter], self.test_gibbs_data['Z_loglik'][n_iter])=self.compute_loglikelihood()
            # Execution time
            self.test_gibbs_data['delta_time'][n_iter]=time.process_time()-t_init
            print('Test n_iter={} in {} with log p(X,Z)={}, log p(X|XD,Z)={}, log p(Z)={}'.format(n_iter, self.test_gibbs_data['delta_time'][n_iter], self.test_gibbs_data['XcondXDZ_loglik'][n_iter]+self.test_gibbs_data['Z_loglik'][n_iter], self.test_gibbs_data['XcondXDZ_loglik'][n_iter], self.test_gibbs_data['Z_loglik'][n_iter]))

        # Save Gibbs data
        with open(save_dir+'/gibbs_data.pickle', 'wb') as f:
            pickle.dump(self.test_gibbs_data, f)

            
    def estimate_test_datalikelihood(self, testX, testXD, estimation, save_dir):
        """ Estimate data likelihood of test set
                Chib style and left to right approaches are implemented        
        Args:
            testX: S by N_s array of test set observations
            testXD: S by N_s array of observations data source
            estimation: what type of estimation to compute and their parameters
                estimation['type']: 'chib' or 'lefttoright' style estimator                
                If estimation['type']='chib'
                    estimation['gibbs_iter']: number of initial Gibbs iterations to run
                    estimation['chib_M']: number of transition steps
                If estimation['type']='lefttoright'
                    estimation['samples_M']: number of samples to use
            save_dir: directory where to place temporary results
        """
                
        # Figure out dimensions
        testS=testX.shape[0]
        testN_s=(~np.isnan(testX)).sum(axis=1)
        # data loglikelihood per set
        loglik_testX=np.zeros(testS)

        ############## CHIB-type estimator ################        
        if estimation['type']=='chib':
            # Test mixture z_star assignments for all sets
            Z_star=np.NaN*np.ones((testS, np.max(testN_s)))
            m_posterior_alpha_Z_star=self.m_prior['alpha']*np.ones((testS,1))

            # Each set is processed independently
            # Per-topic emission distribution posteriors already learned in training
            for s in np.arange(testS):
                print('Data for test set {}/{}'.format(s,testS))
                # Randomly sample mixture assignments from hyperparameters
                n_s,k_s=np.where(stats.multinomial.rvs(1,self.m_prior['alpha']/(self.m_prior['alpha'].sum()), size=testN_s[s]))
                Z_star[s,n_s]=k_s

                # Update test set's mixture posterior
                for k in np.arange(self.K):
                    # Prior + counts of observations assigned to mixture k
                    m_posterior_alpha_Z_star[s,k]=self.m_prior['alpha'][k]+(Z_star[s,:testN_s[s]]==k).sum()
               
                # Pick some observation order
                n_order=np.random.permutation(testN_s[s])
                # And preallocate likelihood of observations (to avoid recomputation)
                x_lik=np.zeros((testN_s[s],self.K))
                for n in n_order:
                    x_lik[n,:]=self.__compute_xlikelihood_per_mixture(testX[s,n],int(testXD[s,n]))
                                    
                # Some initial Gibbs iterations
                n_iter=0
                while n_iter < estimation['gibbs_iter']:
                    # iterate over observations
                    for n in n_order:
                        # Update test posterior: delete z_{s,n}
                        m_posterior_alpha_Z_star[s,int(Z_star[s,n])]-=1
                        p_znew=m_posterior_alpha_Z_star[s]*x_lik[n,:]
                        p_znew=(p_znew/p_znew.max())/((p_znew/p_znew.max()).sum())
                        # New mixture assignment
                        Z_star[s,n]=np.where(stats.multinomial.rvs(1,p_znew))[0][0]
                        # Update posterior: add z_{s,n}
                        m_posterior_alpha_Z_star[s,int(Z_star[s,n])]+=1
                    # Update variables
                    n_iter+=1
                    
                # Max probability mixture assignment
                for n in n_order:
                    # Update test posterior: delete z_{s,n}
                    m_posterior_alpha_Z_star[s,int(Z_star[s,n])]-=1
                    p_znew=m_posterior_alpha_Z_star[s]*x_lik[n,:]
                    # New mixture assignment
                    Z_star[s,n]=p_znew.argmax()
                    # Update posterior: add z_{s,n}
                    m_posterior_alpha_Z_star[s,int(Z_star[s,n])]+=1

                # Variables for Chib-style estimator (per set)
                Z_samples=np.zeros((estimation['chib_M'],testN_s[s]))
                m_posterior_alpha_Z_samples=np.zeros((estimation['chib_M'],self.K))
                # Initial z(m)
                m_init=np.random.randint(estimation['chib_M'])
                Z_samples[m_init]=Z_star[s,:testN_s[s]]
                m_posterior_alpha_Z_samples[m_init]=m_posterior_alpha_Z_star[s]
                
                # Find intermediate probability state z(m_init), backwards
                for n in n_order[::-1]:
                    # Update test posterior: delete z_{s,n}
                    m_posterior_alpha_Z_samples[m_init,int(Z_samples[m_init,n])]-=1
                    p_znew=m_posterior_alpha_Z_samples[m_init]*x_lik[n,:]
                    p_znew=(p_znew/p_znew.max())/((p_znew/p_znew.max()).sum())
                    # New mixture assignment
                    Z_samples[m_init,n]=np.where(stats.multinomial.rvs(1,p_znew))[0][0]
                    # Update posterior: add z_{s,n}
                    m_posterior_alpha_Z_samples[m_init,int(Z_samples[m_init,n])]+=1
                
                # Find probability states z(0:m_init-1) backwards
                for m in np.arange(m_init)[::-1]:
                    Z_samples[m]=Z_samples[m+1]
                    m_posterior_alpha_Z_samples[m]=m_posterior_alpha_Z_samples[m+1]
                    # Reverse order
                    for n in n_order[::-1]:
                        # Update test posterior: delete z_{s,n}
                        m_posterior_alpha_Z_samples[m,int(Z_samples[m,n])]-=1
                        p_znew=m_posterior_alpha_Z_samples[m]*x_lik[n,:]
                        p_znew=(p_znew/p_znew.max())/((p_znew/p_znew.max()).sum())
                        # New mixture assignment
                        Z_samples[m,n]=np.where(stats.multinomial.rvs(1,p_znew))[0][0]
                        # Update posterior: add z_{s,n}
                        m_posterior_alpha_Z_samples[m,int(Z_samples[m,n])]+=1
                    
                # Find probability states z(m_init+1:M) forwards
                for m in np.arange(m_init+1, estimation['chib_M']):
                    Z_samples[m]=Z_samples[m-1]
                    m_posterior_alpha_Z_samples[m]=m_posterior_alpha_Z_samples[m-1]
                    for n in n_order:
                        # Update test posterior: delete z_{s,n}
                        m_posterior_alpha_Z_samples[m,int(Z_samples[m,n])]-=1
                        p_znew=m_posterior_alpha_Z_samples[m]*x_lik[n,:]
                        p_znew=(p_znew/p_znew.max())/((p_znew/p_znew.max()).sum())
                        # New mixture assignment
                        Z_samples[m,n]=np.where(stats.multinomial.rvs(1,p_znew))[0][0]
                        # Update posterior: add z_{s,n}
                        m_posterior_alpha_Z_samples[m,int(Z_samples[m,n])]+=1

                # Compute forward probabilities T(z(m) to z*)
                T_logprob=np.zeros((estimation['chib_M'], testN_s[s]))
                for n in n_order:
                    # Update test posterior: delete z_{s,n}
                    m_posterior_alpha_Z_samples[np.arange(estimation['chib_M']),Z_samples[:,n].astype(int)]-=1
                    p_znew=m_posterior_alpha_Z_samples*x_lik[n,:][None,:]
                    # Probability normalization by avoiding numerical errors
                    p_znew=(p_znew/p_znew.max(axis=1, keepdims=True))/((p_znew/p_znew.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True))
                    # Probability of transitioning from z(m) to z*
                    T_logprob[:,n]=np.log(p_znew[:,int(Z_star[s,n])])
                    # Update posterior: add z_{s,n}
                    m_posterior_alpha_Z_samples[:,int(Z_star[s,n])]+=1
                
                # Compute log(XcondZ*)
                loglikXcondZ_star=np.log(x_lik[np.arange(testN_s[s]),Z_star[s,:testN_s[s]].astype(int)]).sum()
                # Compute logP(Z*)
                loglikZ_star=special.gammaln(self.m_prior['alpha'].sum())-special.gammaln(m_posterior_alpha_Z_star[s].sum())+special.gammaln(m_posterior_alpha_Z_star[s]).sum()-special.gammaln(self.m_prior['alpha']).sum()
                # Compute log(X) with Chib
                loglik_testX[s]=loglikXcondZ_star+loglikZ_star+np.log(estimation['chib_M'])-special.logsumexp(T_logprob.sum(axis=1))
                
        ############## Left-to-right estimator ################
        elif estimation['type']=='lefttoright':
            # Each set is processed independently
            # per-topic emission distribution posteriors already learned)
            for s in np.arange(testS):
                print('Data for test set {}/{}'.format(s,testS))
                # Pick some observation order
                n_order=np.random.permutation(testN_s[s])
                # And preallocate likelihood of observations (to avoid recomputation)
                x_lik=np.zeros((testN_s[s],self.K))
                # Preallocate mixture assignment samples and probabilities
                z_samples=np.zeros((estimation['samples_M'],testN_s[s]))
                p_n=np.zeros((estimation['samples_M'],testN_s[s]))
                # Initialize posterior alpha for each particle m
                m_posterior_alpha=self.m_prior['alpha']*np.ones((estimation['samples_M'],1))
                
                # Iterate for observations
                for (n_i, n) in enumerate(n_order):
                    # Precompute likelihood of this observation
                    x_lik[n,:]=self.__compute_xlikelihood_per_mixture(testX[s,n],int(testXD[s,n]))
                    # Up to n-1
                    for n_less in n_order[:n_i]:
                        # Forget about observation n_less
                        m_posterior_alpha[np.arange(estimation['samples_M']),z_samples[:,n_less].astype(int)]-=1
                        # Compute probabilities
                        p_znew=m_posterior_alpha*x_lik[n_less,:][None,:]
                        # Normalize for conditional
                        p_znew=(p_znew/p_znew.max(axis=1, keepdims=True))/((p_znew/p_znew.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True))
                        # New mixture assignment
                        z_samples[:,n_less]=(np.random.rand(estimation['samples_M'])[:,None]>p_znew.cumsum(axis=1)).sum(axis=1)
                        # Update posterior
                        m_posterior_alpha[np.arange(estimation['samples_M']),z_samples[:,n_less].astype(int)]+=1
                    # For n
                    p_znew=m_posterior_alpha*x_lik[n,:][None,:]
                    # Normalize for conditional
                    p_znew=(p_znew/p_znew.max(axis=1, keepdims=True))/((p_znew/p_znew.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True))
                    # New mixture assignment
                    z_samples[:,n]=(np.random.rand(estimation['samples_M'])[:,None]>p_znew.cumsum(axis=1)).sum(axis=1)
                    # Compute p_n from joint p(z=k)p(w|z=k) and sum over k
                    p_n[:,n]=(m_posterior_alpha/m_posterior_alpha.sum(axis=1,keepdims=True)*x_lik[n,:][None,:]).sum(axis=1)
                    # Update posterior
                    m_posterior_alpha[np.arange(estimation['samples_M']),z_samples[:,n].astype(int)]+=1
                    
                # Compute log(X)
                loglik_testX[s]=np.log(p_n.mean(axis=0)).sum()

        else:
            raise ValueError('Data loglikelihood estimatino type {} not implemented yet'.format(estimation['type']))

        ############## SAVE AND RETURN ################
        with open(save_dir+'/loglik_testX_{}.pickle'.format(estimation['type']), 'wb') as f:
            pickle.dump(loglik_testX, f)

        return loglik_testX    
    
    #### PLOTTING ####
    def plot_loglikelihoods(self, load_dir, plot_save=None):
        """ Plot computed log-likelihood evolutions
        
        Args:
            load_dir: directory with likelihood information
            plot_save: whether to show (default) or save (in provided directory) the plot
        """

        # Load
        with open(load_dir+'/gibbs_data.pickle', 'rb') as f:
            gibbs_data = pickle.load(f)
        
        # Figure 
        plt.figure()
        plt.plot(np.arange(len(gibbs_data['Z_loglik'])),gibbs_data['Z_loglik'], 'b', label='log p(Z)')
        plt.plot(np.arange(len(gibbs_data['XcondXDZ_loglik'])),gibbs_data['XcondXDZ_loglik'], 'g', label='log p(X|XD,Z)')
        plt.plot(np.arange(len(gibbs_data['XcondXDZ_loglik'])),gibbs_data['XcondXDZ_loglik']+gibbs_data['Z_loglik'], 'r', label='log p(X,Z|XD)')
        plt.xlim([1,1+(gibbs_data['XcondXDZ_loglik']!=-np.inf).sum()])
        plt.xlabel('n_iter')
        plt.ylabel(r'$log p( )$')
        plt.title('Data log-likelihoods')
        legend = plt.legend(loc='upper left', ncol=1, shadow=True)
        if plot_save is None: 
            plt.show()
        else:
            plt.savefig(plot_save+'/loglikelihood.pdf', format='pdf', bbox_inches='tight')
            plt.close()

    def plot_posterior_mixture_density(self, plot_save=None):
        """ Plot posterior mixture density
                From analytical posterior
        Args:
            plot_save: whether to show (default) or save (in provided directory) the plot
        """
        
        # Iterate over sets for plotting
        for s in np.arange(self.S):
            # Figure 
            plt.figure()
            plt.stem(np.arange(self.K), self.m_posterior['alpha'][s,:]/self.m_posterior['alpha'][s,:].sum(), 'r')
            plt.xlabel('k')
            plt.ylabel(r'$\phi_{sk}$')
            plt.title('Mixture posterior for set {}'.format(s))
            plt.axis([0-0.1, self.K-0.9, 0, 1])
            if plot_save is None: 
                plt.show()
            else:
                plt.savefig(plot_save+'/mixturePosterior_s{}.pdf'.format(s), format='pdf', bbox_inches='tight')
                plt.close()
            
        # For all, plot heatmap
        plt.figure()
        plt.pcolor(self.m_posterior['alpha']/self.m_posterior['alpha'].sum(axis=1, keepdims=True), cmap='inferno')
        plt.colorbar()
        plt.xlabel('k')
        plt.ylabel('s')
        plt.title('Mixture posterior heatmap')
        if plot_save is None: 
            plt.show()
        else:
            plt.savefig(plot_save+'/mixturePosterior_all.pdf', format='pdf', bbox_inches='tight')
            plt.close()

    def plot_posterior_emission_params(self, plot_save=None):
        """ Plot posterior emission parameters
                From analytical posterior
        Args:
            plot_save: whether to show (default) or save (in provided directory) the plot
        """
             
        # Data sufficient statistics per mixture component k and data source d
        for k in np.arange(self.K):
            k_idx=(self.Z==k)
            for d in np.arange(self.D):
                # Data assigned to mixture k and data source d
                kd_component_data=self.X[(k_idx) & (self.XD==d)]
                if 'multinomial' in self.f_emission[d]['dist'].__str__():
                    # Posterior
                    plt.stem(np.arange(self.f_emission[d]['d_x']),self.g_posterior[d]['beta'][k]/self.g_posterior[d]['beta_sum'][k], 'r')
                    plt.xlabel('v')
                    plt.ylabel(r'$\theta_{k,d,v}$')
                    plt.title('Infered emission density params for mixture {} and data source {}'.format(k,d))
                    plt.axis([0-0.1, self.f_emission[d]['d_x']-0.9, 0, 1])
                    if plot_save is None: 
                        plt.show()
                    else:
                        plt.savefig(plot_save+'/theta_{}{}v.pdf'.format(k,d), format='pdf', bbox_inches='tight')
                        plt.close()
                else:
                    raise ValueError('f_emission distribution {} not implemented yet'.format(self.f_emission[d]['dist']))
  
    def plot_mixture_assignments(self, colors, plot_save=None):
        """ Plot provided data mixture assignments
        
        Args:
            colors: list of colors to use per mixture component
            plot_save: whether to show (default) or save (in provided directory) the plot
        """

        # Figure for Z
        plt.figure()        
        for k in np.arange(self.K):
            k_idx=(self.Z==k)
            plt.scatter(*np.where(k_idx), color=colors[k], label='k={} N_k={}'.format(k, (k_idx).sum()))
        plt.xlabel('s')
        plt.ylabel(r'$n$')
        plt.title('Z_{s,n} mixture assignments')
        plt.axis([-0.1, self.S-1+0.1, -0.1, np.max(self.N_s)-1+0.1])
        legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
        if plot_save is None: 
            plt.show()
        else:
            plt.savefig(plot_save+'/Z_sn.pdf', format='pdf', bbox_inches='tight')
            plt.close()
	
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
