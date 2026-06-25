""" Adjusted grid search function for stepmix 
[https://github.com/Labo-Lacourse/stepmix/blob/master/stepmix] 
Major improvement: display several metrics for consideration,
alongside the convergence status, with visual plots to enable 
better decision-making
Not yet exactly a "grid" search, as just one step condition will be explored

v7: addition of Average Posterior Class of Assignment (AvePP), Odds of latent class correct classification (OCC), minimal symptom frequency, etc., as complementary information
v6: minor adjustments, especially plots and storage of information
v5: conversion to class, to merge with exploreGridsearch
v4: add option to store plots
v3: add log-likelihood test for nested models
v2: store object, report convergence
v1: other metrics beyond log-likelihood is reported
for now, not yet implemented:
- for different nsteps (just the default n_steps=1)

"""
import os
import itertools
import pandas as pd
import warnings
import copy
from scipy.stats import norm
import matplotlib.pyplot as plt
from stepmix.stepmix import StepMix
import numpy as np
import tqdm

from sklearn.base import clone
from sklearn.utils.validation import check_random_state, check_is_fitted

from scipy.stats import chi2

class optimize_stepMix():
    def __init__(self, data,
                 predictors,outcome=None, random_state=42):
        X=data.loc[:,predictors]
        
        self.predictors=predictors
        #self.outcome=outcome
        self.data0=data
        if outcome ==None:
            Y=None
        else:
            Y=data.loc[:,outcome]
        self.Y=Y
        self.X=X
        self.random_state=random_state
    def gridSearch(self, low, high,  max_iter=2000,verbose=True, sample_weight=None
                   ):        
        self.high=high
        self.low=low        
        X=self.X        
        Y=self.Y
        random_state=self.random_state
        model = StepMix(n_components=3, n_steps=1, measurement='bernoulli',max_iter=max_iter,
                structural='gaussian_unit', random_state=30, verbose=0, progress_bar=0)
        self.model=model
        """GridSearch for LCA optimization
    
        Get fit measures for a range of number of classes. For example, if you set low=1 and high=4, the function
        will return the metrics log-likelihood, AIC, BIC for each of the number of classes from 1 to 4. Note those are point-estimates, and no bootstrapping is applied.
    
        Parameters
        ----------
        model : StepMix instance
            A StepMix model.
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples, n_features_structural), default=None
        low: int, default=1
            Minimum number of classes to test.
        high: int, default=5
            Maximum number of classes to test.    
        random_state : int, default=None
        verbose : bool, default=True
    
        Returns
        ----------
        p-values: DataFrame
            p-values of the BLRT test for each comparison.
        """
        test_string = list()
        p_values = list()
        stats=list()
        obj_list=list()
        n_samples = X.shape[0]
        
        for k in range(low, high+1):
            print(f"Testing {k} classes...")
            estimator_rep = clone(model)
            estimator_rep.set_params(n_components=k)
            estimator_rep.set_params(verbose=0) # added to check if it stops printing report
            estimator_rep.fit(X, Y, sample_weight=sample_weight)
            avg_ll = estimator_rep.score(X, Y, sample_weight=sample_weight)
            
            ll = (
                avg_ll * np.sum(sample_weight)
                if sample_weight is not None
                else avg_ll * n_samples
            )
            npar=estimator_rep.n_parameters
            n = X.shape[0]
            ncomp=estimator_rep.n_components
            aic=( -2 * avg_ll * n + 2 * npar)
            bic=(-2 * avg_ll * n + npar * np.log( n ))        
            caic=( -2 * avg_ll * n + npar * (np.log(n) + 1))
            sabic=(-2 *avg_ll * n + npar * np.log(
                n * ((n + 2) / 24)
            ))
            entropy=estimator_rep.entropy(X)
            relentropy= (
                1 - entropy / (n * np.log(ncomp))
                if ncomp > 1
                else np.nan
            )
            npar=estimator_rep.n_parameters
            n = X.shape[0]
            ncol = X.shape[1]
            ncomp=estimator_rep.n_components
            #degrees of freedom - binary variables        
            dof=(2**ncol)-((ncomp-1)+ncol*(2))
           
            stats_k = {'k':k,"LL": np.array(ll), "score": np.array(avg_ll),
                'aic':np.array(aic),'bic':np.array(bic),
                'caic':np.array(caic), 'sabic': np.array(sabic),
                 'entropy': np.array(entropy),
                 'relative_entropy': np.array(relentropy),
                'convergence':estimator_rep.converged_,
            'npar':npar,'n':n,'ncomp':ncomp, 'dof':dof
                }       
            stats.append(pd.DataFrame([stats_k]))
            obj_list.append(estimator_rep)
        
        stats=pd.concat(stats)
       # if verbose:
            #print("\nResults")
           # print(stats_k)
        
        # perform LRT ================
        nestedModelsLRT=list()
        for k in range(low+1,high+1):
            k_p=k-1
            ll_k=stats.loc[stats.ncomp==k,'LL'].values[0]
            ll_kp=stats.loc[stats.ncomp==k_p,'LL'].values[0]
            dof_k=stats.loc[stats.ncomp==k,'npar'][0] # should be npar
            dof_kp=stats.loc[stats.ncomp==k_p,'npar'][0]
            a=_lrt(ll_kp, ll_k, dof_kp,dof_k) # should pass npar
            a['k']=k
            nestedModelsLRT.append(a)

            
        nestedModelsLRT=pd.DataFrame(nestedModelsLRT)
        
        nestedModelsLRT=pd.merge(stats,nestedModelsLRT)
        print(nestedModelsLRT[['ncomp','npar','n','convergence','LL','dof','log-likelihood diff','dof diff', 'p value','aic','bic','entropy','relative_entropy']])
        
        # generate plots =================
        adicText=' (max_iterations ='+str(estimator_rep.max_iter) +')'
        _plot_stats(stats, adicText)
        self.stats=stats        
        self.nestedModelsLRT=nestedModelsLRT
        self.obj_list=obj_list
        #TODO list
        k_list=[obj.n_components for obj in obj_list]
        stepMixObjs=dict(zip(k_list,obj_list))
        self.modelObjs=stepMixObjs
        self.k_list=k_list        

        # compute diagnostic classification metrics =========
        post_prob_dict=dict()
        diagClassMetric=[]
        ave_pp_dict=dict()
        occ_ratios_dict=dict()
        var_relevance=dict()
        for k, model in zip(k_list, obj_list):
            post_probs = model.predict_proba(X)
            post_probs_df = pd.DataFrame(post_probs, columns=[f'Class_{i}' for i in range(model.n_components)])
            post_prob_dict[k]=post_probs_df
        
            # Get modal class assignments
            modal_classes = model.predict(X)
            
            # Compute AvePP for each class
            #good model should have average posterior probabilities \(>0.70\) or \(>0.80\) for all classes.
            ave_pp = []
            for i in range(model.n_components):
                # Mean of posterior probabilities for class k, restricted to those assigned to class k
                pk_avg = post_probs[modal_classes == i, i].mean()
                ave_pp.append(pk_avg)
        
            #store min ave_pp
            ave_pp_dict[k]=ave_pp
                # Model-estimated class proportions (prior probabilities)
            pi = model.weights_
            
            occ_ratios = []
            for i in range(model.n_components):
                # Odds of being in class k given classification
                odds_correct = ave_pp[i] / (1 - ave_pp[i])
                # Odds of being in class k based on model priors
                odds_prior = pi[i] / (1 - pi[i])
                
                occ_k = odds_correct / odds_prior
                occ_ratios.append(occ_k)
            occ_ratios_dict[k]=occ_ratios
            # variable relevance 
            var_relevance[k]=model.get_mm_df().max(axis=1).droplevel([0,1])
            diagClassMetric.append(pd.DataFrame({'k':[k], 'ave_pp_min': min(ave_pp), 'occ_k_min':(np.nan if occ_k==np.inf  else min(occ_ratios)),
                                                'min_mm_f':model.get_mm_df().max(axis=1).min(),
                                                'min_class_perc':model.weights_.min()*100}))
            
        diagClassMetric=pd.concat(diagClassMetric)
        nestedModelsLRT=pd.concat([ nestedModelsLRT.set_index('k'), diagClassMetric.set_index('k')], join='outer',ignore_index=False, axis=1)

        self.nestedModelsLRT=nestedModelsLRT
        var_relevance=pd.concat(var_relevance,axis=1).T
        var_relevance['k']=k_list
        var_relevance.columns.name=None
        
        
        post_prob_df=pd.concat(post_prob_dict,axis=1)
        
        
        ave_pp_df=[pd.DataFrame(x) for x in ave_pp_dict.values()]
        ave_pp_df=pd.concat(ave_pp_df,axis=1)
        ave_pp_df.columns=k_list
        
        
        occ_ratios_df=[pd.DataFrame(x) for x in occ_ratios_dict.values()]
        occ_ratios_df=pd.concat(occ_ratios_df,axis=1)
        occ_ratios_df.columns=k_list
        
        
        DiagClass={'occ_ratios':occ_ratios_df, 'AvePP': ave_pp_df, 'var_relevance': var_relevance, 'posterior_prob': post_prob_df}
        self.DiagClass=DiagClass
        
    def summary(self,adicText='',path=None):
        _plot_stats(self.stats,adicText=adicText,resultsDir=path)
        if path is not None:
            self.nestedModelsLRT.to_csv(f'{path}/gridSearch/gridSearch_metrics.csv')
        
        pass
        
    def describe_k(self,k,covariates=None,resultsDir=None):        
        X=self.X
        Y=self.Y
        self.Y=Y
        if resultsDir is not None:
            if not os.path.exists(resultsDir):
                os.makedirs(resultsDir)
            if not os.path.exists(f'{resultsDir}/ncomp_{k}'):
                os.makedirs(f'{resultsDir}/ncomp_{k}')
        # print statistics
        print(f'======================= LCA with k = {k}=========================')
        model_k=self.modelObjs[k]
        model_k.report(X)
        #stats=self.stats
        #print(stats[stats.k==k].T)
        # Conditional probabilities
        #print('------------------------ Conditional probabillities ------------------------------')
        mmdf=model_k.get_mm_df()
        mmdf2=mmdf.reset_index(level=['model_name','param'],drop=True)
        mmdf2.reset_index(drop=False,inplace=True)
        mmdf2=mmdf2.set_index('variable')
        #print(mmdf2)
        mmdf=mmdf.reset_index()

        # line plot ======================
        fig = plt.figure(figsize=(20,8))
        ax=mmdf.drop(['model_name','param','variable'],axis=1).plot(kind='line',figsize=(15,5))
        #ax.set_xticklabels(varList)
        plt.xticks(ticks=range(len(self.predictors)), labels=mmdf.variable, rotation=45)
        if resultsDir is not None:
            #plt.savefig(f'{resultsDir}/ncomp_{k}/allvar_line.png', bbox_inches='tight')
            mmdf.to_csv(f'{resultsDir}/ncomp_{k}/conditionalProb.csv')
        plt.show()
        # radar plot =========
        mmdf=mmdf.reset_index(drop=False)
        ticks=[x for x in mmdf.columns[1:].values]
        categories=mmdf.variable
        N=len(categories)
        from math import pi
        
        for i, v in enumerate(range(k)):
            # But we need to repeat the first value to close the circular graph:
            values=mmdf[v].tolist()
            values += values[:1]
            values
        
            # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]
            
            fig, ax = plt.subplots(1,figsize=(5,5), subplot_kw={'projection': 'polar'})
            
            # Draw one axe per variable + add labels
            plt.xticks(angles[:-1], categories, color='blue', size=12)
        
            # Draw ylabels
            ax.set_rlabel_position(0)
            plt.yticks([0.1,0.5,1], ["0.1","0.5","1"], color="grey", size=7)
            plt.ylim(0,1)
        
            # Plot data
            ax.plot(angles, values, linewidth=1, linestyle='solid')
        
            # Fill area
            ax.fill(angles, values, 'darkblue', alpha=0.1)
            ax.set_title(f'class {v}')
        
            # Show the graph
            if resultsDir is not None:
                plt.savefig(f'{resultsDir}/ncomp_{k}/{v}_radar.png', bbox_inches='tight')
            #plt.show()

        # predictions
        print('------------------------ predictions ------------------------------')
        predicted=model_k.predict(X)

        ct=pd.Series(predicted).value_counts()
        count_df=pd.DataFrame({'cl': [f'cl_{x}' for x in ct.index], 'n':ct, 'perc':round(ct/sum(ct)*100,2)})
        lbl=[f'{p}%\n N={x}' for p,x in zip(count_df.perc,count_df.n)]
       
        fig, ax = plt.subplots(figsize=(8,5))
                    
        bars=ax.barh(count_df.cl, count_df.perc)
        #plt.bar_label(lbl, padding=5)
        ax.bar_label(bars, labels=lbl,
                     padding=1, color='b', fontsize=8)
        plt.ylabel('Classes')
        plt.xlabel('Percentages')
        plt.xlim(0,max(count_df.perc+10))
        plt.title(f'Frequencies')
        if resultsDir is not None:
            plt.savefig(f'{resultsDir}/ncomp_{k}/Frequecies.png', bbox_inches='tight')
        plt.show()
        
        print(pd.crosstab(Y, predicted,normalize='columns'))

    def predict_k(self,k=1):     
        X=self.X
        print(f'======================= LCA with k = {k}=========================')
        model_k=self.modelObjs[k]
        predicted=model_k.predict(X)
        return predicted
        
    def decide(self, k=None):
        if k is None:
            return 'k must be informed'
        else:
            #TOOD Issue alert if overwriting 
            self.k=k
            model_k=self.modelObjs[k]
            X=self.X
            self.otpimized_object=model_k
            data=self.data0
            #TODO - relevel one referential level as reference
            predicted=model_k.predict(X)
            data['predicted']=predicted
            self.predicted_Y=predicted
            data['predicted'] = data['predicted'].astype('category').cat.codes
            self.data=data
            print(f'decision on k={k} stored')




def _lrt(null_ll, alternative_ll, null_dof, alternative_dof):
    """Naive LRT Test    (not recommended, I included but not connected with others, as should not be recommended)

    Parameters
    ----------
    null_ll: log-likelihood with k classes.
    alternative_ll : log-likelihood with k + 1 classes.   

    Returns
    ----------
    p-value: float
        p-value of the LRT test. A significant test indicates the alternative k + 1 model provides a
        significantly better fit of the data.
    """
    
    LR = -2 * (null_ll-alternative_ll)     
    dof_diff=alternative_dof-null_dof
    
    p_value = chi2.sf(LR, dof_diff)
    return dict({'log-likelihood diff': LR, 'dof diff': dof_diff, 'p value': p_value})



def _plot_stats(stats, adicText='',resultsDir=None):
    low=stats.ncomp.min()
    high=stats.ncomp.max()
    #TODO - add differential markers for converged x failed runs
    marker_mapping = {
        'True': 'o', # circle
        'False': 'x'  # square
    }
    if resultsDir is not None:
            if not os.path.exists(resultsDir):
                os.makedirs(resultsDir)
            if not os.path.exists(f'{resultsDir}/gridSearch'):
                os.makedirs(f'{resultsDir}/gridSearch')
    # plot for log-likelihood==============
    fig, ax = plt.subplots()
    df_m=stats[['ncomp','LL']]
    ax.plot(df_m.ncomp, df_m[['LL']],'-',label='_log-likelihood')# 4. Add labels, legend, and display the plot
    ax.set_xlabel("k")
    ax.set_ylabel('log-likelihood')
    ax.set_title('log-likelihood'+adicText)
    plt.xticks(range(low, high,1))
    for category, marker_style in marker_mapping.items():
        # Select data points for the current category
        # Convert lists to numpy arrays for easier filtering
        x_arr = np.array(stats[['ncomp']])
        y_arr = np.array(stats[['LL']])
        categories = np.array([str(x[0]) for x in stats[['convergence']].values])
        cat_arr = np.array(categories)
        # Filter data
        x_cat = x_arr[cat_arr == category]
        y_cat = y_arr[cat_arr == category]
        
        # Plot the subset of data with the specific marker and add a label for the legend
        
        plt.scatter(x_cat, y_cat, marker=marker_style, label=f'Convergence {category}')
    plt.legend()
    # Show the graph
    if resultsDir is not None:
        plt.savefig(f'{resultsDir}/gridSearch/gridSearch_LL.png', bbox_inches='tight')
    plt.show()
    
    #bic sapib etc plots ======
    fig, ax = plt.subplots()    
    for m in ['bic','aic','sabic','caic']:
        df_m=stats[['ncomp',m]]
        ax.plot(df_m.ncomp, df_m[[m]],'-', label=m,         
            markersize=2, )# 4. Add labels, legend, and display the plot
        ax.set_xlabel("k")
        ax.set_ylabel('metric')
        
        for category, marker_style in marker_mapping.items():
            # Select data points for the current category
            # Convert lists to numpy arrays for easier filtering
            x_arr = np.array(stats[['ncomp']])
            y_arr = np.array(stats[m])
            categories = np.array([str(x[0]) for x in stats[['convergence']].values])
            cat_arr = np.array(categories)
            # Filter data
            x_cat = x_arr[cat_arr == category]
            y_cat = y_arr[cat_arr == category]
            
            # Plot the subset of data with the specific marker and add a label for the legend
            
            plt.scatter(x_cat, y_cat, marker=marker_style, label=f'_Convergence {category}')
        ax.legend()
    plt.xticks(range(low, high,1))
    ax.set_title("Curve for obtained metrics"+adicText)
    if resultsDir is not None:
        plt.savefig(f'{resultsDir}/gridSearch/gridSearch_InfoC.png', bbox_inches='tight')
    plt.show()
    
    # plots for entropy ================
    fig, ax = plt.subplots()
    df_m=stats[['ncomp','entropy']]
    ax.plot(df_m.ncomp, df_m[['entropy']],'-',label='_entropy')# 4. Add labels, legend, and display the plot
    ax.set_xlabel("k")
    ax.set_ylabel('entropy')
    ax.set_title('entropy'+adicText)
    plt.xticks(range(low, high,1))
    for category, marker_style in marker_mapping.items():
        # Select data points for the current category
        # Convert lists to numpy arrays for easier filtering
        x_arr = np.array(stats[['ncomp']])
        y_arr = np.array(stats[['entropy']])
        categories = np.array([str(x[0]) for x in stats[['convergence']].values])
        cat_arr = np.array(categories)
        # Filter data
        x_cat = x_arr[cat_arr == category]
        y_cat = y_arr[cat_arr == category]
        
        # Plot the subset of data with the specific marker and add a label for the legend
        
        plt.scatter(x_cat, y_cat, marker=marker_style, label=f'Convergence {category}')
    plt.legend()
    if resultsDir is not None:
        plt.savefig(f'{resultsDir}/gridSearch/gridSearch_entropy.png', bbox_inches='tight')
    plt.show()
    
    #relative entropy =============
    fig, ax = plt.subplots()
    df_m=stats[['ncomp','relative_entropy']]
    ax.plot(df_m.ncomp, df_m[['relative_entropy']],'-',label='_relative entropy')# 4. Add labels, legend, and display the plot
    ax.set_xlabel("k")
    ax.set_ylabel('relative entropy')
    ax.set_title('Relative Entropy'+adicText)
    for category, marker_style in marker_mapping.items():
        x_arr = np.array(stats[['ncomp']])
        y_arr = np.array(stats[['relative_entropy']])
        categories = np.array([str(x[0]) for x in stats[['convergence']].values])
        cat_arr = np.array(categories)
        # Filter data
        x_cat = x_arr[cat_arr == category]
        y_cat = y_arr[cat_arr == category]
        
        # Plot the subset of data with the specific marker and add a label for the legend
        
        plt.scatter(x_cat, y_cat, marker=marker_style, label=f'Convergence {category}')
    plt.legend()    
    plt.xticks(range(low, high,1))
    if resultsDir is not None:
        plt.savefig(f'{resultsDir}/gridSearch/gridSearch_relEntropy.png', bbox_inches='tight')
    plt.show()