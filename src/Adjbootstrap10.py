""" Adjusted versions of utility functions available in stepmix 
[https://github.com/Labo-Lacourse/stepmix/blob/master/stepmix/bootstrap.py] 
for model bootstrapping and confidence intervals. minor correction bias corrected confidence interval computation

v10 naive temp storage to deal with energy (and other) related issues (not deleted for now, just in case), with recovery function
v9 bootstrapped selection rate added, the confusing n_rep is replaced with n_bootstrap to make it more clear
V8 store all in single object
v7 move plot to out of main function to enable later visualization function
v6 percentage convergence added in the plots
v5 optimize, sample just one time
v4 store convergence stat, 
v3 optimize metrics computation
v2 store other parameters beyond log-likelihood, compute non-parametric bootsrapped (and bias corrected) confidence intervals
v1 very draft, store boostrapped results for ll
"""

from stepmix.stepmix import StepMix
import itertools
import pandas as pd
import warnings
import copy
from scipy.stats import norm
import matplotlib.pyplot as plt
import pickle
# import shutil
#from pathlib import Path

import numpy as np
import tqdm
import tempfile
import os
import datetime
import inspect

from sklearn.base import clone
from sklearn.utils.validation import check_random_state, check_is_fitted

from stepmix.bootstrap import mse, find_best_permutation







class bootGridSearch():
    """Bootstrapped grid search
    
        Run grid search with bootstrapping for IC metrics, and BLRT test for a range of number of classes. For example, if you set low=1 and high=4, the function
        will return the result of 3 tests [1 vs 2, 2 vs 3, 3 vs 4]."""
    
    def __init__(self, data,varList, max_iter=1000,  Y=None, verbose=True
    ):
        """    
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
        n_bootstrap: int
            Number of repetitions to fit.
        random_state : int, default=None
        verbose : bool, default=True    
        
        """
        model = StepMix(n_components=3, n_steps=1, measurement='bernoulli',
                structural='gaussian_unit', max_iter=max_iter,verbose=0, progress_bar=0)
        self.model=model
        X=data.loc[:,varList]
        self.data0=data
        self.varList=varList
        self.X=X
        self.Y=Y
        
    def bootstrapGridSearch(self, low=1, high=5,n_bootstrap=30, random_state=42, verbose=False,path=os.getcwd()):

        model=self.model
        X=self.X
        Y=self.Y
        
        self.low=low
        self.high=high
        self.n_bootstrap=n_bootstrap
        self.random_state=random_state
        self.n_bootstrap=n_bootstrap
        test_string = list()
        p_values = list()
        boot_stats=list()
        boot_ci_stats=list()
        # run all bootstrappings without blrt
        full_models=list()
        ks=list()
        date_prefix = datetime.datetime.now().strftime("BootGridSearch_tempFile_%Y-%m-%d_")

        tmp_dir=tempfile.mkdtemp(dir=path, prefix=date_prefix) 
        # run blrt
        for k in range(low, high+1):
            print(f"Testing {k} classes...")
            ks.append(k)
            alternative_model = clone(model)
            alternative_model.set_params(n_components=k)
    
            alternative_model.fit(X, Y)
            full_models.append(alternative_model)
            # run bootstrapping
            return_df, alternative_model_stats = _bootstrap( # not used yet but return_df can be used for bootstrapped CI for models... 
                alternative_model,
                X,
                Y,
                n_bootstrap=n_bootstrap,
                identify_classes=False,
                sampler=alternative_model,
                random_state=random_state,
                parametric=True,
            )
            alternative_model_stats['k']=k
            boot_stats.append(alternative_model_stats)
    
            bic_ci=_ci_boot(alternative_model_stats,alternative_model,0.05,X) #stats_null,estimator,alpha
            #TODO append k
            bic_ci=pd.DataFrame(bic_ci).T
            bic_ci['k']=k
            perc_conv=sum(alternative_model_stats['convergency_buffer'])/len(alternative_model_stats['convergency_buffer'])
            bic_ci['converngent']=perc_conv        
            boot_ci_stats.append(bic_ci)
            # already computed stats
            #boot_stats.append(alternative_model_stats)     
            # confidence intervals
            bic_hat=alternative_model.bic(X)
    
            print('ncomp = '+str(k)+': '+str(perc_conv*100)+'% of '+str(n_bootstrap)+' repetitions converged')
    
            if k>low:
                orderprev=ks.index(k)
                # compute p blrt if k not low
                null_model=full_models[orderprev-1]
                null_model_stats=boot_stats[orderprev-1]
                p =_blrt(
                    null_model,
                    alternative_model,
                    null_model_stats, 
                    alternative_model_stats,
                    X,
                    Y=Y,
                    n_bootstrap=n_bootstrap,
                    random_state=random_state,
                )
                # pvalues
                p_values.append(p        
                )
                test_string.append(f"{k} vs. {k - 1} classes")     
            else:
                p_values.append(np.nan)
                test_string.append('-')     

            # store temp file boot_ci_stats boot_stats p_values test_string full_models ks self
            # theoreticaly does not make sense to keep all steps, as they are redundant... should keep just last k
            pickle_path = os.path.join(tmp_dir, f"temp_{k}.pkl")
            with open(pickle_path, 'wb') as f:
                pickle.dump(boot_ci_stats, f)
                pickle.dump(boot_stats, f)
                pickle.dump(p_values, f)
                pickle.dump(test_string, f)
                pickle.dump(full_models, f)
                pickle.dump(ks, f)
                pickle.dump(self.__dict__, f)

            # possibility of deleting pkl files of k-2 to save disk space...
                    
        boot_ci_stats=pd.concat(boot_ci_stats)
        stats_df2=pd.concat(boot_stats)
        boot_results=pd.concat(
                [pd.pivot_table(
                        stats_df2,
                        index='k',
                        values=["LL",'aic','bic','caic','sabic'],
                        aggfunc=np.mean,
                    ).set_axis(['avg_LL','avg_aic','avg_bic','avg_caic','avg_sabic'],axis=1),
                pd.pivot_table(
                        stats_df2,
                        index='k',
                        values=["LL",'aic','bic','caic','sabic'],
                        aggfunc=np.std,
                    ).set_axis(['std_LL','std_aic','std_bic','std_caic','std_sabic'],axis=1)],axis=1
            )
        boot_stats=[pd.DataFrame(x) for x in boot_stats]
        boot_stats=pd.concat(boot_stats)
        self._boot_stats=boot_stats
        self.boot_results=boot_results
        self.blrt_p=p_values
        self.boot_ci_stats=boot_ci_stats
        # percentile ci ============
        self.plot_percentile_ci(path=None)
            
        #TODO store also 
        df = pd.DataFrame({"Test": test_string, "p": p_values}).set_index("Test")
        if verbose:
            print("\nBLRT Sweep Results")
            print(df.round(4))
        # bias-corrected ci (computed inside ci_boot) =============
        self.plot_biasCorrected_ci(path=None)
    
        ci_df=boot_ci_stats.reset_index(names='metric')
        ci_df_wide=pd.pivot_table(
                       ci_df,
                        index=['k'],
            columns='metric',
                        values=['theta_hat','z0','ci_low','ci_high','converngent']).reset_index()
        ci_df_wide.columns = ['_'.join(col) for col in ci_df_wide.columns.values]        
        resDf=pd.concat([df.reset_index(),ci_df_wide.reset_index()],axis=1)
        print(df)
        self.pvalues=df
        self.fitMetrics=resDf


        self._boot_selectionRate()

        # Manual cleanup of temporary directory if the function completes successfully. For now, deactivated just in case, as with the path parameter, it would make sense to keep it.
        
        #shutil.rmtree(temp_dir)
        # if to keep the directory, all temporary files  except the last one should be kept
        #folder=Path(tmp_dir)
        #file_to_keep= os.path.join(tmp_dir, f"temp_{high}.pkl")
        #for file in folder.iterdir():
            # Only target files (skips directories) and exclude the target name
            #if file.is_file() and file.name != file_to_keep:
               #file.unlink()
        
        
    def _boot_selectionRate(self, conv_criteria=0.3):
        fitMetrics=self.fitMetrics
        self.conv_criteria=conv_criteria
        convs=[x[0] for x in fitMetrics.loc[fitMetrics.converngent_caic>conv_criteria,['k_']].values]
        metric='bic'
        
        
        boot_stats=self._boot_stats
        min_k=self.low
        max_k=self.high
        nboot=self.n_bootstrap
        df_bootstats_filtered=boot_stats.copy()
        df_bootstats_filtered=df_bootstats_filtered[df_bootstats_filtered['k'].isin(convs)]
        
        df_bootstats_filtered['temp_idx'] = df_bootstats_filtered.groupby('k').cumcount()
        metric='aic'
        bootSelRate=[]
        for metric in ['aic','bic','caic','sabic','relative_entropy']:
            df_aic=pd.pivot_table(
                                       df_bootstats_filtered[['temp_idx','k',metric]],
                index='temp_idx',
                        columns='k',
                                        values=[metric]).reset_index(drop=True)
            df_aic.columns = df_aic.columns.get_level_values(1)
            indices = np.argmin(np.array(df_aic), axis=1)
            freq_table = pd.crosstab(index=pd.Series(df_aic.columns[indices]), columns="count")
            freq_table[metric]=[x/nboot*100 for x in freq_table['count']]
            freq_table.columns = freq_table.columns.to_flat_index()
            freq_table = freq_table.rename_axis(None, axis=1)
            freq_table=freq_table.reset_index().rename(columns={'row_0':'k'}).reset_index(drop=True)   
            
            freq_table.columns = freq_table.columns.astype(str)
            freq_table['k'] = freq_table['k'].astype(str)
            a=freq_table[['k',metric]]
            bootSelRate.append(a)
        for metric in ['entropy']:
            df_aic=pd.pivot_table(
                                       df_bootstats_filtered[['temp_idx','k',metric]],
                index='temp_idx',
                        columns='k',
                                        values=[metric]).reset_index(drop=True)
            df_aic.columns = df_aic.columns.get_level_values(1)
            indices = np.argmax(np.array(df_aic), axis=1)
            freq_table = pd.crosstab(index=df_aic.columns[indices].values, columns="count")
            freq_table[metric]=[x/nboot*100 for x in freq_table['count']]
            freq_table.columns = freq_table.columns.to_flat_index()
            freq_table = freq_table.rename_axis(None, axis=1)
            freq_table=freq_table.reset_index().rename(columns={'row_0':'k'}).reset_index(drop=True)   
            
            freq_table.columns = freq_table.columns.astype(str)
            freq_table['k'] = freq_table['k'].astype(str)
            a=freq_table[['k',metric]]
            bootSelRate.append(a)
        df = pd.concat([d.set_index('k') for d in bootSelRate], axis=1, join='outer').reset_index()
        emptydf=pd.DataFrame({'k':range(min_k,max_k+1),'nclasses':range(min_k,max_k+1),
                              'convs':['o' if x >conv_criteria else 'x' for x in fitMetrics.converngent_caic]})
        emptydf['k'] = emptydf['k'].astype(str)
        df=pd.concat([emptydf.set_index('k'),df.set_index('k')],axis=1, join='outer')
        
        self.bootstrap_selectionRate=df
        
    def plot_selectionRate(self,path=None):
        boot_selRate_df=self.bootstrap_selectionRate
        min_k=self.low
        max_k=self.high
        long_df = pd.melt(boot_selRate_df.reset_index(), id_vars=['k','convs'], 
                  value_vars=['aic',	'bic',	'caic',	'sabic',	'entropy',	'relative_entropy'], 
                  var_name='metric', value_name='SelectionRate')
        fig, ax = plt.subplots()
        long_df['SelectionRate'] = long_df['SelectionRate'].fillna(0)
        long_df['k'] = long_df['k'].astype(int)
        # Loop through each group and plot
        for label, group_df in long_df.groupby('metric'):
            ax.plot(group_df['k'], group_df['SelectionRate'], label=label)
            for i in range(len(group_df['k'])):
                ax.plot(group_df.iloc[i]['k'], group_df.iloc[i]['SelectionRate'], marker=group_df.iloc[i]['convs'])
        ax.set_xlabel('k')
        ax.set_ylabel('Selection Rate (%)')
        ax.legend(title='metric')
        plt.xticks(range(min_k,max_k+1,1))
        plt.figtext(0.5, 0.01, f'***selection rate ignoring points with convergency rate below {self.conv_criteria*100}%', ha="center", fontsize=8)
        plt.subplots_adjust(bottom=0.2) 
        if path is not None:
            plt.savefig(path+'/bSelBias.png')
        plt.show()

    def plot_biasCorrected_ci(self,path=None):
        boot_ci_stats=self.boot_ci_stats
        for m in ['bic','aic','sabic','caic', "score",'entropy','relative_entropy']:
            fig, ax = plt.subplots()
            df_m=boot_ci_stats[boot_ci_stats.index==m]        
            
            low=df_m.k.min()
            high=df_m.k.max()
            ax.fill_between(df_m.k, df_m.ci_low, df_m.ci_high, color='blue', alpha=0.15, label=m)# 4. Add labels, legend, and display the plot
            conv=['C:'+str(round(x*100))+'%' for x in df_m.converngent]
            for xi, yi, text in zip(df_m.k,  df_m.ci_high, conv):
                ax.annotate(text,
                            xy=(xi, yi), xycoords='data',
                            xytext=(1.5, 1.5), textcoords='offset points',size=8,color='blue')
            ax.set_xlabel("k")
            ax.set_ylabel("metric")
            ax.set_title("Bias-corrected Confidence Interval")
            ax.legend()
            plt.xticks(range(low, high,1))
            if path is not None:
                plt.savefig(path+'/bBiasCorrCI_'+m+'.png')
            plt.show()
    
    def plot_percentile_ci(self,path=None):      
        boot_stats=self._boot_stats
        boot_ci_stats=self.boot_ci_stats
        for m in ['bic','aic','sabic','caic', "score",'entropy','relative_entropy']:
            fig, ax = plt.subplots()
            low=boot_stats.k.min()
            high=boot_stats.k.max()
            df_m=boot_stats.groupby('k')[m].describe().reset_index()
            ax.fill_between(df_m.k, df_m['25%'], df_m['75%'], color='blue', alpha=0.15, label=m)# 4. Add labels, legend, and display the plot
            ax.set_xlabel("k")
            ax.set_ylabel("metric")
            df_c=boot_ci_stats[boot_ci_stats.index==m]
            conv=['C:'+str(round(x*100))+'%' for x in df_c.converngent]
            for xi, yi, text in zip(df_c.k,  df_m['75%'], conv):
                ax.annotate(text,
                            xy=(xi, yi), xycoords='data',
                            xytext=(1.5, 1.5), textcoords='offset points',size=8,color='blue')
            ax.set_title("Bootstrapped Percentile Confidence Interval")
            ax.legend()        
            plt.xticks(range(low, high,1))
            if path is not None:
                plt.savefig(path+'/bPercCI_'+m+'.png')
            plt.show()
            

    def report(self, path=None):
        boot_stats=self._boot_stats
        boot_ci_stats=self.boot_ci_stats
        p_values=self.blrt_p
        
        self.plot_percentile_ci(path=path)
        self.plot_biasCorrected_ci(path=path)

        ci_df=boot_ci_stats.reset_index(names='metric')
        ci_df['CI']=[f'[{x:,.3f}:{y:,.3f}]' for x,y in zip(ci_df.ci_low,ci_df.ci_high)]
        ci_df_wide=pd.pivot_table(
                           ci_df[['k','metric','theta_hat','CI']],
                            index=['k'],
            columns='metric',
                            values=['theta_hat', 'CI'],
            aggfunc=lambda x: ', '.join(str(v) for v in x) ).reset_index()
        ci_df_wide.columns = ['_'.join(col) for col in ci_df_wide.columns.values]  
        conv_df=pd.pivot_table(
                   ci_df[ci_df.metric=='bic'],
                    index=['k'],
                    values=['converngent']).reset_index()
        ci_cf=pd.concat([pd.Series(p_values,name='blrt_p'),ci_df_wide.reset_index(drop=True)],axis=1)
        ci_cf=pd.concat([ci_cf.reset_index(drop=True),conv_df.reset_index(drop=True)],axis=1)
        if path is not None:
            ci_cf.to_csv(f'{path}/metrics_bootstrap.csv')
        print(ci_cf)
        self.plot_selectionRate(path=path)

        
    
def _blrt(null_model, alternative_model,stats_null, stats_alternative, X, Y=None, n_bootstrap=30, random_state=42):
    """BLRT Test

    References
    ----------
    Dziak, John J., Stephanie T. Lanza, and Xianming Tan. "Effect size, statistical power, and sample size requirements for the bootstrap likelihood ratio test in latent class analysis." Structural equation modeling: a multidisciplinary journal 21.4 (2014): 534-552.
    Nylund, Karen L., Tihomir Asparouhov, and Bengt O. Muthén. "Deciding on the number of classes in latent class analysis and growth mixture modeling: A Monte Carlo simulation study." Structural equation modeling: A multidisciplinary Journal 14.4 (2007): 535-569.

    Parameters
    ----------
    null_model : StepMix instance
        A StepMix model with k classes.
    alternative_model : StepMix instance
        A StepMix model with k + 1 classes.
    X : array-like of shape (n_samples, n_features)
    Y : array-like of shape (n_samples, n_features_structural), default=None
    n_bootstrap: int
        Number of repetitions to fit.
    random_state : int, default=None

    Returns
    ----------
    p-value: float
        Bootstrap p-value of the BLRT test. A significant test indicates the alternative k + 1 model provides a
        significantly better fit of the data.
    """
    n_samples = X.shape[0]

    # Fit both models on real data
    
    real_stat = 2 * (alternative_model.score(X, Y) - null_model.score(X, Y)) * n_samples
    gen_stats = 2 * (stats_alternative["LL"] - stats_null["LL"])
    b = np.sum(gen_stats > real_stat)
    

    
    return b / n_bootstrap
    
def _ci_boot(stats_null,estimator,alpha,X):
    """ generate confidence intervals from bootraped data"""
    stats=dict()
    # bias corrected boostrap CI need to  be doen with estimator
    for m in ['bic','aic','sabic','caic', "score",'entropy','relative_entropy']:#unable to change LL because the method is 'score'
        method_to_call = getattr(estimator, m, None)
        n_repetition=len(stats_null[m])
        theta_hat= result = method_to_call(X)  
        sorted_theta_star = np.sort(stats_null[m])
        num_bellow_thetaHat=np.sum(stats_null[m]<theta_hat)
        bias_Prop=num_bellow_thetaHat/n_repetition
        z0=norm.ppf(bias_Prop)
        if z0==-np.inf:
            z0=norm.ppf(1/n_repetition)
        if z0==np.inf:
            z0=norm.ppf(1-1/n_repetition)
        za_low=norm.ppf(alpha/2)
        za_high=norm.ppf(1-alpha/2)
        phi_low=norm.cdf(2*z0+za_low)
        phi_high=norm.cdf(2*z0+za_high)
        phi_low=min(n_repetition-1,int(np.floor(phi_low*n_repetition)))
        phi_high=min(n_repetition-1,int(np.ceil(phi_high*(n_repetition))))
        phi_low=max(0,phi_low)
        phi_high=max(0,phi_high)
        ci_low=sorted_theta_star[phi_low]
        ci_high=sorted_theta_star[phi_high]
        
        ic_stats = {'z0': z0,'theta_hat':theta_hat,
                    'ci_low':ci_low,'ci_high': ci_high
                }
        stats[m]=ic_stats
    
    # BCa not yet implemented as it would require jackknife, that would increase processing time
    return stats
def _bootstrap(
    estimator,
    X,
    Y=None,
    n_bootstrap=1000,
    sample_weight=None,
    parametric=False,
    sampler=None,
    identify_classes=True,
    progress_bar=True,
    random_state=None,
):
    """Parametric or Non-parametric bootstrap of a StepMix estimator.

    Fit n_bootstrap clones of the estimator on resampled datasets.

    If identify_classes=True, repeated parameter estimates are aligned with the class order of the main estimator using
    a permutation search.

    Parameters
    ----------
    estimator : StepMix instance
        A fitted StepMix estimator. Used as a template to clone bootstrap estimator.
    X : array-like of shape (n_samples, n_features)
    Y : array-like of shape (n_samples, n_features_structural), default=None
    n_bootstrap: int
        Number of repetitions to fit.
    sample_weight : array-like of shape(n_samples,), default=None
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight. Ignored if parametric=True.
    parametric: bool, default=False
        Use parametric bootstrap instead of non-parametric. Data will be generated by sampling the estimator.
    sampler: bool, default=None
        Another fitted estimator to use for sampling instead of the main estimator. Only used for parametric
        bootstrapping.
    identify_classes: bool, default=True
        Run a permutation test to align the classes of the repetitions to the classes of the main estimator. This is
        required if inference on the model parameters is needed, but can be turned off if only the likelihood needs
        to be bootstrapped to save computations.
    progress_bar : bool, default=True
        Display a tqdm progress bar for repetitions.
    random_state : int, default=None
    Returns
    ----------
    samples: DataFrame
        DataFrame of all repetitions. Follows the convention of StepMix.get_parameters_df() with an additional
        'rep' column.
    rep_stats: DataFrame
        Likelihood statistics of each repetition.
        'rep' column. None if identy_classes=False.
    stats: DataFrame
        Various statistics of bootstrapped estimators.
    """
    check_is_fitted(estimator)
    estimator = copy.deepcopy(estimator)
    estimator.set_params(random_state=random_state)

    if sampler is not None:
        check_is_fitted(sampler)
        sampler = copy.deepcopy(sampler)
        sampler.set_params(random_state=random_state)

    n_samples = X.shape[0]
    x_names = estimator.x_names_
    y_names = estimator.y_names_ if hasattr(estimator, "y_names_") else None

    # Use the estimator built-in method to check the input
    # This will ensure that X and Y are numpy arrays for the rest of the bootstrap procedure
    X, Y = estimator._check_x_y(X, Y, reset=False)

    # Get class probabilities of main estimator as reference
    ref_class_probabilities = estimator.predict_proba(X, Y)

    # Raise warning if trying to permute too many columns
    if identify_classes and estimator.n_components > 6:
        warnings.warn(
            "Bootstrapping with identfy_classes=True requires permuting latent classes. Permuting latent classes with n_components > 6 may be slow."
        )

    # Now fit n_bootstrap estimator with resampling and save parameters
    rng = check_random_state(estimator.random_state)
    parameters = list()
    avg_ll_buffer = list()
    ll_buffer = list()
    aic_buffer=list()
    bic_buffer=list()
    caic_buffer=list()
    sabic_buffer=list()
    entropy_buffer=list()
    relentropy_buffer=list()
    convergency_buffer=list()

    if progress_bar:
        print("\nBootstrapping estimator...")

    tqdm_rep = tqdm.trange(
        n_bootstrap, disable=not progress_bar, desc="Bootstrap Repetitions    "
    )
    
    for rep in tqdm_rep:
        # Resample data
        if parametric and sampler is not None:
            X_rep, Y_rep, _ = sampler.sample(n_samples)
            sample_weight_rep = None
        elif parametric:
            X_rep, Y_rep, _ = sampler.sample(n_samples)
            sample_weight_rep = None
        else:
            # Sample with replacement
            rep_samples = rng.choice(n_samples, size=(n_samples,), replace=True)
            X_rep = X[rep_samples]
            Y_rep = Y[rep_samples] if Y is not None else None
            sample_weight_rep = (
                sample_weight[rep_samples] if sample_weight is not None else None
            )

        # Fit estimator on resample data
        estimator_rep = clone(estimator)
        
        # Disable printing for repeated estimators and fit
        estimator_rep.set_params(verbose=0, progress_bar=0, random_state=random_state)
        estimator_rep.fit(X_rep, Y_rep, sample_weight=sample_weight_rep)

        # Class ordering may be different. Reorder based on the best permutation of class probabilities
        if identify_classes:
            rep_class_probabilities = estimator_rep.predict_proba(
                X, Y
            )  # Inference on original samples
            perm = find_best_permutation(
                ref_class_probabilities, rep_class_probabilities
            )
            estimator_rep.permute_classes(perm)

        # Save parameters (but not yet used, nor passed for next steps
        df_i = estimator_rep.get_parameters_df(x_names, y_names)
        df_i["rep"] = rep
        parameters.append(df_i)

        # Save likelihood
        avg_ll = estimator_rep.score(X_rep, Y_rep, sample_weight=sample_weight_rep)
        ll = (
            avg_ll * np.sum(sample_weight)
            if sample_weight is not None
            else avg_ll * n_samples
        )
        avg_ll_buffer.append(avg_ll)
        ll_buffer.append(ll)
        npar=estimator_rep.n_parameters
        n = X.shape[0]
        ncomp=estimator_rep.n_components
        aic_buffer.append( -2 * avg_ll * n + 2 * npar)
        bic_buffer.append(-2 * avg_ll * n + npar * np.log( n ))        
        caic_buffer.append( -2 * avg_ll * n + npar * (np.log(n) + 1))
        sabic_buffer.append(-2 *avg_ll * n + npar * np.log(
            n * ((n + 2) / 24)
        ))
        entropy=estimator_rep.entropy(X_rep)
        entropy_buffer.append(entropy)
        relentropy_buffer.append( 
            1 - entropy / (n * np.log(ncomp))
            if ncomp > 1
            else np.nan
        )
        convergency_buffer.append(estimator_rep.converged_)
        # Ask tqdm to display current max likelihood
        tqdm_rep.set_postfix(
            median_LL=np.median(ll_buffer),
            # min_avg_LL=np.min(avg_ll_buffer),
            # max_avg_LL=np.max(avg_ll_buffer),
            min_LL=np.min(ll_buffer),
            max_LL=np.max(ll_buffer),
        )

    if identify_classes:
        return_df = pd.concat(parameters)
        return_df.sort_index(inplace=True)
    else:
        # Do not return parameters if classes are not identified
        return_df = None
    
    ncomp=estimator_rep.n_components
    #todo degrees of freedom
    # Add likelihoods statistics
    stats = {"LL": np.array(ll_buffer), "score": np.array(avg_ll_buffer),
            'aic':np.array(aic_buffer),'bic':np.array(bic_buffer),
            'caic':np.array(caic_buffer), 'sabic': np.array(sabic_buffer),
             'entropy': np.array(entropy_buffer),
             'relative_entropy': np.array(relentropy_buffer),
             'convergency_buffer':convergency_buffer,
             'parameters':parameters,
             'npar':npar,'n':n,'ncomp':ncomp, #'dof':dof
            }
    #todo we need also z for each 

    return return_df, pd.DataFrame.from_dict(stats)
        
def rescue_bootGridSearch(path):
    with open(path, 'rb') as f:
            boot_ci_stats=pickle.load(f)
            boot_stats=pickle.load(f)
            p_values=pickle.load(f)
            test_string=pickle.load(f)
            full_models=pickle.load(f)
            ks=pickle.load(f)
            selfT=pickle.load(f)
        
    valid_args = inspect.signature(bootGridSearch.__init__).parameters
    filtered_data = {k: v for k, v in selfT.items() if k in valid_args}
    data=selfT['data0']
    boot_res = bootGridSearch(data=data,**filtered_data)
    boot_res.low=low=selfT['low']
    boot_res.high=high=selfT['high']
    boot_res.n_bootstrap=n_bootstrap=selfT['n_bootstrap']
    boot_res.random_state=random_state=selfT['random_state']
    boot_res.blrt_p=p_values
    last_k=ks[-1]
    if last_k<high:
        # run blrt
        pickle_path= os.path.dirname(path)
        model=boot_res.model
        X=boot_res.X
        Y=boot_res.Y
        for k in range(last_k+1, high+1):
            print(f"Testing {k} classes...")
            ks.append(k)
            alternative_model = clone(model)
            alternative_model.set_params(n_components=k)
    
            alternative_model.fit(X, Y)
            full_models.append(alternative_model)
            # run bootstrapping
            return_df, alternative_model_stats = _bootstrap( # not used yet but return_df can be used for bootstrapped CI for models... 
                alternative_model,
                X,
                Y,
                n_bootstrap=n_bootstrap,
                identify_classes=False,
                sampler=alternative_model,
                random_state=random_state,
                parametric=True,
            )
            alternative_model_stats['k']=k
            boot_stats.append(alternative_model_stats)
    
            bic_ci=_ci_boot(alternative_model_stats,alternative_model,0.05,X) #stats_null,estimator,alpha
            #TODO append k
            bic_ci=pd.DataFrame(bic_ci).T
            bic_ci['k']=k
            perc_conv=sum(alternative_model_stats['convergency_buffer'])/len(alternative_model_stats['convergency_buffer'])
            bic_ci['converngent']=perc_conv        
            boot_ci_stats.append(bic_ci)
            # already computed stats
            #boot_stats.append(alternative_model_stats)     
            # confidence intervals
            bic_hat=alternative_model.bic(X)
    
            print('ncomp = '+str(k)+': '+str(perc_conv*100)+'% of '+str(n_bootstrap)+' repetitions converged')
    
            if k>low:
                orderprev=ks.index(k)
                # compute p blrt if k not low
                null_model=full_models[orderprev-1]
                null_model_stats=boot_stats[orderprev-1]
                p =_blrt(
                    null_model,
                    alternative_model,
                    null_model_stats, 
                    alternative_model_stats,
                    X,
                    Y=Y,
                    n_bootstrap=n_bootstrap,
                    random_state=random_state,
                )
                # pvalues
                p_values.append(p        
                )
                test_string.append(f"{k} vs. {k - 1} classes")     
            else:
                p_values.append(np.nan)
                test_string.append('-')     
    
            # store temp file boot_ci_stats boot_stats p_values test_string full_models ks self
            # theoreticaly does not make sense to keep all steps, as they are redundant... should keep just last k
            pickle_path2 = os.path.join(pickle_path, f"temp_{k}.pkl")
            with open(pickle_path2, 'wb') as f:
                pickle.dump(boot_ci_stats, f)
                pickle.dump(boot_stats, f)
                pickle.dump(p_values, f)
                pickle.dump(test_string, f)
                pickle.dump(full_models, f)
                pickle.dump(ks, f)
                pickle.dump(boot_res.__dict__, f)
    
    boot_ci_stats=pd.concat(boot_ci_stats)
    stats_df2=pd.concat(boot_stats)
    boot_results=pd.concat(
            [pd.pivot_table(
                    stats_df2,
                    index='k',
                    values=["LL",'aic','bic','caic','sabic'],
                    aggfunc=np.mean,
                ).set_axis(['avg_LL','avg_aic','avg_bic','avg_caic','avg_sabic'],axis=1),
            pd.pivot_table(
                    stats_df2,
                    index='k',
                    values=["LL",'aic','bic','caic','sabic'],
                    aggfunc=np.std,
                ).set_axis(['std_LL','std_aic','std_bic','std_caic','std_sabic'],axis=1)],axis=1
        )
    boot_stats=[pd.DataFrame(x) for x in boot_stats]
    boot_stats=pd.concat(boot_stats)
    boot_res._boot_stats=boot_stats
    
    boot_res.boot_results=boot_results
    
    boot_res.boot_ci_stats=boot_ci_stats
    boot_res.plot_percentile_ci(path=None)
    verbose=False
                
    #TODO store also 
    df = pd.DataFrame({"Test": test_string, "p": p_values}).set_index("Test")
    if verbose:
        print("\nBLRT Sweep Results")
        print(df.round(4))
    # bias-corrected ci (computed inside ci_boot) =============
    boot_res.plot_biasCorrected_ci(path=None)
    
    ci_df=boot_ci_stats.reset_index(names='metric')
    ci_df_wide=pd.pivot_table(
                   ci_df,
                    index=['k'],
        columns='metric',
                    values=['theta_hat','z0','ci_low','ci_high','converngent']).reset_index()
    ci_df_wide.columns = ['_'.join(col) for col in ci_df_wide.columns.values]        
    resDf=pd.concat([df.reset_index(),ci_df_wide.reset_index()],axis=1)
    print(df)
    boot_res.pvalues=df
    boot_res.fitMetrics=resDf
    boot_res._boot_selectionRate()
    boot_res.plot_selectionRate()    
    

    return boot_res
        