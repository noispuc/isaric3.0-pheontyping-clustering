""" Functions to run and compare probit logistic regression or multinomial logistic regression models 

models_str must be a list with strings comprising the model formulas. Data must be clean and categorised.
"""

import itertools
import pandas as pd
import numpy as np
from forestplot import forestplot
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from statsmodels.discrete.discrete_model import MNLogit
import statsmodels.api as sm
from scipy.stats import chi2
import patsy
import os
from AdjGridSearch5 import optimize_stepMix

def compareModels_MNLogit(models_str,data, path=None,silent=True):
    fits=[]
    params=[]
    models_str0=models_str.copy()
    models_str2=[]
    modelObjs=[]
    depVars=[]
    for m in models_str0:
        y, X = patsy.dmatrices(m, data, return_type='dataframe')
        model_h0 = MNLogit(y,X)
        try:            
            modelObjs.append(model_h0)
            results = model_h0.fit()
            fits.append(results)
            models_str2.append(m)
            depVars.append([a for a in y.columns])
        except:
            print(f'the following model failed to fit and will be ignored:\n{m}\n')
            models_str0.remove(m)
        
    nmodel=len(fits)#models_str
    comps=[dict({'reduced model':'-','model':models_str2[0],
                 'Reduced Model Log-Likelihood':np.nan,
                'Model Log-Likelihood':fits[0].llf,
                 'Model df':fits[0].df_model,
                'LRT Statistic':np.nan,
                'df_diff':np.nan,'p-value':np.nan,
                'pseudoR-squared': round(fits[0].prsquared,3),
                          'bic':round(fits[0].bic,3),
                          'aic':round(fits[0].aic,3)})]
    results=fits[0]
    pval=pd.DataFrame(np.where(results.pvalues > 0.001, round(results.pvalues,3).astype(str), '<0.001'),index=results.params.index)
    param_df=pd.DataFrame(round(results.params,3).astype(str),index=pval.index)+' (p='+pval+')'
    params.append(param_df)
    for x in range(1,len(fits)):
        model_larger=fits[x]
        model_smaller=fits[x-1]
        mobj=modelObjs[x]
        depVar=depVars[x]
        f_larger=models_str0[x]
        f_smaller=models_str0[x-1]
        ll1 = model_larger.llf # Log-likelihood of Model 1
        df1 = model_larger.df_model # Degrees of freedom of Model 1
    
        ll2 = model_smaller.llf # Log-likelihood of Model 1
        df2 = model_smaller.df_model # Degrees of freedom of Model 1
    
        lsrt_stat=2*(ll1-ll2)
        df_diff=df1-df2
        p=chi2.sf(lsrt_stat,df_diff)
        comps.append(dict({'reduced model':f_smaller,
                           'model':f_larger,
                 'Reduced Model Log-Likelihood':round(ll2,3),
                'Model Log-Likelihood':round(ll1,3),
                'LRT Statistic':round(lsrt_stat,3),
                'df_diff':int(df_diff),'p-value':p,      
                           'Model df':int(df1),
                         'pseudoR-squared': round(model_larger.prsquared,3),
                          'bic':round(model_larger.bic,3),
                          'aic':round(model_larger.aic,3),
                         'endogenous':mobj.endog_names,
                           'exogenous':mobj.exog_names,
                           'depVars':depVar
                          },
                        
        ))
        pval=pd.DataFrame(np.where(model_larger.pvalues > 0.001, round(model_larger.pvalues,3).astype(str), '<0.001'),index=model_larger.params.index)
        param_df=pd.DataFrame(round(model_larger.params,3).astype(str),index=pval.index)+' (p='+pval+')'
        params.append(param_df)
    comps_df=pd.DataFrame(comps)
    # possibly return fits is also interesting
    clNames=[x for x in y.columns[1:y.shape[1]]]
    for x in range(len(params)):
        params[x].columns=clNames
        a=pd.concat(params, axis=1)
    pardict = {}
    for x in clNames:
        pardict[x]=a[x]
        #TODO implement save to file
    pardict2=[x.set_axis(models_str2, axis=1).set_axis(y+'_'+x.index.astype(str)) for x,y in zip(pardict.values(),pardict.keys())]
    pardict2=pd.concat(pardict2,axis=0)
    if path is not None:
        pardict2.to_csv(f'{path}/phenotype_MNLogitmodels_comp_pars.csv')
        comps_df.to_csv(f'{path}/phenotype_MNLogitmodels_comp_models.csv')
    
    return comps_df, pardict2, fits,modelObjs

def compareProbitModels(models_str,data,silent=True,path=None):
    fits=[]
    params=[]
    model_str0=models_str.copy()
    for m in model_str0:        
        try:
            y, X = patsy.dmatrices(m, data, return_type='dataframe')
            model_h0 = sm.Probit(y.iloc[:,1],X)
            results = model_h0.fit(maxiter=5000)
            fits.append(results)
            pval=pd.DataFrame(np.where(results.pvalues > 0.001, round(results.pvalues,3).astype(str), '<0.001'),index=results.params.index)
            param_df=pd.DataFrame(round(results.params,3).astype(str),index=pval.index)+' (p='+pval+')'
            params.append(param_df)
        except:
            print(f'the following model failed to fit and will be ignored:\n{m}\n')
            model_str0.remove(m)
            
        
    nmodel=len(fits)#models_str
    if nmodel>=2:
        comps=[dict({'reduced model':'-','model':model_str0[0],
                     'Reduced Model Log-Likelihood':np.nan,
                    'Model Log-Likelihood':fits[0].llf,
                     'Model df':fits[0].df_model,
                    'LRT Statistic':np.nan,
                    'df_diff':np.nan,'p-value':np.nan,
                    'pseudoR-squared': round(fits[0].prsquared,3),
                              'bic':round(fits[0].bic,3),
                              'aic':round(fits[0].aic,3)})]
        
        for x in range(1,len(fits)):
            model_larger=fits[x]
            model_smaller=fits[x-1]
            f_larger=model_str0[x]
            f_smaller=model_str0[x-1]
            ll1 = model_larger.llf # Log-likelihood of Model 1
            df1 = model_larger.df_model # Degrees of freedom of Model 1
        
            ll2 = model_smaller.llf # Log-likelihood of Model 1
            df2 = model_smaller.df_model # Degrees of freedom of Model 1
        
            lsrt_stat=2*(ll1-ll2)
            df_diff=df1-df2
            p=chi2.sf(lsrt_stat,df_diff)
            #TODO just if converged
            if (model_larger.converged &model_smaller.converged):
                comps.append(dict({'reduced model':f_smaller,
                                   'model':f_larger,
                         'Reduced Model Log-Likelihood':round(ll2,3),
                        'Model Log-Likelihood':round(ll1,3),
                        'LRT Statistic':round(lsrt_stat,3),
                        'df_diff':int(df_diff),'p-value':p,      
                                   'Model df':int(df1),
                                 'pseudoR-squared': round(model_larger.prsquared,3),
                                  'bic':round(model_larger.bic,3),
                                  'aic':round(model_larger.aic,3)}))
            else:
                comps.append(dict({'reduced model':f_smaller,
                                   'model':f_larger,
                         'Reduced Model Log-Likelihood':np.nan,
                        'Model Log-Likelihood':np.nan,
                        'LRT Statistic':np.nan,
                        'df_diff':int(df_diff),
                                   'p-value':np.nan,      
                                   'Model df':int(df1),
                                 'pseudoR-squared': np.nan,
                                  'bic':np.nan,
                                  'aic':np.nan}))
        comps_df=pd.DataFrame(comps)
        params_df=pd.concat(params,axis=1).T.reset_index(drop=True)
        res=pd.concat([params_df,comps_df],axis=1) 
        if path is not None:
            res.to_csv(f'{path}/ProbitComp_results.csv')    
        return res, fits


def blind_chisq(data,covariates,predictedVal=None, path=None):
    chiTest=[]
    crossTabs=[]
    if predictedVal is None:
        combs = list(itertools.combinations(covariates, 2))
    else:
        combs=[(predictedVal, *c) for c in itertools.combinations(covariates, 2-1)]
        
    for x, y in combs:
        filename=f'{path}/chisqTests_{x}_{y}.csv'
        chi2, p, dof, ex = chi2_contingency(pd.crosstab(data[x],data[y]))
        # report chi-square test ==============
        print(f'{x}-{y}: p={p:.3f}')
        chiTest.append({'var':x,'chi2':chi2,'p':p,'dof':dof,'ex':ex})
        pd.DataFrame.from_dict(
            {x:['','chi-squared test','']}
        ).to_csv(filename, index=False)
        pd.DataFrame.from_dict({'var':[x],'chi2':[chi2],'p':[p],'dof':[dof]}).to_csv(
            filename,mode='a', index=False)
    
        proptab0=pd.crosstab(data[x],data[y])
        ex2=pd.DataFrame(ex)
        ex2.index=proptab0.index
        ex2.columns=proptab0.columns
        residuals=(proptab0-ex2)/np.sqrt(ex2)
        # report table counts ==============
    
        
        crossTabs.append(proptab0)
        
        pd.DataFrame.from_dict(
            {'table':['','observed quantities','']}
        ).to_csv(filename,mode='a', index=False)
        proptab0.to_csv(filename,mode='a')
        
        
        
        
        # report proportions by columns ==================    
        proptab=pd.crosstab(data[x],data[y],normalize='columns')
        pd.DataFrame.from_dict(
            {'table':['','percentage by columns','']}
        ).to_csv(filename,mode='a', index=False)
        proptab.to_csv(filename,mode='a')
        
        pd.DataFrame.from_dict(
            {'table':['','percentage by columns + indication of significance','']}
        ).to_csv(filename,mode='a', index=False)
        sigResiduals = np.where(abs(residuals) > 3, "**", np.where( abs(residuals)>2,"*" , ''))
        proptab_perc=round(proptab*100,1)
        prop_res=proptab0.astype(str)+' ('+proptab_perc.astype(str) +'% '+sigResiduals.astype(str) +')'
        prop_res.to_csv(filename,mode='a')
        # report with residuals 
    
        pd.DataFrame.from_dict(
            {'table':['','percentage by columns + residuals','']}
        ).to_csv(filename,mode='a', index=False)
        proptab_perc=round(proptab*100,1)
        prop_res=proptab0.astype(str)+' ('+proptab_perc.astype(str) +'%, z= '+residuals.astype(str) +')'
        prop_res.to_csv(filename,mode='a')
    
        # report proportions by index ==============
        pd.DataFrame.from_dict(
            {'table':['','percentage by index','']}
        ).to_csv(filename,mode='a', index=False)
        proptab=pd.crosstab(data[x],data[y],normalize='index')
        proptab.to_csv(filename,mode='a')
    
        pd.DataFrame.from_dict(
            {'table':['','percentage by index + indication of significance','']}
        ).to_csv(filename,mode='a', index=False)
        proptab_perc=round(proptab*100,1)
        prop_res=proptab0.astype(str)+' ('+proptab_perc.astype(str) +'% '+sigResiduals.astype(str) +')'
        prop_res.to_csv(filename,mode='a')
    
        pd.DataFrame.from_dict(
            {'table':['','percentage by index + residuals','']}
        ).to_csv(filename,mode='a', index=False)
        proptab_perc=round(proptab*100,1)
        prop_res=proptab0.astype(str)+' ('+proptab_perc.astype(str) +'%, z= '+residuals.astype(str) +')'
        prop_res.to_csv(filename,mode='a')
    
        # expected frequencies ===============
        pd.DataFrame.from_dict(
            {'table':['','expected frequencies','']}
        ).to_csv(filename,mode='a', index=False)
        
        ex2.to_csv(filename,mode='a')
        
        # REPORT RESIDUALS ===========
        
        pd.DataFrame.from_dict(
            {'table':['','residuals','']}
        ).to_csv(filename,mode='a', index=False)
        residuals.to_csv(filename,mode='a')    

def reportProbit(fitsList,n, path, xlim=4):
    # get OR confidence intervals   
    fit=fitsList[n]
    if fit.converged==True:
        pvaluestab=pd.DataFrame(fit.pvalues)
        #pvaluestab.columns=[x for x in fit.model.endog_names]
        pvaluestab.columns=['pvalue']
        pvaluestab=pvaluestab.reset_index(names='parameter')
        estimate=fit.params
        estimate=pd.DataFrame(np.exp(estimate))
        estimate.columns=['estimate']
        #stimate.columns=v[1:]
        estimate=estimate.reset_index(names='parameter')
        
        betaCI=fit.conf_int()
        ORCI=pd.DataFrame(np.exp(betaCI))
        ORCI.columns=['lower','upper']
        ORCI=ORCI.reset_index(names=['parameter'])
        ORCI=pd.merge(ORCI, pvaluestab, on=['parameter'], how='left')
        ORCI=pd.merge(ORCI, estimate, on=['parameter'], how='left')
        ORCI['pv']=[round(x,3) if x>=0.001 else '<0.001' for x in ORCI.pvalue]
        # plot forest plot
        fig, ax = plt.subplots(figsize=(7, 10))  # Width, Height in inches
        forestplot(ORCI, 
               estimate="estimate",
               ll="lower",
               hl="upper",
               varlabel='parameter',
               vlines=True,
               vline_color="red",
               #vline_style="--",
               #group_by="classes", # Use the study names as labels
               pvalues="pvalue", # Optional: display p-values
               xticks=[0.1, 0.5, 1, 2,3,4], # Optional: set x-axis ticks
               xlim=[0.01, xlim], # Optional: set x-axis limits
               ylabel="severity",
               xlabel="Odds Ratio",
               color_alt_rows=True,  # Gray alternate rows
               sort=False,
               table=True,
               #annote=["est_ci"],  # columns to report on left of plot
               #annoteheaders=["N", "Est. (95% Conf. Int.)"],  # ^corresponding headers
               rightannote=["pv"],  # columns to report on right of plot 
               right_annoteheaders=["P-value"], 
               log_scale=True, # Common for odds ratios/hazard ratios
                ax=ax,
               flush=False,
               linecolor='blue',
               **{
                      "xline": 1,
                      "xlinestyle": (1, (10, 5)) # long dash for x-reference line
                  }
              )
        #fig.set_size_inches(5,10)
        plt.savefig(f'{path}/forestplot_Probit_severity_m{n}.png', bbox_inches="tight")
        
        plt.show()
        # report pseudo r-squared etc.
        return pd.DataFrame.from_dict({'pseudo-rsquared':[fit.prsquared], 'AIC':[fit.aic],'BIC':[fit.bic]})   
    else:
        return "model not converged"
# report ================
def reportMNlogit(mnList, n,path,xlim=3.5):
    fit=mnList['fits'][n]
    v=mnList['comparisons'].loc[n,'depVars']
    # get OR confidence intervals    
    pvaluestab=fit.pvalues
    pvaluestab.columns=v[1:]
    pvaluestab=pvaluestab.reset_index(names='parameter')
    estimate=fit.params
    estimate=np.exp(estimate)
    estimate.columns=v[1:]
    estimate=estimate.reset_index(names='parameter')
    plong = pd.melt(pvaluestab, id_vars=['parameter'], value_vars=v[1:],
                  var_name='classes', value_name='pvalue')
    estlong = pd.melt(estimate, id_vars=['parameter'], value_vars=v[1:],
                  var_name='classes', value_name='estimate')
    betaCI=fit.conf_int()
    ORCI=np.exp(betaCI)
    ORCI=ORCI.reset_index(names=['classes','parameter'])
    ORCI['class_parameter']=[y+'_'+x for x,y in zip(ORCI.parameter,ORCI.classes)]
    ORCI=pd.merge(ORCI, plong, on=['classes','parameter'], how='left')
    ORCI=pd.merge(ORCI, estlong, on=['classes','parameter'], how='left')
    ORCI['pv']=[round(x,3) if x>=0.001 else '<0.001' for x in ORCI.pvalue]
    # plot forest plot
    fig, ax = plt.subplots(figsize=(7, 30))  # Width, Height in inches
    forestplot(ORCI, 
           estimate="estimate",
           ll="lower",
           hl="upper",
           varlabel='parameter',
           vlines=True,
           vline_color="red",
           #vline_style="--",
           #group_by="classes", # Use the study names as labels
           groupvar="classes",
           pvalues="pvalue", # Optional: display p-values
           xticks=[0.1, 0.5, 1, 2,3,4], # Optional: set x-axis ticks
           xlim=[0.01, xlim], # Optional: set x-axis limits
           ylabel="Variable class",
           xlabel="Odds Ratio",
           color_alt_rows=True,  # Gray alternate rows
           sort=False,
           table=True,
           #annote=["est_ci"],  # columns to report on left of plot
           #annoteheaders=["N", "Est. (95% Conf. Int.)"],  # ^corresponding headers
           rightannote=["pv"],  # columns to report on right of plot 
           right_annoteheaders=["P-value"], 
           log_scale=True, # Common for odds ratios/hazard ratios
            ax=ax,
           flush=False,
           linecolor='blue',
           **{
                  "xline": 1,
                  "xlinestyle": (1, (10, 5)) # long dash for x-reference line
              }
          )
    #fig.set_size_inches(5,10)
    plt.savefig(f'{path}/forestplot_MNLogit_phenotype_m{n}.png', bbox_inches="tight")
    
    plt.show()
    
    # report pseudo r-squared etc.
    return pd.DataFrame.from_dict({'pseudo-rsquared':[fit.prsquared], 'AIC':[fit.aic],'BIC':[fit.bic]})   

class exploreCovariateAssoc():
    def __init__(self,obj, k, ref_class, path,
                 data=None,
                          varTestChi=None,
                           phenotype_models_str=None,
                          hosp_models_str=None,
                           severe_models_str=None,
                           severe_curv_models_str=None
                         ):
        if not isinstance(obj, optimize_stepMix):
            print("The object provided is not a RAPID gridsearch object for LCA")
            sys.exit(1)
        if data is None:
            data=obj.data0
            
        predicted=obj.predict_k(k=k)
        self.k=k
        self.LCAGridSearch=obj
        self.ref_class=ref_class
        self.varTestChi=varTestChi
        self.phenotype_models_str=phenotype_models_str
        self.hosp_models_str=hosp_models_str
        self.severe_models_str=severe_models_str
        self.severe_curv_models_str=severe_curv_models_str
        
        data['predicted']=[f'cl_{x}' for x in predicted]
        data['predicted']= data['predicted'].astype('category')    
        data['classLCA']=['almost asymptomatic' if x==f'cl_{ref_class}' else x for x in data['predicted'] ]
        self.data=data
    
        if path is not None:
                if not os.path.exists(path):
                    os.makedirs(path)
        
        fig, ax = plt.subplots(figsize=(12,6))
        for label, df in data.groupby('predicted'):
            ax=df.Age.plot(kind="kde", ax=ax, label=label)
        plt.legend()
        plt.xlim(0,15)
        plt.savefig(f'{path}/Age_classes.png')
        plt.show()
        if varTestChi!=None:
            if path is not None:
                if not os.path.exists(f'{path}/testChi'):
                    os.makedirs(f'{path}/testChi')
            print(f'============================ Chi-squared tests ==========================')
            blind_chisq(data=data,covariates=varTestChi,predictedVal='predicted', path=f'{path}/testChi')
        if phenotype_models_str!=None:
            if path is not None:
                if not os.path.exists(f'{path}/MNLogit'):
                    os.makedirs(f'{path}/MNLogit')
            print(f'============================ Multinomial logit with nested models ==========================')
            comps_df, pardict2, fits, modelObjs=compareModels_MNLogit(models_str=phenotype_models_str,data=data,path=f'{path}/MNLogit')
        
            display(comps_df)
            display(pardict2)
            self.res={'multinomial_pheotype':{'comparisons':comps_df,'parameters':pardict2,'fits':fits,'modelObjs':modelObjs}}
        if hosp_models_str!=None:
            if path is not None:            
                if not os.path.exists(f'{path}/ProbitM_hosp'):
                    os.makedirs(f'{path}/ProbitM_hosp')
            print(f'============================ Hospitalization: Probit model with nested models ==========================')
            res_hosp, fits_hosp=compareProbitModels(hosp_models_str,data=data,path=f'{path}/ProbitM_hosp')
            display(res_hosp)
            self.res['hosp_probit']={'results':res_hosp,'fits':fits_hosp}
        if severe_models_str!=None:
            if path is not None:
                if not os.path.exists(f'{path}/ProbitM_severe'):
                    os.makedirs(f'{path}/ProbitM_severe')
            print(f'============================ Severoty: Probit model with nested models ==========================')
            res_severe, fits_severe=compareProbitModels(severe_models_str,data=data,path=f'{path}/ProbitM_severe')
            self.res['severe']={'results':res_severe, 'fits':fits_severe}
            display(res_severe)
        if severe_curv_models_str!=None:
            if path is not None:
                if not os.path.exists(f'{path}/ProbitM_sevCurv'):
                    os.makedirs(f'{path}/ProbitM_sevCurv')
            print(f'============================ severity curvilinear: Probit model with nested models ==========================')
            res_severeCurv, fits_severeCurv=compareProbitModels(severe_curv_models_str,data=data,path=f'{path}/ProbitM_sevCurv')
            self.res['severeCurvilinear']={'results':res_severeCurv, 'fits':fits_severeCurv}
            display(res_severeCurv)
    
        
