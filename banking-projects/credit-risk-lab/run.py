"""Default-risk benchmark with disjoint fit/calibration/validation/test groups."""
import argparse,hashlib,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,brier_score_loss,precision_recall_curve,recall_score
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from banklab.common import *
from banklab.features import make_features,probability_to_score,REQUIRED

def main(source):
 raw=pd.read_excel(source,header=1)
 assert raw.shape==(30000,25) and raw.ID.is_unique
 y=raw['default payment next month'].astype(int);x=make_features(raw)
 assert set(y.unique())=={0,1}
 # Identical financial profiles remain in a single split, even if labels conflict.
 groups=pd.util.hash_pandas_object(raw[REQUIRED],index=False).astype(str)
 folds=np.full(len(raw),-1)
 for fold,(_,idx) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(x,y,groups)):folds[idx]=fold
 masks={'train':folds>=3,'calibration':folds==2,'validation':folds==1,'test':folds==0}
 assert np.all(sum(masks.values())==1)
 group_sets={name:set(groups[mask]) for name,mask in masks.items()}
 for a in group_sets:
  for b in group_sets:
   if a!=b:assert group_sets[a].isdisjoint(group_sets[b])
 tr,ca,va,te=[masks[k] for k in ['train','calibration','validation','test']]
 candidates={
  'logistic':make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=2000,random_state=42)),
  'gradient_boosting':HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,l2_regularization=20,early_stopping=False,random_state=42)}
 fitted={};comparison=[]
 for name,model in candidates.items():
  model.fit(x.loc[tr],y[tr]);cal=fit_calibrator(model,x.loc[ca],y[ca]);p=predict(model,cal,x.loc[va])
  comparison.append({'model':name,**metrics(y[va],p),'calibration_slope':float(cal.coef_[0,0])});fitted[name]=(model,cal)
 comparison=pd.DataFrame(comparison).sort_values('brier');selected=comparison.iloc[0].model;model,cal=fitted[selected]
 vp=predict(model,cal,x.loc[va]);threshold=float(np.quantile(vp,.8))
 p=predict(model,cal,x.loc[te]);actual=y[te].to_numpy();base=np.full(len(p),float(y[tr].mean()))
 result={'rows':len(raw),'split_rows':{k:int(v.sum()) for k,v in masks.items()},'selected_model':selected,'test':metrics(actual,p),'constant_prior_brier':brier_score_loss(actual,base),'roc_auc_bootstrap_95_ci':interval_bootstrap(actual,p,roc_auc_score),'validation_80pct_threshold':threshold,'test_review_share':float((p>=threshold).mean()),'test_recall_at_threshold':recall_score(actual,p>=threshold),'score_psi_train_test':psi(predict(model,cal,x.loc[tr]),p),'raw_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
 # One-month illustrative proxy: PD x positive latest bill balance x assumed LGD.
 exposure=raw.loc[te,'BILL_AMT1'].clip(lower=0).to_numpy();scenarios=[]
 for multiplier in [1,1.25,1.5]:
  stressed=1/(1+np.exp(-(logit(p)+np.log(multiplier))))
  for lgd in [.35,.45,.60]:scenarios.append({'pd_odds_multiplier':multiplier,'assumed_lgd':lgd,'bill_exposure_proxy_twd':float(exposure.sum()),'one_month_loss_proxy_twd':float(np.sum(stressed*exposure*lgd))})
 scenarios=pd.DataFrame(scenarios);scenarios.to_csv(REPORT/'stress_sensitivity.csv',index=False)
 result['base_one_month_loss_proxy_twd']=float(scenarios.loc[(scenarios.pd_odds_multiplier==1)&(scenarios.assumed_lgd==.45),'one_month_loss_proxy_twd'].iloc[0])
 score=probability_to_score(p);risk=pd.DataFrame({'probability':p,'actual':actual,'custom_score':score})
 risk['risk_decile']=pd.qcut(risk.probability.rank(method='first'),10,labels=range(1,11))
 decile=risk.groupby('risk_decile',observed=True).agg(rows=('actual','size'),mean_pd=('probability','mean'),observed_default=('actual','mean'),mean_score=('custom_score','mean'));decile.to_csv(REPORT/'risk_deciles.csv')
 audit=[]
 sub=raw.loc[te].copy();sub['actual']=actual;sub['pd']=p;sub['flag']=p>=threshold;sub['age_band']=pd.cut(sub.AGE,[0,29,49,100],labels=['up to 29','30 to 49','50+'])
 for attribute in ['SEX','age_band']:
  for value,frame in sub.groupby(attribute,observed=True):
   if len(frame)>=200:audit.append({'attribute':attribute,'group':str(value),'rows':len(frame),'observed_default':float(frame.actual.mean()),'mean_pd':float(frame.pd.mean()),'review_share':float(frame.flag.mean()),'recall':recall_score(frame.actual,frame.flag,zero_division=0)})
 pd.DataFrame(audit).to_csv(REPORT/'subgroup_audit.csv',index=False)
 importance=permutation_importance(model,x.loc[va],y[va],scoring='roc_auc',n_repeats=3,random_state=42,n_jobs=2)
 imp=pd.DataFrame({'feature':x.columns,'validation_auc_drop':importance.importances_mean}).sort_values('validation_auc_drop',ascending=False);imp.to_csv(REPORT/'validation_importance.csv',index=False)
 # Transparent challenger coefficients, irrespective of which model is selected.
 lr=fitted['logistic'][0];pd.DataFrame({'feature':x.columns,'standardised_coefficient':lr[-1].coef_[0]}).sort_values('standardised_coefficient').to_csv(REPORT/'logistic_coefficients.csv',index=False)
 comparison.to_csv(REPORT/'validation_comparison.csv',index=False)
 pd.DataFrame([{'split':k,'rows':int(v.sum()),'default_rate':float(y[v].mean()),'profile_groups':len(group_sets[k])} for k,v in masks.items()]).to_csv(REPORT/'split_summary.csv',index=False)
 dump('metrics.json',result)
 dump('data_audit.json',{'missing_cells':int(raw.isna().sum().sum()),'duplicate_financial_profiles':int(groups.duplicated().sum()),'group_overlap':0,'protected_or_demographic_columns_excluded':['SEX','AGE','EDUCATION','MARRIAGE'],'feature_names':x.columns.tolist(),'chronological_validation_available':False})
 # Row-level predictions and the fitted estimator remain local, outside the published reports.
 pd.DataFrame({'source_id':raw.loc[te,'ID'],'actual':actual,'pd':p,'score':score}).to_csv(ARTIFACT/'test_predictions.csv',index=False)
 pd.DataFrame({'source_id':raw.ID,'profile_group':groups,'split':np.select([tr,ca,va,te],['train','calibration','validation','test'],default='unassigned')}).to_csv(ARTIFACT/'split_manifest.csv',index=False)
 joblib.dump({'model':model,'calibrator':cal,'features':x.columns.tolist(),'threshold':threshold},ARTIFACT/'model.joblib')
 fig,axs=plt.subplots(2,2,figsize=(13,9));fig.suptitle('Credit Risk: Held-out Probability Validation',fontsize=19,weight='bold')
 frac,mean=calibration_curve(actual,p,n_bins=10,strategy='quantile');axs[0,0].plot(mean,frac,'o-',color='#167D8D');axs[0,0].plot([0,1],[0,1],'--',color='grey');axs[0,0].set(title='Calibration',xlabel='Mean predicted default',ylabel='Observed default')
 pr,re,_=precision_recall_curve(actual,p);axs[0,1].plot(re,pr,color='#4275B5');axs[0,1].axhline(actual.mean(),ls='--',color='grey');axs[0,1].set(title='Precision–recall',xlabel='Recall',ylabel='Precision')
 decile[['mean_pd','observed_default']].rename(columns={'mean_pd':'Mean predicted PD','observed_default':'Observed default'}).plot.bar(ax=axs[1,0],color=['#167D8D','#E6A23C']);axs[1,0].set(title='Risk deciles',xlabel='Increasing predicted risk',ylabel='Default share')
 for lgd,frame in scenarios.groupby('assumed_lgd'):axs[1,1].plot(frame.pd_odds_multiplier,frame.one_month_loss_proxy_twd/1e6,'o-',label=f'LGD assumption {lgd:.0%}')
 axs[1,1].set(title='Illustrative one-month loss sensitivity',xlabel='Assumed PD odds multiplier',ylabel='Proxy loss (TWD millions)');axs[1,1].legend();chart(fig,'validation-dashboard')
 text=f'''# Measured findings\n\nValidation Brier score selected **{selected}**. Untouched test ROC AUC is **{result['test']['roc_auc']:.3f}** and average precision is **{result['test']['average_precision']:.3f}** against prevalence {actual.mean():.3f}. Brier score is **{result['test']['brier']:.4f}**, compared with {result['constant_prior_brier']:.4f} for the training-prior baseline.\n\nThe validation threshold flags {result['test_review_share']:.1%} of test profiles and captures {result['test_recall_at_threshold']:.1%} of observed defaults. This is a research review policy, not a lending decision.\n\nThe baseline one-month loss proxy is TWD {result['base_one_month_loss_proxy_twd']:,.0f} at an assumed 45% LGD. Bill balance is only a proxy for exposure; this is not an IFRS 9, lifetime, regulatory-capital or validated economic forecast. Odds shifts are sensitivity assumptions.\n\nRecommendation: retain the benchmark as evidence of modelling and calibration methods. Recent local, out-of-time data, validated exposure/recovery inputs, policy review and broader fairness testing would be needed before any banking use. The source is one historical Taiwanese cohort and cannot establish UK portfolio performance.\n'''
 (REPORT/'FINDINGS.md').write_text(text);print(json.dumps(result,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=ROOT/'data/credit_default.xls');main(parser.parse_args().source)
