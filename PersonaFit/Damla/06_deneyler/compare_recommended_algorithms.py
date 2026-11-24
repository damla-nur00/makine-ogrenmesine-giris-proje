# Portable paths only; original exploratory model logic unchanged.
# Run ONLY on a prepared development dataset; never score sealed test participants.
"""Exploratory fixed-configuration grouped CV; NO sealed test, NO repository changes.
Candidates chosen after previous exploratory feature search: not independent validation.
"""
import os,sys,math,time,json
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import RandomForestClassifier,GradientBoostingRegressor
from sklearn.metrics import roc_auc_score,average_precision_score,mean_absolute_error
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'02_yorgunluk_tahmini'))
from fatigue_models import add_personal_features,PERSONAL_COLUMNS,FATIGUE_HISTORY_COLUMNS
OUT=Path(__file__).resolve().parents[1]/'outputs'/'compare_recommended_algorithms'; OUT.mkdir(parents=True,exist_ok=True)
DEV=['p01','p02','p03','p04','p06','p07','p08','p09','p10','p11','p13','p14','p15','p16']
raw=pd.read_csv(Path(__file__).resolve().parents[1]/'outputs'/'fatigue_daily.csv',low_memory=False)
d=raw.loc[raw.participant.isin(DEV)].copy();del raw
d['date']=pd.to_datetime(d.date);d=add_personal_features(d).sort_values(['participant','date']).reset_index(drop=True)
assert len(d)==1512 and d.participant.nunique()==14 and not d.participant.isin(['p05','p12']).any() and (d.fatigue>=4).sum()==113
old13=sorted(PERSONAL_COLUMNS+FATIGUE_HISTORY_COLUMNS+['fat_trend','fat_vs_pers'])
new13=sorted((set(old13)-{'p_median','p_mean14'})|{'history_wellness_readiness_lag1','history_wellness_soreness_lag1'})
core9=['history_wellness_fatigue_lag1','history_wellness_fatigue_mean7','history_wellness_fatigue_mean28','p_mean','p_std','fat_trend','fat_vs_pers','history_wellness_readiness_lag1','history_wellness_soreness_lag1']
sleep4=['dv_sleep_score_overall_score','dw_sleep_score_overall_score','history_sleep_efficiency_mean7','history_sleep_minutesAsleep_mean7']
heart3=['history_cardio_resting_heart_rate_mean7','dv_cardio_resting_heart_rate','dw_cardio_resting_heart_rate']
training3=['history_srpe_load_sum7','history_srpe_acute_chronic_ratio','dw_srpe_load']
wellness3=['history_wellness_stress_lag1','dw_wellness_mood','history_wellness_sleep_quality_mean7']
activity3=['history_activity_very_active_minutes_sum7','history_movement_steps_total_mean7','dow']
feature_sets={'new13':new13,'core9_plus_sleep':core9+sleep4,'all25':core9+sleep4+heart3+training3+wellness3+activity3}
y=(d.fatigue>=4).to_numpy(dtype=int);yf=d.fatigue.to_numpy(dtype=float);g=d.participant.to_numpy()
folds=list(LeaveOneGroupOut().split(d,y,g))
def fit_model(kind, train_y, columns):
 if kind=='logistic_l2':return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=.01,class_weight='balanced',max_iter=3000,random_state=42))
 if kind=='logistic_l1':return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=.1,penalty='l1',solver='liblinear',class_weight='balanced',max_iter=3000,random_state=42))
 if kind=='elasticnet':return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=.1,penalty='elasticnet',l1_ratio=.5,solver='saga',class_weight='balanced',max_iter=3000,tol=.001,random_state=42))
 if kind=='random_forest':return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),RandomForestClassifier(n_estimators=180,max_depth=4,min_samples_leaf=10,max_features='sqrt',class_weight='balanced_subsample',n_jobs=1,random_state=42))
 if kind=='xgboost':return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),XGBClassifier(n_estimators=140,max_depth=2,learning_rate=.04,min_child_weight=20,reg_lambda=20,subsample=.8,colsample_bytree=.85,scale_pos_weight=(len(train_y)-train_y.sum())/max(train_y.sum(),1),tree_method='hist',n_jobs=1,random_state=42,eval_metric='logloss'))
 if kind=='lightgbm':return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),LGBMClassifier(n_estimators=140,max_depth=2,num_leaves=4,min_child_samples=35,learning_rate=.04,reg_lambda=20,subsample=.8,subsample_freq=1,colsample_bytree=.85,scale_pos_weight=(len(train_y)-train_y.sum())/max(train_y.sum(),1),verbosity=-1,n_jobs=1,random_state=42))
 raise ValueError(kind)
methods=['logistic_l2','logistic_l1','elasticnet','lightgbm','xgboost','random_forest']
predictions=[];results=[]

def summarize(name,p):
 k=math.ceil(len(y)*.1);chosen=np.argsort(-p,kind='stable')[:k];tp=int(y[chosen].sum());inside=[]
 for person in np.unique(g):
  idx=np.flatnonzero(g==person);ys=y[idx]
  if ys.sum()>=3 and (len(ys)-ys.sum())>=3:inside.append(float(roc_auc_score(ys,p[idx])))
 return dict(model=name,auc=roc_auc_score(y,p),ap=average_precision_score(y,p),top10_tp=tp,recall10=tp/y.sum(),precision10=tp/k,within_auc=np.mean(inside),within_n=len(inside))

# score baselines: numeric history levels only ranking scores, NOT probabilities.
for name,column in [('previous_fatigue','history_wellness_fatigue_lag1'),('personal_mean','p_mean')]:
 p=d[column].to_numpy(dtype=float)
 for tr,va in folds:
  fallback=np.nanmedian(yf[tr]);p[va]=np.where(np.isfinite(p[va]),p[va],fallback)
 results.append(summarize(name,p));predictions.append(pd.DataFrame({'model':name,'participant':g,'date':d.date.dt.strftime('%Y-%m-%d'),'y':y,'score':p}))
 print(name,results[-1],flush=True)

for sname,columns in feature_sets.items():
 x=d[columns].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=float)
 scored={m:np.full(len(y),np.nan) for m in methods}
 for fi,(tr,va) in enumerate(folds):
  for kind in methods:
   m=fit_model(kind,y[tr],columns)
   m.fit(x[tr],y[tr]);scored[kind][va]=m.predict_proba(x[va])[:,1]
  print('fold',sname,fi+1,'/14',flush=True)
 for kind,p in scored.items():
  name=f'{sname}__{kind}'
  results.append(summarize(name,p));predictions.append(pd.DataFrame({'model':name,'participant':g,'date':d.date.dt.strftime('%Y-%m-%d'),'y':y,'score':p}))
  print(name,'AP',round(results[-1]['ap'],4),'AUC',round(results[-1]['auc'],4),'TP',results[-1]['top10_tp'],flush=True)
 # fixed blends: scores not necessarily calibrated; this is rank-focused experiment.
 for tag,a,b in [('logistic_xgboost','logistic_l2','xgboost'),('logistic_lightgbm','logistic_l2','lightgbm')]:
  p=(scored[a]+scored[b])*.5;name=f'{sname}__blend_{tag}'
  results.append(summarize(name,p));predictions.append(pd.DataFrame({'model':name,'participant':g,'date':d.date.dt.strftime('%Y-%m-%d'),'y':y,'score':p}))
  print(name,'AP',round(results[-1]['ap'],4),'TP',results[-1]['top10_tp'],flush=True)

# Ordinal-ish level-based models; ONLY risk-rank score, no calibration.
for sname,columns in [('new13',new13),('core9_plus_sleep',core9+sleep4)]:
 x=d[columns].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=float)
 for model_name in ['ridge_regression','gbm_regression']:
  p=np.full(len(y),np.nan)
  for tr,va in folds:
   if model_name=='ridge_regression':m=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=1000))
   else:m=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),GradientBoostingRegressor(loss='absolute_error',n_estimators=150,learning_rate=.05,max_depth=2,subsample=.7,min_samples_leaf=30,random_state=0))
   m.fit(x[tr],yf[tr]);p[va]=np.clip(m.predict(x[va]),1,5)
  name=f'{sname}__{model_name}';row=summarize(name,p);row['fatigue_mae']=mean_absolute_error(yf,p);results.append(row);predictions.append(pd.DataFrame({'model':name,'participant':g,'date':d.date.dt.strftime('%Y-%m-%d'),'y':y,'score':p}))
  print(name,'AP',round(row['ap'],4),'MAE',round(row['fatigue_mae'],4),flush=True)

pd.DataFrame(results).to_csv(OUT/'metrics.csv',index=False)
pd.concat(predictions,ignore_index=True).to_csv(OUT/'oof_predictions.csv',index=False)
report={'type':'exploratory fixed-configuration LOPO; previously searched feature sets, not independent evidence','sealed_test':'p05 and p12 excluded','participants':len(np.unique(g)),'days':len(d),'high_days':int(y.sum()),'features':feature_sets,'models':methods,'metric_note':'ROC-AUC/average precision; top 10 percent of pooled OOF scores is retrospective, not clinical alert threshold','baseline_note':'raw fatigue level scores used for ranking, not calibrated probability','caution':'Multiple model comparisons may overfit development data; outer results should not be used as nested tuning objective. Survey submission times not independently verified.','results':results}
(OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,default=float))
print('DONE',OUT,flush=True)
