# Portable paths only; original exploratory model logic unchanged.
# Run ONLY on a prepared development dataset; never score sealed test participants.
"""Exploratory comparison of proposed 25 features. No repo changes, no sealed test evaluation.
All validation is group-disjoint LOPO. Imputation and scaling fitted inside each training fold.
Historical self-reported wellness variables have unverified survey submission times.
"""
from pathlib import Path
import json,sys,math,time
import numpy as np,pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,mean_absolute_error,mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'02_yorgunluk_tahmini'))
from fatigue_models import add_personal_features,PERSONAL_COLUMNS,FATIGUE_HISTORY_COLUMNS
OUT=Path(__file__).resolve().parents[1]/'outputs'/'experiment_25_feature_groups'; OUT.mkdir(parents=True,exist_ok=True)
DEV=['p01','p02','p03','p04','p06','p07','p08','p09','p10','p11','p13','p14','p15','p16']
raw=pd.read_csv(Path(__file__).resolve().parents[1]/'outputs'/'fatigue_daily.csv',low_memory=False)
d=raw.loc[raw.participant.isin(DEV)].copy();del raw
d['date']=pd.to_datetime(d.date)
d=add_personal_features(d).sort_values(['participant','date']).reset_index(drop=True)
assert len(d)==1512 and d.participant.nunique()==14 and d.participant.isin(['p05','p12']).sum()==0 and (d.fatigue>=4).sum()==113
old13=sorted(PERSONAL_COLUMNS+FATIGUE_HISTORY_COLUMNS+['fat_trend','fat_vs_pers'])
new13=sorted((set(old13)-{'p_median','p_mean14'})|{'history_wellness_readiness_lag1','history_wellness_soreness_lag1'})
groups={
 'core9':['history_wellness_fatigue_lag1','history_wellness_fatigue_mean7','history_wellness_fatigue_mean28','p_mean','p_std','fat_trend','fat_vs_pers','history_wellness_readiness_lag1','history_wellness_soreness_lag1'],
 'sleep4':['dv_sleep_score_overall_score','dw_sleep_score_overall_score','history_sleep_efficiency_mean7','history_sleep_minutesAsleep_mean7'],
 'heart3':['history_cardio_resting_heart_rate_mean7','dv_cardio_resting_heart_rate','dw_cardio_resting_heart_rate'],
 'training3':['history_srpe_load_sum7','history_srpe_acute_chronic_ratio','dw_srpe_load'],
 'wellness3':['history_wellness_stress_lag1','dw_wellness_mood','history_wellness_sleep_quality_mean7'],
 'activity3':['history_activity_very_active_minutes_sum7','history_movement_steps_total_mean7','dow'],
 'morning2':['morning_sleep_score_overall_score_delta28','morning_sleep_minutesAsleep_delta28'],
}
all25=sum((groups[g] for g in ['core9','sleep4','heart3','training3','wellness3','activity3']),[])
assert len(all25)==len(set(all25))==25
missing=set(all25+groups['morning2']+old13+new13)-set(d.columns)
assert not missing,missing
cases={
 'original13':old13,
 'new13_previous':new13,
 'core9':groups['core9'],
 'core9_plus_sleep':groups['core9']+groups['sleep4'],
 'core9_plus_heart':groups['core9']+groups['heart3'],
 'core9_plus_training':groups['core9']+groups['training3'],
 'core9_plus_wellness':groups['core9']+groups['wellness3'],
 'core9_plus_activity':groups['core9']+groups['activity3'],
 'core9_plus_sleep_heart':groups['core9']+groups['sleep4']+groups['heart3'],
 'core9_plus_sleep_heart_training':groups['core9']+groups['sleep4']+groups['heart3']+groups['training3'],
 'core9_plus_sleep_heart_training_wellness':groups['core9']+groups['sleep4']+groups['heart3']+groups['training3']+groups['wellness3'],
 'all25':all25,
 'all25_plus_morning2_EXPLORATORY':all25+groups['morning2'],
 'new13_plus_morning2_EXPLORATORY':new13+groups['morning2'],
}
y=(d.fatigue>=4).to_numpy(dtype=int);ys=d.fatigue.to_numpy(dtype=float)
parts=d.participant.to_numpy();folds=list(LeaveOneGroupOut().split(d,y,parts))
res=[]; preds=[]
start=time.time()
for name,columns in cases.items():
 x=d[columns].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=float)
 p=np.full(len(d),np.nan)
 for tr,va in folds:
  model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=.01,class_weight='balanced',max_iter=3000,solver='lbfgs',random_state=42))
  model.fit(x[tr],y[tr]);p[va]=model.predict_proba(x[va])[:,1]
 assert np.isfinite(p).all()
 top=math.ceil(len(y)*.10); ix=np.argsort(-p,kind='stable')[:top];tp=int(y[ix].sum())
 local=[]
 for id,g in d.groupby('participant'):
  inds=g.index.to_numpy();yy=y[inds]
  if yy.sum()>=3 and (len(yy)-yy.sum())>=3: local.append(roc_auc_score(yy,p[inds]))
 res.append(dict(model=name,n_features=len(columns),roc_auc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)),tp_at10=tp,recall_at10=tp/int(y.sum()),precision_at10=tp/top,within_person_auc=float(np.mean(local)),n_within_person=len(local)))
 preds.append(pd.DataFrame({'model':name,'participant':parts,'date':d.date.dt.strftime('%Y-%m-%d'),'high':y,'score':p}))
 print(name, 'AP',round(res[-1]['average_precision'],4),'AUC',round(res[-1]['roc_auc'],4),'top',tp,'within',round(res[-1]['within_person_auc'],4),flush=True)
print('Classifier seconds:',round(time.time()-start,1),flush=True)
# GBM regression, small reasoned subset of cases: original, previous new13, core9, all25, no morning, optional morning.
reg_cases={}
reg=[]
for name,columns in reg_cases.items():
 x=d[columns].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=float);pred=np.full(len(d),np.nan)
 for tr,va in folds:
  model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),GradientBoostingRegressor(loss='absolute_error',n_estimators=150,learning_rate=.05,max_depth=2,subsample=.7,min_samples_leaf=30,random_state=0))
  model.fit(x[tr],ys[tr]);pred[va]=np.clip(model.predict(x[va]),1,5)
 mask=ys>=4
 reg.append(dict(model=name,n_features=len(columns),mae=mean_absolute_error(ys,pred),rmse=np.sqrt(mean_squared_error(ys,pred)),high_fatigue_mae=mean_absolute_error(ys[mask],pred[mask])))
 print('GBM',name,reg[-1],flush=True)
res_df=pd.DataFrame(res);reg_df=pd.DataFrame(reg)
res_df.to_csv(OUT/'classification_metrics.csv',index=False)
reg_df.to_csv(OUT/'regression_metrics.csv',index=False)
pd.concat(preds,ignore_index=True).to_csv(OUT/'classification_oof_predictions.csv',index=False)
missing_df=pd.DataFrame([{'group':k,'feature':col,'missing_pct':float(d[col].isna().mean()*100),'distinct':int(d[col].nunique(dropna=True))} for k,v in groups.items() for col in v])
missing_df.to_csv(OUT/'feature_missingness.csv',index=False)
report={'scope':'exploratory only; same development participants reused for extensive feature selection; not independent confirmation','n':len(d),'participants':len(DEV),'high':int(y.sum()),'sealed_test': 'p05 and p12 EXCLUDED','model_classifier':'Balanced LogisticRegression C=.01 with per-training-fold median imputation and standardization','regression':'GBM absolute error n_estimators=150 depth=2 train-fold imputation','validation':'Leave one participant out, 14 folds; pooled top 10 percent retrospective ranking','groups':groups,'cases':cases,'classification':res,'regression':reg,'warnings':['Do not use feature selection rankings as independent performance evidence','Morning sleep data may be unavailable at prediction time: exploratory ONLY','Prior-day subjective questionnaires timestamp not independently verified','Computed p_* features use historical known labels of same participant; prospective use requires those records available','Not original frozen seven-fold E3 validation split; no sealed test access','Cross-validation with expanded candidates compared on the same repeatedly queried development cohort can be optimistic']}
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print('RESULT_DIR',OUT)
