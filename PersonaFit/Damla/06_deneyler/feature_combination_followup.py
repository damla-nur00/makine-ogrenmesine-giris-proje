# Portable paths only; original exploratory model logic unchanged.
# Run ONLY on a prepared development dataset; never score sealed test participants.
"""Independent exploratory feature combinations; never writes to project files."""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'02_yorgunluk_tahmini'))
from fatigue_models import add_personal_features,PERSONAL_COLUMNS,FATIGUE_HISTORY_COLUMNS
R=Path(__file__).resolve().parents[1]; O=R/'outputs'/'feature_combination_followup'; O.mkdir(parents=True,exist_ok=True)
DEV=['p01','p02','p03','p04','p06','p07','p08','p09','p10','p11','p13','p14','p15','p16']
d=pd.read_csv(R/'outputs'/'fatigue_daily.csv',low_memory=False)
d=d[d.participant.isin(DEV)].copy();d['date']=pd.to_datetime(d.date)
d=add_personal_features(d).sort_values(['participant','date']).reset_index(drop=True)
assert len(d)==1512 and d.participant.nunique()==14 and (d.fatigue>=4).sum()==113
A=sorted(PERSONAL_COLUMNS+FATIGUE_HISTORY_COLUMNS+['fat_trend','fat_vs_pers'])
remove=['p_median','p_mean14']; add=['history_wellness_readiness_lag1','history_wellness_soreness_lag1']
assert len(A)==13 and len(set(A))==13
cases={
 'original_13':A,
 'remove_median_and_mean14':sorted(set(A)-set(remove)),
 'add_readiness':sorted(set(A)|{add[0]}),
 'add_soreness':sorted(set(A)|{add[1]}),
 'add_readiness_and_soreness':sorted(set(A)|set(add)),
 'remove_two_add_readiness':sorted((set(A)-set(remove))|{add[0]}),
 'remove_two_add_soreness':sorted((set(A)-set(remove))|{add[1]}),
 'remove_two_add_both':sorted((set(A)-set(remove))|set(add)),
}
y=(d.fatigue>=4).astype(int).to_numpy();groups=d.participant.to_numpy(); folds=list(LeaveOneGroupOut().split(d,y,groups))
allprobs={}; out=[]
for name,cols in cases.items():
 x=d[cols].replace([np.inf,-np.inf],np.nan).to_numpy(float); prob=np.full(len(d),np.nan)
 for tr,va in folds:
  mdl=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=.01,class_weight='balanced',max_iter=3000,random_state=42))
  mdl.fit(x[tr],y[tr]);prob[va]=mdl.predict_proba(x[va])[:,1]
 assert np.isfinite(prob).all()
 top_n=int(np.ceil(len(y)*.1));inds=np.argsort(-prob,kind='stable')[:top_n];tp=int(y[inds].sum());pred=prob>=.5
 wp=[]
 for p,g in d.groupby('participant'):
  ix=g.index.to_numpy();yy=y[ix]
  if yy.sum()>=3 and (len(yy)-yy.sum())>=3:wp.append(roc_auc_score(yy,prob[ix]))
 out.append(dict(experiment=name,n_features=len(cols),roc_auc=roc_auc_score(y,prob),average_precision=average_precision_score(y,prob),tp_at10=tp,recall_at10=tp/y.sum(),precision_at10=tp/top_n,f1_at10=2*tp/(top_n+y.sum()),within_person_auc=np.mean(wp),threshold05_recall=int((pred&(y==1)).sum())/y.sum(),threshold05_precision=int((pred&(y==1)).sum())/max(int(pred.sum()),1),threshold05_fp=int((pred&(y==0)).sum()),n_alarm05=int(pred.sum())))
 allprobs[name]=prob
 print(name,'AP=',round(out[-1]['average_precision'],4),'ROC=',round(out[-1]['roc_auc'],4),'TOP=',tp,'within=',round(out[-1]['within_person_auc'],4),flush=True)
metrics=pd.DataFrame(out);base=metrics.iloc[0]
for col in ['roc_auc','average_precision','tp_at10','within_person_auc']:
 metrics['delta_'+col]=metrics[col]-base[col]
metrics.to_csv(O/'feature_combination_metrics.csv',index=False)
predictions=d[['participant','date','fatigue']].copy();predictions['date']=predictions.date.dt.strftime('%Y-%m-%d');predictions['high']=y
for name,prob in allprobs.items():predictions[name]=prob
predictions.to_csv(O/'feature_combination_predictions.csv',index=False)
report={'setting':'Exploratory LOPO 14 groups; same dev people reused in many past searches, optimistic comparisons','n':len(d),'positive':int(y.sum()),'sealed_test_ids_omitted':['p05','p12'],'model':'LogisticRegression C=.01 class_weight=balanced, train-only median imputation + standard scaling','cases':cases,'notes':['No project/repository code modified','Survey timing for readiness/soreness is not independently verified; do not use in real-time until verified','Feature removal in trained ML classifier is not causal attribution','At 10% risk, pooled threshold across OOF is a retrospective ranking, not an operational alert threshold']}
(O/'feature_combination_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
