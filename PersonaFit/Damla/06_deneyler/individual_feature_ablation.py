# Portable paths only; original exploratory model logic unchanged.
# Run ONLY on a prepared development dataset; never score sealed test participants.
"""Independent exploratory LOPO ablation. Does not alter project code or test split."""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'02_yorgunluk_tahmini'))
from fatigue_models import add_personal_features,PERSONAL_COLUMNS,FATIGUE_HISTORY_COLUMNS
R=Path(__file__).resolve().parents[1]; O=R/'outputs'/'individual_feature_ablation'; O.mkdir(parents=True,exist_ok=True)
DEV=['p01','p02','p03','p04','p06','p07','p08','p09','p10','p11','p13','p14','p15','p16']
d=pd.read_csv(R/'outputs'/'fatigue_daily.csv',low_memory=False)
d=d[d.participant.isin(DEV)].copy()
d['date']=pd.to_datetime(d.date)
d=add_personal_features(d).sort_values(['participant','date']).reset_index(drop=True)
assert len(d)==1512 and d.participant.nunique()==14 and int((d.fatigue>=4).sum())==113
A=sorted(PERSONAL_COLUMNS+FATIGUE_HISTORY_COLUMNS+['fat_trend','fat_vs_pers']);assert len(A)==13
W=['history_wellness_stress_lag1','history_wellness_soreness_lag1','history_wellness_readiness_lag1','history_wellness_sleep_quality_lag1']
y=(d.fatigue>=4).astype(int).to_numpy();groups=d.participant.to_numpy();folds=list(LeaveOneGroupOut().split(d,y,groups))

def evaluate(name,cols,typ,changed):
 X=d[cols].replace([np.inf,-np.inf],np.nan).to_numpy(float)
 prob=np.full(len(y),np.nan)
 for tr,va in folds:
  model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=.01,class_weight='balanced',max_iter=3000,random_state=42))
  model.fit(X[tr],y[tr]);prob[va]=model.predict_proba(X[va])[:,1]
 k=int(np.ceil(len(y)*.1));top=np.zeros(len(y),bool);top[np.argsort(-prob,kind='stable')[:k]]=True;tp=int(np.sum(top&(y==1)))
 w=[]
 for p,g in d.groupby('participant',sort=True):
  inds=g.index.to_numpy(); yy=y[inds]
  if yy.sum()>=3 and (len(yy)-yy.sum())>=3:w.append(roc_auc_score(yy,prob[inds]))
 row={'experiment':name,'type':typ,'changed_feature':changed,'features_n':len(cols),'roc_auc':roc_auc_score(y,prob),'average_precision':average_precision_score(y,prob),'tp_at10':tp,'recall_at10':tp/y.sum(),'precision_at10':tp/k,'f1_at10':2*tp/(k+int(y.sum())),'within_person_auc':float(np.mean(w)),'within_people_n':len(w)}
 return row,prob
spec=[('original_A',A,'baseline','')]
spec.extend((f'remove_A_{x}',[f for f in A if f!=x],'remove_A',x) for x in A)
spec.extend((f'add_{x}',sorted(A+[x]),'add_single',x) for x in W)
spec.append(('A_plus_all_4',sorted(A+W),'all_four',''))
spec.extend((f'leaveout_4_{x}',sorted(A+[f for f in W if f!=x]),'remove_from_all_four',x) for x in W)
rows=[];pr=[]
for i,(name,features,kind,changed) in enumerate(spec,1):
 row,proba=evaluate(name,features,kind,changed);rows.append(row)
 pr.append(pd.DataFrame({'participant':d.participant,'date':d.date.dt.strftime('%Y-%m-%d'),'high':y,'experiment':name,'score':proba}))
 print(f'{i:02}/{len(spec)} {name}: AUC {row["roc_auc"]:.4f} AP {row["average_precision"]:.4f} TP@10 {row["tp_at10"]} within {row["within_person_auc"]:.4f}',flush=True)
results=pd.DataFrame(rows)
orig=results.iloc[0];full=results[results.experiment=='A_plus_all_4'].iloc[0]
for metric in ['roc_auc','average_precision','tp_at10','within_person_auc','f1_at10']:
 results[f'delta_{metric}_vs_original']=results[metric]-orig[metric]
 results[f'delta_{metric}_vs_all_four']=results[metric]-full[metric]
results.to_csv(O/'individual_feature_ablation_metrics.csv',index=False)
pd.concat(pr,ignore_index=True).to_csv(O/'individual_feature_ablation_predictions.csv',index=False)
# Top-level summary note with full feature semantics
(O/'individual_feature_ablation_info.json').write_text(json.dumps({'n_rows':len(d),'n_participants':len(DEV),'n_positive':int(y.sum()),'sealed_test_excluded':['p05','p12'],'model':'LogisticRegression(C=0.01,class_weight=balanced), median imputation, standard scaler fitted only on train fold','validation':'LeaveOneGroupOut 14 folds; entirely exploratory after repeated use of same development cohort','primary_baseline_features':A,'added_wellness_features':W,'n_experiments':len(spec),'interpretation':'single-feature removal/addition is marginal and conditional on correlated features; not a causal importance measure; score comparison on reused dev cohort is optimistic','time_availability':'Survey timestamp availability not verified; do not deploy added survey features yet'},indent=2,ensure_ascii=False))
print('\nRESULTS\n'+results[['experiment','roc_auc','average_precision','tp_at10','within_person_auc','delta_average_precision_vs_original']].to_string(index=False,float_format=lambda v:f'{v:.4f}'))
