# Adapted paths to PersonaFit/Damla; experimental model logic otherwise retained.
"""Standalone research experiment; no changes to project code, no sealed test data."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut,GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score,average_precision_score
from catboost import CatBoostClassifier
from sys import path
path.insert(0,str(Path(__file__).resolve().parents[1]/'02_yorgunluk_tahmini'))
from fatigue_models import add_personal_features,PERSONAL_COLUMNS,FATIGUE_HISTORY_COLUMNS
R=Path(__file__).resolve().parents[1]; O=R/'outputs'/'nested_model_selection'; O.mkdir(parents=True,exist_ok=True)
DEV=['p01','p02','p03','p04','p06','p07','p08','p09','p10','p11','p13','p14','p15','p16']
d=pd.read_csv(R/'outputs'/'fatigue_daily.csv',low_memory=False)
d=d[d.participant.isin(DEV)].copy()
d=add_personal_features(d).sort_values(['participant','date']).reset_index(drop=True)
assert len(d)==1512 and d.participant.nunique()==14 and (d.fatigue>=4).sum()==113
cols=sorted(PERSONAL_COLUMNS+FATIGUE_HISTORY_COLUMNS+['fat_trend','fat_vs_pers']); assert len(cols)==13
X=d[cols].replace([np.inf,-np.inf],np.nan).to_numpy(float)
y=(d.fatigue>=4).astype(int).to_numpy();groups=d.participant.to_numpy()
models=['logistic_C1','logistic_C001','catboost_A']
def fit_score(name,tr,va):
  if name=='catboost_A':
    # Fixed new candidate parameters; this is NOT claimed to reproduce previous CatBoost exactly.
    imp=SimpleImputer(strategy='median',keep_empty_features=True); xt=imp.fit_transform(X[tr]); xv=imp.transform(X[va]); clf=CatBoostClassifier(iterations=200,depth=3,learning_rate=.035,l2_leaf_reg=5,loss_function='Logloss',auto_class_weights='Balanced',verbose=False,thread_count=1,random_seed=42); clf.fit(xt,y[tr]); return clf.predict_proba(xv)[:,1]
  else:
    model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=1 if name=='logistic_C1' else .01,class_weight='balanced',max_iter=3000,random_state=42))
  model.fit(X[tr],y[tr]); return model.predict_proba(X[va])[:,1]
def met(yy,ss):
  k=int(np.ceil(len(yy)*.1)); top=np.zeros(len(yy),bool);top[np.argsort(-ss,kind='stable')[:k]]=True
  tp=int(np.sum(top & (yy==1)))
  return {'auc':float(roc_auc_score(yy,ss)),'ap':float(average_precision_score(yy,ss)),'recall_at10':float(tp/sum(yy)),'precision_at10':float(tp/k),'f1_at10':float(2*tp/(k+sum(yy))),'top10_true_positives':tp}
outer=LeaveOneGroupOut(); records=[]; selects=[]
start=time.time()
for j,(ot,ov) in enumerate(outer.split(X,y,groups),1):
  inner=GroupKFold(n_splits=4)
  inner_preds={name:np.full(len(ot),np.nan) for name in models}
  for it,iv in inner.split(X[ot],y[ot],groups[ot]):
    for name in models:
      inner_preds[name][iv]=fit_score(name,ot[it],ot[iv])
  inner_score={name:met(y[ot],inner_preds[name]) for name in models}
  # Predefine single selection rule: highest inner OOF AP; tie -> AUC then simpler model
  winner=max(models,key=lambda n:(inner_score[n]['ap'],inner_score[n]['auc'],-models.index(n)))
  pred={name:fit_score(name,ot,ov) for name in models}
  pred['nested_selected']=pred[winner]
  selects.append({'held_out':str(groups[ov][0]),'selected':winner,**{f'{m}_inner_ap':round(inner_score[m]['ap'],6) for m in models}})
  for name,ss in pred.items():
    for ii,s in zip(ov,ss):records.append({'participant':groups[ii],'date':str(d.date.iloc[ii]),'high':int(y[ii]),'model':name,'score':float(s)})
  print(f'{j:2d}/14 {groups[ov][0]} winner={winner} inner_APs='+','.join(f'{m}:{inner_score[m]["ap"]:.3f}' for m in models)+f' {time.time()-start:.1f}s',flush=True)
pred=pd.DataFrame(records); pred.to_csv(O/'nested_oof_predictions.csv',index=False)
select=pd.DataFrame(selects);select.to_csv(O/'nested_selections.csv',index=False)
rows=[]
for name,g in pred.groupby('model'):
  m=met(g.high.to_numpy(),g.score.to_numpy())
  auc=[]
  for p,h in g.groupby('participant'):
    if h.high.sum()>=3 and (len(h)-h.high.sum())>=3:auc.append(roc_auc_score(h.high,h.score))
  rows.append({'model':name,**m,'within_person_auc':float(np.mean(auc)),'n_within_auc':len(auc)})
result=pd.DataFrame(rows).sort_values('ap',ascending=False)
result.to_csv(O/'nested_metrics.csv',index=False)
print('\nMETRICS\n'+result.to_string(index=False,float_format=lambda z:f'{z:.4f}'),flush=True)
print('\nSELECTIONS\n'+select.selected.value_counts().to_string(),flush=True)
report={'design':'Nested LOPO (14 outer held-out participant folds), inner GroupKFold(4) grouped by participant; selection based on inner OOF AP only. Exploratory after candidate discovery on same 14 dev people, not independent test.','n':len(d),'people':14,'high':int(y.sum()),'held_out_test':['p05','p12'],'features':cols,'model_params':'LogisticRegression class_weight balanced C=1 or .01; CatBoost iteration=200 depth=3 learning_rate=.035 l2=5 balanced auto_class_weights','selection_counts':select.selected.value_counts().to_dict(),'results':rows,'limitations':['Small number of participants and rare positives','Nested selection on development data previously used for model discovery','Outer LOPO scores from different fitted models; high risk scores uncalibrated','Global top10 is descriptive not a deployable person-level alarm threshold','Cannot conclude superiority without uncertainty analysis or truly untouched test']}
(O/'nested_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
