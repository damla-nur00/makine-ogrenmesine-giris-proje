# Paths adapted for this repository; study model logic retained.
"""Exploratory weighted GBM; never evaluates inferred sealed test subjects.
Not the verified frozen protocol. Standalone from supplied daily features.
"""
import sys, json, copy
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'02_yorgunluk_tahmini'))
from fatigue_models import add_personal_features, model_feature_columns, _candidate_models, shrunk_personal_median
src=ROOT/'outputs'/'fatigue_daily.csv'; out=ROOT/'outputs'/'weighted_gbm_exploratory';out.mkdir(parents=True,exist_ok=True)
cols=pd.read_csv(src,nrows=0).columns.tolist()
dev_ids=['p01','p02','p03','p04','p06','p07','p08','p09','p10','p11','p13','p14','p15','p16']
# Avoid ever loading p05, p12 outcomes into working dataframe.
rows=[]
for chunk in pd.read_csv(src,chunksize=250,dtype={'participant':str}):
 rows.append(chunk.loc[chunk.participant.isin(dev_ids)])
df=pd.concat(rows,ignore_index=True)
assert len(df)==1512 and df.participant.nunique()==14
assert df.fatigue.between(1,5).all()
df=add_personal_features(df).sort_values(['participant','date']).reset_index(drop=True)
features=model_feature_columns(df)
weights=[1.0,2.0,4.0]
records=[]
for fold,(tr,va) in enumerate(GroupKFold(n_splits=7).split(df,groups=df.participant),1):
 assert set(df.iloc[tr].participant).isdisjoint(set(df.iloc[va].participant))
 train=df.iloc[tr];valid=df.iloc[va];global_median=float(train.fatigue.median())
 y=train.fatigue.to_numpy()
 models=_candidate_models(features)
 base=shrunk_personal_median(valid,global_median)
 for k,pred in [('baseline_personal_median',base)]:
  for j,v in enumerate(va):records.append({'fold':fold,'participant':df.loc[v,'participant'],'date':df.loc[v,'date'],'fatigue':float(df.loc[v,'fatigue']),'model':k,'prediction':float(np.clip(pred[j],1,5))})
 for w in weights:
  model=copy.deepcopy(models['gbm_abs_personal'])
  # make_pipeline produces 'gradientboostingregressor' final step
  model.fit(train[features],y,gradientboostingregressor__sample_weight=np.where(y>=4,w,1.))
  preds=np.clip(model.predict(valid[features]),1,5)
  for j,v in enumerate(va):records.append({'fold':fold,'participant':df.loc[v,'participant'],'date':df.loc[v,'date'],'fatigue':float(df.loc[v,'fatigue']),'model':f'gbm_high_weight_{w:g}','prediction':float(preds[j])})
res=pd.DataFrame(records)
res.to_csv(out/'oof_predictions.csv',index=False)
stats=[]
for name,g in res.groupby('model',sort=False):
 y=g.fatigue.to_numpy(); z=g.prediction.to_numpy(); high=y>=4; alerts=z>=3.5
 rounded=np.rint(z).astype(int); truths=np.rint(y).astype(int)
 stats.append({'model':name,'n':len(g),'people':g.participant.nunique(),'mae':mean_absolute_error(y,z),'rmse':float(np.sqrt(mean_squared_error(y,z))), 'high_mae':mean_absolute_error(y[high],z[high]), 'high_true':int(high.sum()),'high_predicted':int(alerts.sum()),'high_recall':recall_score(high,alerts,zero_division=0),'high_precision':precision_score(high,alerts,zero_division=0),'high_f1':f1_score(high,alerts,zero_division=0),'macro_f1':f1_score(truths,rounded,labels=[1,2,3,4,5],average='macro',zero_division=0),'auc_from_regression_score':roc_auc_score(high,z),'ap_from_regression_score':average_precision_score(high,z)})
metrics=pd.DataFrame(stats);metrics.to_csv(out/'metrics.csv',index=False)
print(metrics.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
# Participant level paired comparison, same held-out days.
pivot=res.pivot(index=['participant','date'],columns='model',values='prediction')
truth=res.loc[res.model=='baseline_personal_median'].set_index(['participant','date']).fatigue.reindex(pivot.index)
people=pivot.index.get_level_values('participant')
ref=np.abs(pivot['gbm_high_weight_1'].to_numpy()-truth.to_numpy())
comp=[];rng=np.random.default_rng(42)
ids=np.array(sorted(people.unique()));counts=np.array([sum(people==p) for p in ids]);draws=rng.integers(0,len(ids),size=(5000,len(ids)))
for name in ['gbm_high_weight_2','gbm_high_weight_4']:
 delta=np.abs(pivot[name].to_numpy()-truth.to_numpy())-ref
 sums=np.array([delta[people==p].sum() for p in ids]);distribution=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
 comp.append({'strategy':name,'reference':'gbm_high_weight_1','mae_difference':float(np.mean(delta)),'improved_people':int(np.sum(sums<0)),'bootstrap_95ci':[float(v) for v in np.quantile(distribution,[.025,.975])]})
print('PAIRED',json.dumps(comp,indent=2))
report={'scope':'exploratory GroupKFold 7 folds, NOT verified original frozen split','sealed_users':'p05 and p12 entirely excluded before feature calculations','feature_set':'Original gbm_abs_personal 11 core features','model':'GradientBoostingRegressor(loss=absolute_error,n_estimators=150,learning_rate=0.05,max_depth=2,subsample=0.7,min_samples_leaf=30,random_state=0)','weights':weights,'high_label':'fatigue >=4','classification_threshold':'regression prediction >=3.5; fixed before checking outcomes','metrics':stats,'paired_uncertainty':comp}
(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
