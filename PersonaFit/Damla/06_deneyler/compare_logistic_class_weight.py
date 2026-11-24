# Paths adapted for this repository; study model logic retained.
"""Exploratory comparison; NOT the frozen E3 split, never loads sealed p05/p12 rows."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score,precision_score,recall_score,f1_score,confusion_matrix
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '02_yorgunluk_tahmini'))
from fatigue_models import add_personal_features, PERSONAL_COLUMNS, FATIGUE_HISTORY_COLUMNS
root=Path(__file__).resolve().parents[1]; out=root/'outputs'/'compare_logistic_class_weight';out.mkdir(parents=True,exist_ok=True)
ids=['p01','p02','p03','p04','p06','p07','p08','p09','p10','p11','p13','p14','p15','p16']
parts=[]
for chunk in pd.read_csv(root/'outputs'/'fatigue_daily.csv',chunksize=250):
    parts.append(chunk.loc[chunk.participant.isin(ids)].copy())
d=add_personal_features(pd.concat(parts,ignore_index=True)).sort_values(['participant','date']).reset_index(drop=True)
assert len(d)==1512 and d.participant.nunique()==14 and not d.duplicated(['participant','date']).any()
cols=sorted(PERSONAL_COLUMNS+FATIGUE_HISTORY_COLUMNS+['fat_trend','fat_vs_pers'])
assert len(cols)==13 and all(c in d.columns for c in cols)
records=[]
for fold,(train_idx,val_idx) in enumerate(GroupKFold(7).split(d,groups=d.participant),1):
    tr=d.iloc[train_idx];va=d.iloc[val_idx]; y=(tr.fatigue>=4).astype(int)
    assert set(tr.participant).isdisjoint(set(va.participant)) and y.nunique()==2
    for name,weight in [('logistic_unweighted',None),('logistic_balanced','balanced')]:
        clf=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=1,max_iter=3000,class_weight=weight,solver='lbfgs',random_state=42))
        clf.fit(tr[cols].replace([np.inf,-np.inf],np.nan),y)
        scores=clf.predict_proba(va[cols].replace([np.inf,-np.inf],np.nan))[:,1]
        for (participant,date,fatigue),score in zip(va[['participant','date','fatigue']].itertuples(index=False,name=None),scores):
            records.append({'fold':fold,'participant':participant,'date':date,'fatigue':fatigue,'high':int(fatigue>=4),'model':name,'score':score})
pred=pd.DataFrame(records);pred.to_csv(out/'oof_predictions.csv',index=False)
metrics=[]
for name,g in pred.groupby('model'):
    y=g.high.to_numpy(); score=g.score.to_numpy(); alerts=(score>=0.5).astype(int);k=int(np.ceil(.1*len(g)));top=np.zeros(len(g),dtype=int);top[np.argsort(-score,kind='mergesort')[:k]]=1
    person_aucs=[]
    for _,person in g.groupby('participant'):
        if person.high.sum()>=3 and (len(person)-person.high.sum())>=3:person_aucs.append(roc_auc_score(person.high,person.score))
    metrics.append({'model':name,'n':len(g),'high_n':int(y.sum()),'roc_auc':roc_auc_score(y,score),'average_precision':average_precision_score(y,score),'at_0_5_precision':precision_score(y,alerts,zero_division=0),'at_0_5_recall':recall_score(y,alerts,zero_division=0),'at_0_5_f1':f1_score(y,alerts,zero_division=0),'at_0_5_alerts':int(alerts.sum()),'at_0_5_true_positives':int(np.sum((alerts==1)&(y==1))),'top10_recall':recall_score(y,top,zero_division=0),'top10_precision':precision_score(y,top,zero_division=0),'top10_f1':f1_score(y,top,zero_division=0),'top10_true_positives':int(np.sum((top==1)&(y==1))),'within_person_auc_mean':float(np.mean(person_aucs)),'within_person_auc_eligible':len(person_aucs)})
met=pd.DataFrame(metrics);met.to_csv(out/'metrics.csv',index=False)
print(met.to_string(index=False,float_format=lambda n:f'{n:.5f}'))
# Paired participant bootstrap of differences balanced minus unweighted, no re-fitting or tuning.
wide=pred.pivot(index=['participant','date'],columns='model',values='score'); truths=pred.loc[pred.model=='logistic_balanced'].set_index(['participant','date']).high.reindex(wide.index)
users=wide.index.get_level_values('participant'); user_ids=np.array(sorted(users.unique()));rng=np.random.default_rng(42);draws=rng.integers(0,len(user_ids),size=(2000,len(user_ids)))
bootstrap=[]
for indices in draws:
    selected=np.concatenate([np.flatnonzero(users==user_ids[i]) for i in indices]);yt=truths.to_numpy()[selected];
    if np.unique(yt).size<2:continue
    bootstrap.append((roc_auc_score(yt,wide['logistic_balanced'].to_numpy()[selected])-roc_auc_score(yt,wide['logistic_unweighted'].to_numpy()[selected]),average_precision_score(yt,wide['logistic_balanced'].to_numpy()[selected])-average_precision_score(yt,wide['logistic_unweighted'].to_numpy()[selected])))
bootstrap=np.array(bootstrap)
report={'scope':'Exploratory 7-fold GroupKFold; original E3 frozen split unavailable and NOT reconstructed','n':len(d),'participants':ids,'sealed_test_excluded':['p05','p12'],'feature_group':'Original E3 A: 13 historical/personal fatigue features','models':'LogisticRegression C=1 lbfgs max_iter=3000, imputer median then scaler; class_weight either None or balanced','positive_rule':'fatigue>=4 (3.5 negative)','threshold_note':'Fixed 0.5 classification threshold; scores not calibrated probabilities. Top10 selects 10% highest-scored OOF days across entire development set; not prospective within-person alerting.','results':metrics,'paired_balanced_minus_unweighted_bootstrap_95ci':{'roc_auc':np.quantile(bootstrap[:,0],[.025,.975]).tolist(),'average_precision':np.quantile(bootstrap[:,1],[.025,.975]).tolist()},'limitations':['Not same frozen E3 split','Exploratory after previous E3 inspection','No model chosen based on this evaluation','No sealed test used']}
(out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print('BOOTSTRAP',report['paired_balanced_minus_unweighted_bootstrap_95ci'])
