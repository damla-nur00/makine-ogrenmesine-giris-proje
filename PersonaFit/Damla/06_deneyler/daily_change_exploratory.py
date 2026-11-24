# Adapted paths to PersonaFit/Damla; experimental model logic otherwise retained.
"""Standalone exploratory LOPO analyses, no project or sealed test changes."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from sys import path
path.insert(0,str(Path(__file__).resolve().parents[1]/'02_yorgunluk_tahmini'))
from fatigue_models import add_personal_features,PERSONAL_COLUMNS,FATIGUE_HISTORY_COLUMNS
R=Path(__file__).resolve().parents[1]; O=R/'outputs'/'daily_change_exploratory'; O.mkdir(parents=True,exist_ok=True)
DEV=['p01','p02','p03','p04','p06','p07','p08','p09','p10','p11','p13','p14','p15','p16']
d=pd.read_csv(R/'outputs'/'fatigue_daily.csv',low_memory=False)
d=d[d.participant.isin(DEV)].copy();d['date']=pd.to_datetime(d['date']);d=d.sort_values(['participant','date']).reset_index(drop=True)
d['previous_day_score']=d.groupby('participant').fatigue.shift()
d['consecutive_day']=(d.date-d.groupby('participant').date.shift()).dt.days.eq(1)
d=add_personal_features(d).sort_values(['participant','date']).reset_index(drop=True)
assert len(d)==1512 and d.participant.nunique()==14 and (d.fatigue>=4).sum()==113
A=sorted(PERSONAL_COLUMNS+FATIGUE_HISTORY_COLUMNS+['fat_trend','fat_vs_pers']);assert len(A)==13
SENS=['history_movement_steps_total_lag1','history_movement_steps_total_mean7','history_cardio_resting_heart_rate_lag1','history_cardio_resting_heart_rate_mean7','history_cardio_heart_rate_mean_lag1','history_cardio_heart_rate_mean_mean7','history_sleep_minutesAsleep_lag1','history_sleep_minutesAsleep_mean7','history_sleep_efficiency_lag1','history_sleep_efficiency_mean7']
DERV=['dv_movement_steps_total','dv_cardio_resting_heart_rate','dv_cardio_heart_rate_mean']
assert all(c in d.columns for c in A+SENS+DERV),set((A+SENS+DERV))-set(d.columns)
feature_sets={'A_history':A,'A_plus_sensor_lags':A+SENS,'A_plus_sensor_deviations':A+SENS+DERV}
experiments={'high_fatigue':(d,(d.fatigue>=4).to_numpy(dtype=int))}
jump=d.loc[d.consecutive_day].copy().reset_index(drop=True)
experiments['jump_at_least_1']=(jump,((jump.fatigue-jump.previous_day_score)>=1).to_numpy(dtype=int))
experiments['jump_at_least_2']=(jump,((jump.fatigue-jump.previous_day_score)>=2).to_numpy(dtype=int))
models=['A_history','A_plus_sensor_lags','A_plus_sensor_deviations']
records=[];summary=[]
def measure(g):
    y=g.actual.to_numpy(int); s=g.score.to_numpy(float)
    if len(np.unique(y))<2:return {}
    n=len(g);k=int(np.ceil(n*.1));flags=np.zeros(n,bool);flags[np.argsort(-s,kind='stable')[:k]]=True
    tp=int(np.sum(flags&(y==1)));denom=int(np.sum(y)); person_auc=[]
    for _,h in g.groupby('participant'):
        yy=h.actual.to_numpy(int)
        if yy.sum()>=3 and (len(yy)-yy.sum())>=3:person_auc.append(roc_auc_score(yy,h.score))
    return {'n':n,'positive_days':denom,'roc_auc':roc_auc_score(y,s),'average_precision':average_precision_score(y,s),'recall_at_10pct':tp/denom,'precision_at_10pct':tp/k,'f1_at_10pct':2*tp/(denom+k),'tp_at_10pct':tp,'within_person_auc':np.mean(person_auc) if person_auc else np.nan,'within_person_eligible':len(person_auc)}
for task,(frame,y) in experiments.items():
    if task=='jump_at_least_2':
        # Too few positives for reliable participant-level evaluation; descriptive only.
        pass
    gp=frame.participant.to_numpy();
    for name in models:
        X=frame[feature_sets[name]].replace([np.inf,-np.inf],np.nan).to_numpy(float)
        oof=np.full(len(frame),np.nan)
        for tr,va in LeaveOneGroupOut().split(X,y,gp):
            model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=.01,class_weight='balanced',max_iter=3000,random_state=42))
            model.fit(X[tr],y[tr]);oof[va]=model.predict_proba(X[va])[:,1]
        assert np.isfinite(oof).all()
        sub=pd.DataFrame({'task':task,'model':name,'participant':gp,'date':frame.date.dt.strftime('%Y-%m-%d'),'actual':y,'score':oof})
        records.append(sub)
        summary.append({'task':task,'model':name,**measure(sub)})
        print(task,name,summary[-1],flush=True)
res=pd.DataFrame(summary)
res.to_csv(O/'daily_change_metrics.csv',index=False)
pd.concat(records).to_csv(O/'daily_change_oof_predictions.csv',index=False)
print('DONE')
# Separate meaningful alert target: transition from below 4 to 4+ on consecutive days.
onset=jump[jump.previous_day_score<4].copy().reset_index(drop=True)
y=(onset.fatigue>=4).astype(int).to_numpy();gp=onset.participant.to_numpy()
for name in models:
    X=onset[feature_sets[name]].replace([np.inf,-np.inf],np.nan).to_numpy(float)
    oof=np.full(len(onset),np.nan)
    for tr,va in LeaveOneGroupOut().split(X,y,gp):
        clf=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=.01,class_weight='balanced',max_iter=3000,random_state=42))
        clf.fit(X[tr],y[tr]);oof[va]=clf.predict_proba(X[va])[:,1]
    sub=pd.DataFrame({'task':'onset_4_from_below4','model':name,'participant':gp,'date':onset.date.dt.strftime('%Y-%m-%d'),'actual':y,'score':oof})
    records.append(sub);summary.append({'task':'onset_4_from_below4','model':name,**measure(sub)})
    print('ONSET',name,summary[-1],flush=True)
res=pd.DataFrame(summary);res.to_csv(O/'daily_change_metrics.csv',index=False)
pd.concat(records).to_csv(O/'daily_change_oof_predictions.csv',index=False)
