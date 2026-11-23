"""PMData Fitbit ve PMSys kayıtlarını sabah tahmini için günlük tabloya çevirir."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

TIMEZONE = "Europe/Oslo"
CUTOFF = 8
LABEL_END = 12
ID = "participant"
DATE = "date"
TARGET = "fatigue"
SLEEP = ["sleep_minutes", "sleep_efficiency"]
HISTORY = ["steps", "calories", "resting_hr", "moderate_minutes", "very_active_minutes",
           "exercise_minutes", "srpe_load"]
MEASURES = SLEEP + HISTORY


def pmsys_time(values):
    return pd.to_datetime(values.str.replace(" ", "", regex=False), utc=True,
                          format="mixed", errors="coerce").dt.tz_convert(TIMEZONE)


def local_dates(timestamps):
    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_localize(None)
    return timestamps.dt.normalize()


def read_json(path):
    if path.is_file():
        with path.open(encoding="utf-8") as source:
            return pd.DataFrame(json.load(source))
    return pd.DataFrame()


def by_day(path, column, aggregate):
    rows = read_json(path)
    if rows.empty:
        return pd.Series(name=column, dtype=float)
    date = pd.to_datetime(rows["dateTime"], format="mixed", errors="coerce")
    values = rows["value"]
    if column == "resting_hr":
        values = values.map(lambda value: value.get("value") if isinstance(value, dict) else value)
    groups = pd.to_numeric(values, errors="coerce").groupby(local_dates(date))
    result = groups.sum(min_count=1) if aggregate == "sum" else groups.mean()
    return result.rename(column)


def sleep_by_day(path):
    rows = read_json(path)
    if rows.empty:
        return pd.DataFrame(columns=SLEEP, index=pd.DatetimeIndex([]))
    end = pd.to_datetime(rows["endTime"], format="mixed", errors="coerce")
    start = pd.to_datetime(rows["startTime"], format="mixed", errors="coerce")
    date = local_dates(end)
    overnight = (start < date + pd.Timedelta(hours=5)) & (end > start)
    available = overnight & (end <= date + pd.Timedelta(hours=CUTOFF))
    rows = rows.loc[available].copy().assign(date=date, end=end)
    # En uzun tamamlanmış oturumu seç. Öğleden sonraki şekerlemeleri kullanma.
    rows = rows.sort_values(["minutesAsleep", "end"]).drop_duplicates(DATE, keep="last")
    return rows.set_index(DATE).rename(columns={
        "minutesAsleep": "sleep_minutes", "efficiency": "sleep_efficiency"
    })[SLEEP]


def exercises(path):
    rows = read_json(path)
    if rows.empty:
        return pd.Series(name="exercise_minutes", dtype=float)
    start = pd.to_datetime(rows["startTime"], format="mixed", errors="coerce")
    duration = pd.to_numeric(rows["duration"], errors="coerce")
    end = start + pd.to_timedelta(duration, unit="ms")
    end_day = local_dates(end)
    active = pd.to_numeric(rows.get("activeDuration", duration), errors="coerce").fillna(duration)
    if "lastModified" in rows:
        modified = pd.to_datetime(rows["lastModified"], format="mixed", errors="coerce")
        active = active.where(modified.isna() | (modified <= end_day + pd.Timedelta(days=1, hours=CUTOFF)))
    return (active / 60000).groupby(end_day).sum(min_count=1).rename("exercise_minutes")


def srpe(path):
    if not path.is_file():
        return pd.Series(name="srpe_load", dtype=float)
    rows = pd.read_csv(path)
    date = local_dates(pmsys_time(rows["end_date_time"].astype("string")))
    rating = pd.to_numeric(rows["perceived_exertion"], errors="coerce")
    minutes = pd.to_numeric(rows["duration_min"], errors="coerce")
    return (rating * minutes).groupby(date).sum(min_count=1).rename("srpe_load")


def labels(path):
    rows = pd.read_csv(path)
    time = pmsys_time(rows["effective_time_frame"].astype("string"))
    raw = pd.to_numeric(rows[TARGET], errors="coerce")
    in_window = (time.dt.hour >= CUTOFF) & (time.dt.hour < LABEL_END)
    valid = in_window & raw.between(1, 5)
    rows = pd.DataFrame({DATE: local_dates(time[valid]), "time": time[valid],
                         TARGET: 6 - raw[valid]})
    return rows.sort_values("time").drop_duplicates(DATE).set_index(DATE)[TARGET]


def prepare(root):
    frames = []
    for path in sorted(Path(root).rglob("pmsys/wellness.csv")):
        person = path.parent.parent
        fitbit = person / "fitbit"
        daily = pd.concat([
            labels(path),
            sleep_by_day(fitbit / "sleep.json"),
            by_day(fitbit / "steps.json", "steps", "sum"),
            by_day(fitbit / "calories.json", "calories", "sum"),
            by_day(fitbit / "resting_heart_rate.json", "resting_hr", "mean"),
            by_day(fitbit / "moderately_active_minutes.json", "moderate_minutes", "sum"),
            by_day(fitbit / "very_active_minutes.json", "very_active_minutes", "sum"),
            exercises(fitbit / "exercise.json"),
            srpe(person / "pmsys" / "srpe.csv"),
        ], axis=1).rename_axis(DATE)
        daily[ID] = person.name
        frames.append(daily.reset_index())
    if not frames:
        raise ValueError("Verilen klasörde pXX/pmsys/wellness.csv bulunamadı.")
    return pd.concat(frames, ignore_index=True)[[ID, DATE, *MEASURES, TARGET]]


def features(daily):
    """Gün boşluklarını korur. Bir önceki satır yerine bir önceki takvim gününü kullanır."""
    daily = daily.copy()
    daily[DATE] = pd.to_datetime(daily[DATE], errors="raise").dt.normalize()
    if daily[[ID, DATE]].isna().any().any() or daily.duplicated([ID, DATE]).any():
        raise ValueError("Her katılımcı ve tarih çifti geçerli ve tekil olmalı.")
    for col in MEASURES + [TARGET]:
        if col not in daily:
            daily[col] = np.nan
        daily[col] = pd.to_numeric(daily[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    output = []
    for person, rows in daily.groupby(ID, sort=False):
        rows = rows.set_index(DATE).sort_index()
        dates = rows.index
        rows = rows.asfreq("D")
        x = rows[SLEEP].copy()
        past = rows[MEASURES + [TARGET]].shift()
        for col in MEASURES + [TARGET]:
            x[f"{col}_lag1"] = past[col]
            x[f"{col}_mean3"] = past[col].rolling(3, min_periods=2).mean()
            x[f"{col}_mean7"] = past[col].rolling(7, min_periods=3).mean()
        x["sleep_minutes_delta7"] = x["sleep_minutes"] - x["sleep_minutes_mean7"]
        x["resting_hr_delta7"] = x["resting_hr_lag1"] - x["resting_hr_mean7"]
        x[ID] = person
        x[TARGET] = rows[TARGET]
        output.append(x.loc[dates].reset_index())
    if not output:
        raise ValueError("Tahmin için en az bir günlük kayıt gerekli.")
    return pd.concat(output, ignore_index=True)
