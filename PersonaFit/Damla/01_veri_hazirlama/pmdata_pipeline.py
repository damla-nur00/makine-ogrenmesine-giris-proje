"""PMData ingestion, leakage-aware daily features, and project metadata helpers."""

from __future__ import annotations

import ast
import math
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

import ijson
import numpy as np
import pandas as pd


MINUTE_FILES = {
    "movement_steps": ("steps.json", 1.0, True),
    "movement_calories": ("calories.json", 1.0, True),
    "movement_distance_m": ("distance.json", 0.01, True),
}

DAILY_FILES = {
    "activity_sedentary_minutes": "sedentary_minutes.json",
    "activity_lightly_active_minutes": "lightly_active_minutes.json",
    "activity_moderately_active_minutes": "moderately_active_minutes.json",
    "activity_very_active_minutes": "very_active_minutes.json",
    "cardio_resting_heart_rate": "resting_heart_rate.json",
}

SLEEP_SCORE_COLUMNS = (
    "overall_score",
    "composition_score",
    "revitalization_score",
    "duration_score",
    "deep_sleep_in_minutes",
    "resting_heart_rate",
    "restlessness",
)

SLEEP_SESSION_COLUMNS = (
    "minutesAsleep",
    "minutesAwake",
    "efficiency",
    "minutesToFallAsleep",
    "timeInBed",
)

SUM_FEATURES: set[str] = {
    "movement_steps_total",
    "movement_calories_total",
    "movement_distance_m_total",
    "movement_steps_minutes_observed",
    "movement_calories_minutes_observed",
    "movement_distance_minutes_observed",
    "activity_sedentary_minutes",
    "activity_lightly_active_minutes",
    "activity_moderately_active_minutes",
    "activity_very_active_minutes",
    "srpe_load",
    "srpe_duration_min",
    "srpe_sessions",
    "exercise_sessions",
    "exercise_duration_min",
    "exercise_active_duration_min",
    "exercise_calories",
    "exercise_steps",
}


def _iter_json_array(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as stream:
        yield from ijson.items(stream, "item")


def _record_day(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) < 10:
        return None
    day = value[:10]
    try:
        pd.Timestamp(day)
    except (TypeError, ValueError):
        return None
    return day


def _number(value: Any) -> float:
    while isinstance(value, dict):
        if "value" in value:
            value = value["value"]
        elif "bpm" in value:
            value = value["bpm"]
        else:
            return np.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if math.isfinite(number) else np.nan


def _as_series(values: dict[str, float]) -> pd.Series:
    if not values:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    result = pd.Series(values, dtype=float)
    result.index = pd.to_datetime(result.index)
    return result.sort_index()


def _finish_minute_aggregate(
    aggregate: dict[str, list[float]], prefix: str, scale: float
) -> dict[str, pd.Series]:
    days: dict[str, dict[str, float]] = {}
    for day, state in aggregate.items():
        total, count, maximum, _ = state
        days[day] = {
            f"{prefix}_total": total * scale,
            f"{prefix}_minutes_observed": count,
            f"{prefix}_max_per_minute": maximum * scale,
            f"{prefix}_coverage": min(count / 1440.0, 1.0),
        }

    names = {name for row in days.values() for name in row}
    return {
        name: _as_series({day: row[name] for day, row in days.items() if name in row})
        for name in names
    }


def _read_minute_file(
    path: Path, prefix: str, scale: float, audit: dict[str, Any]
) -> dict[str, pd.Series]:
    aggregate: dict[str, list[float]] = defaultdict(
        lambda: [0.0, 0.0, -math.inf, 0.0]
    )
    records = 0
    skipped = 0
    for row in _iter_json_array(path):
        records += 1
        day = _record_day(row.get("dateTime"))
        value = _number(row.get("value"))
        if day is None or not math.isfinite(value):
            skipped += 1
            continue
        state = aggregate[day]
        state[0] += value
        state[1] += 1
        state[2] = max(state[2], value)
        state[3] += value * value

    audit["files"][str(path)] = {"records": records, "skipped": skipped}
    return _finish_minute_aggregate(aggregate, prefix, scale)


def _read_heart_rate(path: Path, audit: dict[str, Any]) -> dict[str, pd.Series]:
    aggregate: dict[str, list[float]] = defaultdict(
        lambda: [0.0, 0.0, math.inf, -math.inf, 0.0, 0.0, 0.0]
    )
    records = 0
    skipped = 0
    for row in _iter_json_array(path):
        records += 1
        day = _record_day(row.get("dateTime"))
        value = row.get("value")
        bpm = _number(value.get("bpm")) if isinstance(value, dict) else np.nan
        confidence = _number(value.get("confidence")) if isinstance(value, dict) else np.nan
        if day is None or not math.isfinite(bpm):
            skipped += 1
            continue
        state = aggregate[day]
        state[0] += bpm
        state[1] += 1
        state[2] = min(state[2], bpm)
        state[3] = max(state[3], bpm)
        state[4] += bpm * bpm
        if math.isfinite(confidence):
            state[5] += confidence
            state[6] += 1

    audit["files"][str(path)] = {"records": records, "skipped": skipped}
    return {
        "cardio_heart_rate_mean": _as_series(
            {day: values[0] / values[1] for day, values in aggregate.items()}
        ),
        "cardio_heart_rate_std": _as_series(
            {
                day: math.sqrt(max(values[4] / values[1] - (values[0] / values[1]) ** 2, 0))
                for day, values in aggregate.items()
            }
        ),
        "cardio_heart_rate_min": _as_series(
            {day: values[2] for day, values in aggregate.items()}
        ),
        "cardio_heart_rate_max": _as_series(
            {day: values[3] for day, values in aggregate.items()}
        ),
        "cardio_heart_rate_samples": _as_series(
            {day: values[1] for day, values in aggregate.items()}
        ),
        "cardio_heart_rate_confidence_mean": _as_series(
            {
                day: values[5] / values[6]
                for day, values in aggregate.items()
                if values[6] > 0
            }
        ),
    }


def _read_daily_json(path: Path, feature: str, audit: dict[str, Any]) -> pd.Series:
    values: dict[str, list[float]] = defaultdict(list)
    records = 0
    skipped = 0
    for row in _iter_json_array(path):
        records += 1
        day = _record_day(row.get("dateTime"))
        value = _number(row.get("value"))
        if day is None or not math.isfinite(value):
            skipped += 1
            continue
        values[day].append(value)
    audit["files"][str(path)] = {"records": records, "skipped": skipped}
    return _as_series({day: float(np.mean(day_values)) for day, day_values in values.items()})


def _read_heart_rate_zones(path: Path, audit: dict[str, Any]) -> dict[str, pd.Series]:
    values: dict[str, dict[str, float]] = defaultdict(dict)
    records = 0
    skipped = 0
    for row in _iter_json_array(path):
        records += 1
        day = _record_day(row.get("dateTime"))
        entry = row.get("value")
        zones = entry.get("valuesInZones") if isinstance(entry, dict) else None
        if day is None or not isinstance(zones, dict):
            skipped += 1
            continue
        for name, minutes in zones.items():
            number = _number(minutes)
            if math.isfinite(number):
                feature = f"cardio_zone_{name.lower()}"
                values[feature][day] = number
            else:
                skipped += 1
    audit["files"][str(path)] = {"records": records, "skipped": skipped}
    return {name: _as_series(days) for name, days in values.items()}


def _read_sleep_score(
    path: Path, audit: dict[str, Any]
) -> tuple[
    dict[str, pd.Series],
    dict[pd.Timestamp, pd.Timestamp],
    dict[str, pd.Timestamp],
]:
    frame = pd.read_csv(path)
    if not {"timestamp", *SLEEP_SCORE_COLUMNS}.issubset(frame.columns):
        raise ValueError(f"{path} is missing required sleep score columns")
    record_count = len(frame)
    frame["_timestamp"] = pd.to_datetime(
        frame["timestamp"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    frame["day"] = frame["_timestamp"].dt.normalize()
    id_timestamps = {
        str(row["sleep_log_entry_id"]): row["_timestamp"]
        for _, row in frame.dropna(subset=["_timestamp"]).iterrows()
        if "sleep_log_entry_id" in frame.columns
    }
    frame = frame.dropna(subset=["day", "_timestamp"])
    frame = frame.sort_values("_timestamp").groupby("day", as_index=False).tail(1)
    result: dict[str, pd.Series] = {}
    for column in SLEEP_SCORE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        values = frame.set_index("day")[column].dropna()
        result[f"sleep_score_{column}"] = values.sort_index()
    timestamps = frame.set_index("day")["_timestamp"].to_dict()
    audit["files"][str(path)] = {
        "records": int(record_count),
        "skipped": int(record_count - len(frame)),
    }
    return result, timestamps, id_timestamps


def _read_sleep_sessions(
    path: Path, audit: dict[str, Any]
) -> tuple[
    dict[str, pd.Series],
    dict[pd.Timestamp, pd.Timestamp],
    dict[pd.Timestamp, str],
]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records = 0
    skipped = 0
    for row in _iter_json_array(path):
        records += 1
        day = _record_day(row.get("dateOfSleep"))
        if day is None:
            skipped += 1
            continue
        by_day[day].append(row)

    daily: dict[str, dict[str, float]] = {}
    timestamps: dict[pd.Timestamp, pd.Timestamp] = {}
    log_ids: dict[pd.Timestamp, str] = {}
    for day, sessions in by_day.items():
        main_sessions = [session for session in sessions if session.get("mainSleep") is True]
        candidates = main_sessions or sessions
        session = max(candidates, key=lambda item: _number(item.get("minutesAsleep")))
        log_ids[pd.Timestamp(day)] = str(session.get("logId", ""))
        raw_end_time = session.get("endTime")
        has_timezone = (
            isinstance(raw_end_time, str)
            and (
                raw_end_time.endswith("Z")
                or (
                    len(raw_end_time) > 19
                    and raw_end_time[19] in {"+", "-"}
                )
            )
        )
        if has_timezone:
            completion_time = pd.to_datetime(
                raw_end_time, errors="coerce", utc=True
            )
            if pd.notna(completion_time):
                timestamps[pd.Timestamp(day)] = completion_time.tz_convert(None)
        row: dict[str, float] = {}
        for column in SLEEP_SESSION_COLUMNS:
            value = _number(session.get(column))
            if math.isfinite(value):
                row[f"sleep_{column}"] = value
        levels = session.get("levels")
        summary = levels.get("summary") if isinstance(levels, dict) else None
        if isinstance(summary, dict):
            for stage in ("deep", "light", "rem", "wake"):
                item = summary.get(stage)
                minutes = _number(item.get("minutes")) if isinstance(item, dict) else np.nan
                if math.isfinite(minutes):
                    row[f"sleep_stage_{stage}_minutes"] = minutes
        if row:
            daily[day] = row

    audit["files"][str(path)] = {
        "records": records,
        "skipped": skipped,
        "days_with_sleep": len(daily),
    }
    names = {name for row in daily.values() for name in row}
    return (
        {
            name: _as_series(
                {day: row[name] for day, row in daily.items() if name in row}
            )
            for name in names
        },
        timestamps,
        log_ids,
    )


def _read_srpe(path: Path, audit: dict[str, Any]) -> dict[str, pd.Series]:
    frame = pd.read_csv(path)
    record_count = len(frame)
    required = {"end_date_time", "perceived_exertion", "duration_min"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain {sorted(required)}")
    frame["day"] = pd.to_datetime(
        frame["end_date_time"], errors="coerce", utc=True
    ).dt.tz_convert(None).dt.normalize()
    frame["perceived_exertion"] = pd.to_numeric(frame["perceived_exertion"], errors="coerce")
    frame["duration_min"] = pd.to_numeric(frame["duration_min"], errors="coerce")
    frame = frame.dropna(subset=["day", "perceived_exertion", "duration_min"])
    frame["srpe_load"] = frame["perceived_exertion"] * frame["duration_min"]
    daily = frame.groupby("day").agg(
        srpe_load=("srpe_load", "sum"),
        srpe_duration_min=("duration_min", "sum"),
        srpe_sessions=("srpe_load", "count"),
        srpe_rpe_mean=("perceived_exertion", "mean"),
    )
    if not daily.empty:
        calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        daily = daily.reindex(calendar, fill_value=0)
    audit["files"][str(path)] = {
        "records": int(record_count),
        "skipped": int(record_count - len(frame)),
    }
    return {name: daily[name].sort_index() for name in daily.columns}


def _activity_category(name: Any) -> str:
    normalized = str(name).strip().lower()
    if any(token in normalized for token in ("walk", "hike")):
        return "walk"
    if any(token in normalized for token in ("run", "treadmill")):
        return "run"
    if any(token in normalized for token in ("bike", "cycling", "spin")):
        return "bike"
    if any(token in normalized for token in ("weight", "strength", "circuit")):
        return "strength"
    return "other"


def _read_exercise(path: Path, audit: dict[str, Any]) -> dict[str, pd.Series]:
    daily: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    records = 0
    skipped = 0
    for row in _iter_json_array(path):
        records += 1
        day = _record_day(row.get("startTime"))
        if day is None:
            skipped += 1
            continue
        values = daily[day]
        values["exercise_sessions"] += 1
        for source, target, divisor in (
            ("duration", "exercise_duration_min", 60000.0),
            ("activeDuration", "exercise_active_duration_min", 60000.0),
            ("calories", "exercise_calories", 1.0),
            ("steps", "exercise_steps", 1.0),
        ):
            number = _number(row.get(source))
            if math.isfinite(number):
                values[target] += number / divisor
        heart_rate = _number(row.get("averageHeartRate"))
        if math.isfinite(heart_rate):
            values["exercise_average_heart_rate_sum"] += heart_rate
            values["exercise_average_heart_rate_count"] += 1
        category = _activity_category(row.get("activityName"))
        values[f"exercise_{category}_sessions"] += 1

    result: dict[str, dict[str, float]] = defaultdict(dict)
    for day, values in daily.items():
        for name, value in values.items():
            if name == "exercise_average_heart_rate_sum":
                continue
            if name == "exercise_average_heart_rate_count":
                if value:
                    result["exercise_average_heart_rate"][day] = (
                        values["exercise_average_heart_rate_sum"] / value
                    )
                continue
            result[name][day] = value
    audit["files"][str(path)] = {"records": records, "skipped": skipped}
    return {name: _as_series(days) for name, days in result.items()}


def _read_wellness(
    path: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.Series],
    dict[str, Any],
    dict[pd.Timestamp, pd.Timestamp],
]:
    frame = pd.read_csv(path)
    required = {
        "effective_time_frame",
        "fatigue",
        "mood",
        "readiness",
        "sleep_duration_h",
        "sleep_quality",
        "soreness",
        "stress",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} is missing required columns: {sorted(required)}")
    frame["day"] = pd.to_datetime(
        frame["effective_time_frame"], errors="coerce", utc=True
    ).dt.tz_convert(None).dt.normalize()
    wellness_columns = [
        "fatigue",
        "mood",
        "readiness",
        "sleep_duration_h",
        "sleep_quality",
        "soreness",
        "stress",
    ]
    for column in wellness_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid_label_rows = frame["fatigue"].between(1, 5)
    invalid_labels = int((frame["fatigue"].notna() & ~valid_label_rows).sum())
    frame.loc[~valid_label_rows, "fatigue"] = np.nan
    invalid_wellness_values = 0
    for column in ("mood", "sleep_quality", "soreness", "stress"):
        invalid = frame[column].notna() & ~frame[column].between(1, 5)
        invalid_wellness_values += int(invalid.sum())
        frame.loc[invalid, column] = np.nan
    valid_target_reports = frame.loc[
        frame["fatigue"].notna() & frame["day"].notna()
    ].copy()
    valid_target_reports["report_time"] = pd.to_datetime(
        valid_target_reports["effective_time_frame"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    first_report_times = (
        valid_target_reports.groupby("day")["report_time"].min().dropna().to_dict()
    )
    daily = (
        frame.dropna(subset=["day"])
        .groupby("day")[wellness_columns]
        .median()
        .sort_index()
    )
    history = {
        f"wellness_{column}": daily[column]
        for column in wellness_columns
    }
    audit = {
        "reports": int(len(frame)),
        "invalid_fatigue_reports": invalid_labels,
        "invalid_wellness_values": invalid_wellness_values,
        "invalid_timestamps": int(frame["day"].isna().sum()),
        "duplicate_report_days": int((frame.dropna(subset=["day"]).groupby("day").size() > 1).sum()),
        "valid_target_days": int(daily["fatigue"].notna().sum()),
        "label_counts": {
            str(int(label)): int(count)
            for label, count in daily["fatigue"].value_counts().sort_index().items()
        },
    }
    return daily, history, audit, first_report_times


def _read_reporting(path: Path, audit: dict[str, Any]) -> dict[str, pd.Series]:
    frame = pd.read_csv(path)
    required = {"date", "meals", "weight", "glasses_of_fluid", "alcohol_consumed"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} is missing required reporting columns")
    frame["day"] = pd.to_datetime(frame["date"], format="%d/%m/%Y", errors="coerce")
    frame["nutrition_weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    frame["nutrition_fluid_glasses"] = pd.to_numeric(frame["glasses_of_fluid"], errors="coerce")
    frame["nutrition_meal_count"] = frame["meals"].map(
        lambda value: (
            sum(bool(part.strip()) for part in str(value).split(","))
            if isinstance(value, str) and value.strip()
            else np.nan
        )
    )
    alcohol = frame["alcohol_consumed"].astype(str).str.strip().str.lower()
    frame["nutrition_alcohol"] = alcohol.map({"yes": 1.0, "no": 0.0})
    frame["nutrition_report_present"] = 1.0
    columns = (
        "nutrition_weight",
        "nutrition_fluid_glasses",
        "nutrition_meal_count",
        "nutrition_alcohol",
        "nutrition_report_present",
    )
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    daily = frame.dropna(subset=["day"]).groupby("day")[list(columns)].mean().sort_index()
    audit["files"][str(path)] = {
        "records": int(len(frame)),
        "skipped": int(frame["day"].isna().sum()),
        "missing_weight": int(frame["nutrition_weight"].isna().sum()),
    }
    return {name: daily[name] for name in daily.columns}


def _read_participant_series(
    participant_dir: Path,
    audit: dict[str, Any],
    include_reporting: bool,
) -> tuple[
    dict[str, pd.Series],
    dict[str, pd.Series],
    dict[str, dict[pd.Timestamp, pd.Timestamp]],
]:
    fitbit_dir = participant_dir / "fitbit"
    series: dict[str, pd.Series] = {}
    morning_sleep: dict[str, pd.Series] = {}
    morning_times: dict[str, dict[pd.Timestamp, pd.Timestamp]] = {}
    score_id_timestamps: dict[str, pd.Timestamp] = {}

    for feature, (filename, scale, _) in MINUTE_FILES.items():
        path = fitbit_dir / filename
        if path.exists():
            series.update(_read_minute_file(path, feature, scale, audit))

    heart_path = fitbit_dir / "heart_rate.json"
    if heart_path.exists():
        series.update(_read_heart_rate(heart_path, audit))

    for feature, filename in DAILY_FILES.items():
        path = fitbit_dir / filename
        if path.exists():
            series[feature] = _read_daily_json(path, feature, audit)

    zones_path = fitbit_dir / "time_in_heart_rate_zones.json"
    if zones_path.exists():
        series.update(_read_heart_rate_zones(zones_path, audit))

    score_path = fitbit_dir / "sleep_score.csv"
    if score_path.exists():
        score, timestamps, score_id_timestamps = _read_sleep_score(score_path, audit)
        series.update(score)
        morning_sleep.update(score)
        morning_times.update({name: timestamps for name in score})

    sleep_path = fitbit_dir / "sleep.json"
    if sleep_path.exists():
        sleep_sessions, timestamps, sleep_log_ids = _read_sleep_sessions(sleep_path, audit)
        for day, log_id in sleep_log_ids.items():
            score_time = score_id_timestamps.get(log_id)
            if score_time is not None:
                timestamps[day] = score_time
        series.update(sleep_sessions)
        morning_sleep.update(sleep_sessions)
        morning_times.update({name: timestamps for name in sleep_sessions})

    srpe_path = participant_dir / "pmsys" / "srpe.csv"
    if srpe_path.exists():
        series.update(_read_srpe(srpe_path, audit))

    exercise_path = fitbit_dir / "exercise.json"
    if exercise_path.exists():
        series.update(_read_exercise(exercise_path, audit))

    if include_reporting:
        reporting_path = participant_dir / "googledocs" / "reporting.csv"
        if reporting_path.exists():
            series.update(_read_reporting(reporting_path, audit))

    return series, morning_sleep, morning_times


def _history_features(
    day: pd.Timestamp, series_by_name: dict[str, pd.Series]
) -> dict[str, float]:
    previous_day = day - pd.Timedelta(days=1)
    features: dict[str, float] = {}
    for name, series in series_by_name.items():
        if previous_day in series.index:
            features[f"history_{name}_lag1"] = float(series.loc[previous_day])
        else:
            features[f"history_{name}_lag1"] = np.nan
        if name == "wellness_fatigue":
            recent = series.loc[
                (series.index >= day - pd.Timedelta(days=7))
                & (series.index < day)
            ].dropna()
            features["history_wellness_fatigue_last7"] = (
                float(recent.iloc[-1]) if not recent.empty else np.nan
            )

        for window in (7, 28):
            start = day - pd.Timedelta(days=window)
            values = series.loc[(series.index >= start) & (series.index < day)].dropna()
            features[f"history_{name}_mean{window}"] = (
                float(values.mean()) if not values.empty else np.nan
            )
            if name in SUM_FEATURES:
                features[f"history_{name}_sum{window}"] = (
                    float(values.sum()) if not values.empty else np.nan
                )

    srpe_load = series_by_name.get("srpe_load")
    if srpe_load is not None:
        acute_start = day - pd.Timedelta(days=3)
        acute_values = srpe_load.loc[
            (srpe_load.index >= acute_start) & (srpe_load.index < day)
        ].dropna()
        chronic_start = day - pd.Timedelta(days=28)
        chronic_values = srpe_load.loc[
            (srpe_load.index >= chronic_start) & (srpe_load.index < day)
        ].dropna()
        acute = float(acute_values.sum()) if not acute_values.empty else np.nan
        chronic_week = float(chronic_values.sum()) / 4 if not chronic_values.empty else np.nan
        features["history_srpe_acute_chronic_ratio"] = (
            acute / chronic_week if math.isfinite(acute) and chronic_week > 0 else np.nan
        )
    return features


def _morning_features(
    day: pd.Timestamp,
    series_by_name: dict[str, pd.Series],
    timestamp_by_name: dict[str, dict[pd.Timestamp, pd.Timestamp]],
    report_time: pd.Timestamp | None,
) -> dict[str, float]:
    features: dict[str, float] = {}
    for name, series in series_by_name.items():
        measurement_time = timestamp_by_name[name].get(day)
        if (
            day not in series.index
            or report_time is None
            or measurement_time is None
            or measurement_time > report_time
        ):
            features[f"morning_{name}"] = np.nan
            continue
        value = float(series.loc[day])
        features[f"morning_{name}"] = value
        start = day - pd.Timedelta(days=28)
        baseline = series.loc[(series.index >= start) & (series.index < day)].dropna()
        features[f"morning_{name}_delta28"] = (
            value - float(baseline.mean()) if not baseline.empty else np.nan
        )
    return features


def build_daily_dataset(
    data_root: Path, include_reporting: bool = True
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one row per reported participant-day with strictly historical inputs."""
    data_root = data_root.resolve()
    rows: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}

    for participant_dir in sorted(
        path for path in data_root.glob("p*") if path.is_dir()
    ):
        wellness_path = participant_dir / "pmsys" / "wellness.csv"
        if not wellness_path.exists():
            continue

        (
            daily_wellness,
            wellness_history,
            wellness_audit,
            first_report_times,
        ) = _read_wellness(wellness_path)
        participant_audit: dict[str, Any] = {"files": {}}
        series, morning_sleep, morning_times = _read_participant_series(
            participant_dir, participant_audit, include_reporting
        )
        series.update(wellness_history)
        audits[participant_dir.name] = {
            "wellness": wellness_audit,
            "files": participant_audit["files"],
        }

        for day in daily_wellness.index:
            features = _history_features(day, series)
            features.update(
                _morning_features(
                    day,
                    morning_sleep,
                    morning_times,
                    first_report_times.get(day),
                )
            )
            rows.append(
                {
                    "participant": participant_dir.name,
                    "date": day,
                    "fatigue": (
                        float(daily_wellness.at[day, "fatigue"])
                        if pd.notna(daily_wellness.at[day, "fatigue"])
                        else np.nan
                    ),
                    **features,
                }
            )

    if not rows:
        raise ValueError(f"No wellness.csv data found under {data_root}")

    dataset = pd.DataFrame(rows).sort_values(["participant", "date"]).reset_index(drop=True)
    feature_columns = [column for column in dataset.columns if column.startswith(("history_", "morning_"))]
    target_rows = dataset["fatigue"].notna()
    label_counts = {
        str(int(label)): int(count)
        for label, count in dataset.loc[target_rows, "fatigue"].value_counts().sort_index().items()
    }
    audit = {
        "data_root": str(data_root),
        "participants": int(dataset["participant"].nunique()),
        "participant_days": int(len(dataset)),
        "valid_target_days": int(target_rows.sum()),
        "invalid_fatigue_reports": int(
            sum(item["wellness"]["invalid_fatigue_reports"] for item in audits.values())
        ),
        "invalid_wellness_values": int(
            sum(item["wellness"]["invalid_wellness_values"] for item in audits.values())
        ),
        "duplicate_report_days": int(
            sum(item["wellness"]["duplicate_report_days"] for item in audits.values())
        ),
        "label_counts": label_counts,
        "date_start": str(dataset["date"].min().date()),
        "date_end": str(dataset["date"].max().date()),
        "feature_count": len(feature_columns),
        "feature_missing_fraction": {
            column: float(dataset[column].isna().mean())
            for column in feature_columns
        },
        "participants_detail": audits,
    }
    return dataset, audit


def load_participant_profiles(workbook_path: Path) -> dict[str, dict[str, Any]]:
    """Read participant attributes for simulator setup, not fatigue model inputs."""
    frame = pd.read_excel(workbook_path, header=1)
    id_column = "Participant ID"
    if id_column not in frame.columns:
        raise ValueError(f"{workbook_path} is missing {id_column!r}")

    profiles: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        participant = str(row.get(id_column, "")).strip()
        if not participant.startswith("p"):
            continue

        flags: list[str] = []
        raw_date = row.get("Date")
        run_date: str | None = None
        if isinstance(raw_date, (pd.Timestamp, datetime)):
            run_date = raw_date.date().isoformat()
        elif isinstance(raw_date, date):
            run_date = raw_date.isoformat()
        elif isinstance(raw_date, (int, float, np.integer, np.floating)) and pd.notna(
            raw_date
        ):
            run_date = (
                pd.Timestamp("1899-12-30") + pd.to_timedelta(float(raw_date), unit="D")
            ).date().isoformat()
        if run_date is None:
            flags.append("invalid_first_5km_date")
        elif not ("2019-11-01" <= run_date <= "2020-03-31"):
            flags.append("first_5km_outside_study_period")

        def numeric(column: str) -> float | None:
            value = pd.to_numeric(row.get(column), errors="coerce")
            return float(value) if pd.notna(value) else None

        run_minutes = numeric("Minutes")
        run_seconds = numeric("Seconds")
        if (
            run_minutes is None
            or run_seconds is None
            or run_minutes < 0
            or not 0 <= run_seconds < 60
        ):
            flags.append("invalid_first_5km_duration")

        walk_stride = numeric("Stride walk")
        reported_run_stride = numeric("Stride run")
        run_stride = reported_run_stride
        if walk_stride is not None and not 50 <= walk_stride <= 150:
            flags.append("suspicious_walk_stride")
            walk_stride = None
        if reported_run_stride is not None and not 50 <= reported_run_stride <= 180:
            flags.append("suspicious_run_stride")
            run_stride = None

        gender = row.get("Gender")
        max_heart_rate = numeric("Max heart rate")
        if max_heart_rate is None:
            flags.append("missing_max_heart_rate")
        profiles[participant] = {
            "age": numeric("Age"),
            "height_cm": numeric("Height"),
            "gender": str(gender).replace("\xa0", " ").strip().lower()
            if pd.notna(gender)
            else None,
            "max_heart_rate": max_heart_rate,
            "first_5km_date": run_date,
            "first_5km_duration_seconds": (
                run_minutes * 60 + run_seconds
                if run_minutes is not None
                and run_seconds is not None
                and run_minutes >= 0
                and 0 <= run_seconds < 60
                else None
            ),
            "stride_walk_cm": walk_stride,
            "stride_run_cm": run_stride,
            "stride_run_reported_cm": reported_run_stride,
            "data_quality_flags": flags,
        }
    return profiles


def injury_state_at(
    participant_dir: Path, as_of: pd.Timestamp
) -> dict[str, str] | None:
    """Return the latest reported injury snapshot at or before a decision time."""
    path = participant_dir / "pmsys" / "injury.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    required = {"effective_time_frame", "injuries"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} is missing required injury columns")
    frame["timestamp"] = pd.to_datetime(
        frame["effective_time_frame"], errors="coerce", utc=True
    )
    frame = frame.dropna(subset=["timestamp"])
    cutoff = pd.Timestamp(as_of)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    eligible = frame.loc[frame["timestamp"] <= cutoff]
    if eligible.empty:
        return None
    raw = eligible.sort_values("timestamp").iloc[-1]["injuries"]
    try:
        injuries = ast.literal_eval(str(raw))
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"Cannot parse injury snapshot in {path}") from error
    if not isinstance(injuries, dict):
        raise ValueError(f"Injury snapshot in {path} is not a mapping")
    return {
        str(location): str(severity).strip().lower()
        for location, severity in injuries.items()
    }
