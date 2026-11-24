"""E1 OOF tahminleriyle günlük yorgunluk DEĞİŞİMİ hata analizi.

Yeniden eğitim yoktur. Sadece 14 geliştirme katılımcısının E1
validation_predictions.csv verisini okur. Mühürlü test verisini okumaz.
Veriler GitHub'a gönderilmez; çıktılar outputs/ altında kalır.

Calistirma (repo kokunden, grup klasorleri PYTHONPATH'a eklenmisken):
python 06_deneyler/experiment_fatigue_change_errors.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FROZEN_SPLIT_SHA256 = "8f053e7dc1ee6a4fd7d34a7a590079a11fe53cc9036328b27a63f2fc1a76437e"
STRATEGIES = (
    "baseline_train_median",
    "baseline_personal_median",
    "gbm_abs_personal",
)
KEY = ["participant", "date"]
REQUIRED = set(KEY + ["strategy", "fatigue", "prediction"])
GROUPS = (
    "all",
    "has_previous_report",
    "consecutive_calendar_day",
    "rise_ge_1_observation",
    "rise_ge_2_observation",
    "rise_ge_1_next_day",
    "rise_ge_2_next_day",
    "cross_to_high_ge_4",
    "already_high_ge_4",
    "fall_ge_1_observation",
    "stable_from_previous",
)


def load_development_split(path: Path) -> tuple[set[str], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != FROZEN_SPLIT_SHA256:
        raise ValueError("Frozen participant split SHA256 mismatch; refusing analysis")
    split = json.loads(raw)
    development = set(map(str, split["development"]))
    test = set(map(str, split["test"]))
    if len(development) != 14 or len(test) != 2 or development & test:
        raise ValueError("Expected 14 development and 2 non-overlapping test people")
    return development, digest


def prepare_diagnostics(predictions: pd.DataFrame, development: set[str]) -> pd.DataFrame:
    missing = REQUIRED - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing required E1 columns: {sorted(missing)}")
    df = predictions.copy()
    df["participant"] = df["participant"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if df[KEY + ["strategy"]].isna().any().any():
        raise ValueError("Missing person/date/strategy")
    if set(df["participant"]) != development:
        raise ValueError("Unexpected participant: E1 data must contain development users ONLY")
    if set(df["strategy"]) != set(STRATEGIES):
        raise ValueError("Unexpected E1 strategies")
    if df.duplicated(KEY + ["strategy"]).any():
        raise ValueError("Duplicate predictions for a participant-date-strategy")
    df["fatigue"] = pd.to_numeric(df["fatigue"], errors="raise")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="raise")
    if not np.isfinite(df[["fatigue", "prediction"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite labels or predictions")
    if not df["fatigue"].between(1, 5).all():
        raise ValueError("Unexpected fatigue label outside 1..5")
    if not df["prediction"].between(1, 5).all():
        raise ValueError("Unexpected prediction outside 1..5")

    coverage = df.groupby(KEY, observed=True)["strategy"].nunique()
    if not coverage.eq(len(STRATEGIES)).all():
        raise ValueError("Not all E1 strategies predict every participant-day")
    if not df.groupby(KEY, observed=True)["fatigue"].nunique().eq(1).all():
        raise ValueError("Different fatigue labels for the same participant-day")

    # A person's previous observation is their prior LABEL, never their current target.
    # The change itself is used ONLY for retrospective scoring, not for training.
    actual = (df.loc[df["strategy"].eq(STRATEGIES[0]), KEY + ["fatigue"]]
              .sort_values(KEY).reset_index(drop=True))
    actual["prev_fatigue"] = actual.groupby("participant")["fatigue"].shift(1)
    actual["prev_date"] = actual.groupby("participant")["date"].shift(1)
    actual["days_since_prev"] = (actual["date"] - actual["prev_date"]).dt.days
    actual["change"] = actual["fatigue"] - actual["prev_fatigue"]
    actual["has_previous_report"] = actual["prev_fatigue"].notna()
    actual["consecutive_calendar_day"] = actual["days_since_prev"].eq(1)
    actual["rise_ge_1_observation"] = actual["change"].ge(1)
    actual["rise_ge_2_observation"] = actual["change"].ge(2)
    actual["rise_ge_1_next_day"] = actual["rise_ge_1_observation"] & actual["consecutive_calendar_day"]
    actual["rise_ge_2_next_day"] = actual["rise_ge_2_observation"] & actual["consecutive_calendar_day"]
    actual["cross_to_high_ge_4"] = actual["prev_fatigue"].lt(4) & actual["fatigue"].ge(4)
    actual["already_high_ge_4"] = actual["fatigue"].ge(4)
    actual["fall_ge_1_observation"] = actual["change"].le(-1)
    actual["stable_from_previous"] = actual["change"].eq(0)
    actual["all"] = True

    diag = df.merge(actual.drop(columns=["fatigue"]), on=KEY,
                    how="inner", validate="many_to_one")
    if len(diag) != len(df):
        raise AssertionError("Rows lost when joining history")
    diag["abs_error"] = (diag["fatigue"] - diag["prediction"]).abs()
    diag["underprediction"] = diag["fatigue"] - diag["prediction"]
    return diag


def score(group: pd.DataFrame) -> dict:
    if group.empty:
        return {"days": 0, "people": 0, "mae": None,
                "underpredicted_by_ge_1_fraction": None, "mean_underprediction": None}
    return {
        "days": int(len(group)),
        "people": int(group["participant"].nunique()),
        "mae": float(group["abs_error"].mean()),
        "underpredicted_by_ge_1_fraction": float(group["underprediction"].ge(1).mean()),
        "mean_underprediction": float(group["underprediction"].mean()),
    }


def summarize(diag: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, persons = [], []
    for name in GROUPS:
        for strategy in STRATEGIES:
            chunk = diag.loc[diag["strategy"].eq(strategy) & diag[name]]
            records.append({"group": name, "strategy": strategy, **score(chunk)})
            for participant, part in chunk.groupby("participant", sort=True):
                persons.append({"group": name, "participant": participant,
                                "strategy": strategy, **score(part)})
    summary = pd.DataFrame(records)
    reference = (summary.loc[summary["strategy"].eq("baseline_personal_median"),
                             ["group", "mae"]].rename(columns={"mae": "median_mae"}))
    summary = summary.merge(reference, on="group", validate="many_to_one")
    summary["delta_mae_vs_personal_median"] = summary["mae"] - summary["median_mae"]
    return summary.drop(columns=["median_mae"]), pd.DataFrame(persons)


def run(input_path: Path, split_path: Path, output_dir: Path) -> dict:
    development, digest = load_development_split(split_path)
    data = pd.read_csv(input_path)
    diag = prepare_diagnostics(data, development)
    by_group, by_person = summarize(diag)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_group.to_csv(output_dir / "error_by_change.csv", index=False)
    by_person.to_csv(output_dir / "error_by_person.csv", index=False)
    diag.to_csv(output_dir / "daily_diagnostics.csv", index=False, float_format="%.8f")
    # Worst days are selected for description ONLY, never model/threshold tuning.
    worst = (diag.loc[diag["strategy"].eq("gbm_abs_personal")]
             .sort_values("abs_error", ascending=False).head(30))
    worst.to_csv(output_dir / "worst_gbm_days.csv", index=False)
    report = {
        "status": "development_only_exploratory_diagnostic",
        "source": str(input_path),
        "split_sha256": digest,
        "people": len(development),
        "days": int(diag.loc[diag["strategy"].eq(STRATEGIES[0])].shape[0]),
        "strategies": list(STRATEGIES),
        "note": ("Change compares successive OBSERVED reports; next_day groups "
                 "require exactly one calendar day. The target-day label is used "
                 "only retrospectively, never as a prediction feature."),
        "test_evaluated": False,
        "metric_file": "error_by_change.csv",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n" + by_group.loc[by_group["group"].isin([
        "all", "rise_ge_1_next_day", "rise_ge_2_next_day",
        "cross_to_high_ge_4", "already_high_ge_4"
    ])].to_string(index=False))
    print(f"\nAnalysis saved locally: {output_dir}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path,
                   default=Path("outputs/experiment_cold_start/validation_predictions.csv"))
    p.add_argument("--split", type=Path, default=Path("outputs/split.json"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/experiment_fatigue_change_errors"))
    args = p.parse_args()
    run(args.predictions, args.split, args.out_dir)


if __name__ == "__main__":
    main()
