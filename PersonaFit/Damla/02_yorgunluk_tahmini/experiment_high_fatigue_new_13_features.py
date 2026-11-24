"""Research-only: NEW 13-FEATURE high-fatigue classifier comparison.

Candidate changes versus the original 13-feature Logistic Regression A:
  REMOVE: p_median, p_mean14
  ADD: history_wellness_readiness_lag1, history_wellness_soreness_lag1

This script DOES NOT select or deploy a replacement model, modify existing
model code, or evaluate sealed test participants. It is exploratory: the
candidate features were chosen after looking at the development data.
Survey submission timestamps need separate verification before deployment.
Scores from predict_proba have not been calibrated as risk probabilities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "01_veri_hazirlama"))
sys.path.insert(0, str(REPO / "02_yorgunluk_tahmini"))

from fatigue_models import (  # noqa: E402
    FATIGUE_HISTORY_COLUMNS, PERSONAL_COLUMNS, add_personal_features,
)
from split_participants import validate_split  # noqa: E402

OLD_13 = sorted(PERSONAL_COLUMNS + FATIGUE_HISTORY_COLUMNS + [
    "fat_trend", "fat_vs_pers",
])
NEW_13 = sorted((set(OLD_13) - {"p_median", "p_mean14"}) | {
    "history_wellness_readiness_lag1",
    "history_wellness_soreness_lag1",
})
assert len(OLD_13) == len(NEW_13) == 13


def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(
            C=0.01, class_weight="balanced", max_iter=3000,
            solver="lbfgs", random_state=42,
        ),
    )


def evaluate(frame: pd.DataFrame) -> dict:
    """Pooled LOPO metrics. Top 10% is a ranking budget, not a deployed threshold."""
    actual = frame["high"].to_numpy(dtype=int)
    scores = frame["score"].to_numpy(dtype=float)
    ordered = frame.sort_values(
        ["score", "participant", "date"],
        ascending=[False, True, True], kind="mergesort",
    )
    k = math.ceil(len(frame) * 0.10)
    caught = int(ordered.head(k)["high"].sum())
    within = []
    for _, part in frame.groupby("participant"):
        y = part["high"].to_numpy(dtype=int)
        if y.sum() >= 3 and (len(y) - y.sum()) >= 3:
            within.append(float(roc_auc_score(y, part["score"])))
    return {
        "n": len(frame), "high_days": int(actual.sum()),
        "roc_auc": float(roc_auc_score(actual, scores)),
        "average_precision": float(average_precision_score(actual, scores)),
        "top_10pct_count": k,
        "top_10pct_caught": caught,
        "recall_at_10pct": caught / int(actual.sum()),
        "precision_at_10pct": caught / k,
        "within_person_auc": float(np.mean(within)) if within else None,
        "within_person_eligible": len(within),
    }


def main(features: Path, split_file: Path, protocol_file: Path, output: Path):
    # The frozen split is mandatory. Do not recreate it if unavailable.
    split_bytes = split_file.read_bytes()
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    expected = protocol["provenance"]["previous_grouped_split_sha256"]
    if hashlib.sha256(split_bytes).hexdigest() != expected:
        raise ValueError("Frozen split checksum mismatch; refusing evaluation.")
    split = json.loads(split_bytes)
    validate_split(split)
    dev = set(split["development"])
    sealed = set(split["test"])
    if dev & sealed:
        raise ValueError("Sealed test overlaps development data.")

    # Read only the named development participants into the experiment.
    raw = pd.read_csv(features, parse_dates=["date"])
    data = raw.loc[raw["participant"].isin(dev) & raw["fatigue"].notna()].copy()
    del raw
    if set(data["participant"]) != dev:
        raise ValueError("Expected all 14 development participants.")
    if data.duplicated(["participant", "date"]).any():
        raise ValueError("Duplicate participant/date data.")
    data = add_personal_features(data).sort_values(
        ["participant", "date"]
    ).reset_index(drop=True)
    missing = sorted(set(OLD_13 + NEW_13) - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    y = (data["fatigue"].to_numpy(dtype=float) >= 4).astype(int)
    groups = data["participant"].to_numpy()
    predictions = []
    # Fit an independent preprocessing pipeline within each held-out person.
    for tr, va in LeaveOneGroupOut().split(data, y, groups):
        for label, columns in (("original_13", OLD_13), ("new_13", NEW_13)):
            xtrain = data.iloc[tr][columns].replace(
                [np.inf, -np.inf], np.nan
            )
            xvalid = data.iloc[va][columns].replace(
                [np.inf, -np.inf], np.nan
            )
            model = make_model()
            model.fit(xtrain, y[tr])
            scores = model.predict_proba(xvalid)[:, 1]
            for idx, score in zip(va, scores):
                predictions.append({
                    "model": label,
                    "participant": str(groups[idx]),
                    "date": data.at[idx, "date"].strftime("%Y-%m-%d"),
                    "high": int(y[idx]),
                    "score": float(score),
                })

    oof = pd.DataFrame(predictions)
    summary = [
        {"model": label, **evaluate(oof.loc[oof["model"] == label])}
        for label in ("original_13", "new_13")
    ]
    output.mkdir(parents=True, exist_ok=True)
    oof.to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "metrics.csv", index=False)
    report = {
        "experiment": "new_13_features_exploratory",
        "selected_for_production": False,
        "test_evaluated": False,
        "frozen_split_sha256": expected,
        "original_features": OLD_13,
        "new_features": NEW_13,
        "results": summary,
        "warnings": [
            "Feature selection used this development cohort; not independent confirmation.",
            "Prior-day survey submission timestamps are not verified.",
            "Risk scores are not calibrated probabilities.",
            "Original classifier and regression models are unchanged.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path,
                        default=Path("outputs/fatigue_daily.csv"))
    parser.add_argument("--split", type=Path,
                        default=Path("outputs/split.json"))
    parser.add_argument("--protocol", type=Path,
                        default=Path("experiment_protocol.json"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("outputs/experiment_high_fatigue_new_13_features"))
    args = parser.parse_args()
    main(args.features, args.split, args.protocol, args.out_dir)
