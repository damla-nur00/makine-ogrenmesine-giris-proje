"""Experimental HIGH-FATIGUE classifier: 9 CORE + 4 SLEEP features (13 total).

Research-only comparison to the previously added NEW-13 model.
Does not replace any existing classifier, make a deployment decision, or
evaluate sealed test participants. The feature sets were investigated on
the same development data, so LOPO results are exploratory, not independent.
Previous-day questionnaire submission times are not independently verified.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "01_veri_hazirlama"))
sys.path.insert(0, str(REPO_ROOT / "02_yorgunluk_tahmini"))

from fatigue_models import add_personal_features  # noqa: E402
from split_participants import validate_split  # noqa: E402

CORE_9 = [
    "history_wellness_fatigue_lag1",
    "history_wellness_fatigue_mean7",
    "history_wellness_fatigue_mean28",
    "p_mean",
    "p_std",
    "fat_trend",
    "fat_vs_pers",
    "history_wellness_readiness_lag1",
    "history_wellness_soreness_lag1",
]
SLEEP_4 = [
    "dv_sleep_score_overall_score",
    "dw_sleep_score_overall_score",
    "history_sleep_efficiency_mean7",
    "history_sleep_minutesAsleep_mean7",
]
CORE_9_SLEEP_4 = CORE_9 + SLEEP_4
PREVIOUS_NEW_13 = sorted([
    "p_n", "p_mean", "p_std", "p_high_rate", "p_low_rate",
    "fat_trend", "fat_vs_pers",
    "history_wellness_fatigue_lag1",
    "history_wellness_fatigue_last7",
    "history_wellness_fatigue_mean7",
    "history_wellness_fatigue_mean28",
    "history_wellness_readiness_lag1",
    "history_wellness_soreness_lag1",
])
assert len(set(CORE_9_SLEEP_4)) == len(set(PREVIOUS_NEW_13)) == 13


def build_model():
    """Training-fold imputation/scaling; no cross-participant preprocessing fit."""
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(
            C=0.01, class_weight="balanced", max_iter=3000,
            solver="lbfgs", random_state=42,
        ),
    )


def scores_for(frame):
    y = frame["high"].to_numpy(dtype=int)
    score = frame["score"].to_numpy(dtype=float)
    top_n = math.ceil(len(frame) * 0.10)
    ranked = frame.sort_values(
        ["score", "participant", "date"],
        ascending=[False, True, True], kind="mergesort",
    )
    tp = int(ranked.head(top_n)["high"].sum())
    within = []
    for _, person in frame.groupby("participant"):
        yy = person["high"].to_numpy(dtype=int)
        if yy.sum() >= 3 and (len(yy) - yy.sum()) >= 3:
            within.append(float(roc_auc_score(yy, person["score"])))
    return {
        "days": len(frame),
        "high_fatigue_days": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "top10_count": top_n,
        "top10_true_positive": tp,
        "recall_at_10pct": tp / int(y.sum()),
        "precision_at_10pct": tp / top_n,
        "within_person_roc_auc": float(np.mean(within)) if within else None,
        "eligible_participants": len(within),
    }


def main(features, split_file, protocol_file, output):
    # Require the original frozen participant split. Never regenerate it.
    split_bytes = split_file.read_bytes()
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    expected_hash = protocol["provenance"]["previous_grouped_split_sha256"]
    if hashlib.sha256(split_bytes).hexdigest() != expected_hash:
        raise ValueError("Frozen split checksum mismatch; refusing to run")
    split = json.loads(split_bytes)
    validate_split(split)
    dev = set(split["development"])
    if dev.intersection(split["test"]):
        raise ValueError("Sealed test participant included in development")

    raw = pd.read_csv(features, parse_dates=["date"])
    df = raw.loc[raw["participant"].isin(dev) & raw["fatigue"].notna()].copy()
    del raw
    if set(df["participant"]) != dev or df.duplicated(["participant", "date"]).any():
        raise ValueError("Development participants incomplete or duplicate days")
    df = add_personal_features(df).sort_values(
        ["participant", "date"]
    ).reset_index(drop=True)
    columns = {
        "new_13_previous": PREVIOUS_NEW_13,
        "core_9_plus_sleep_4": CORE_9_SLEEP_4,
    }
    missing = sorted(set(sum(columns.values(), [])) - set(df.columns))
    if missing:
        raise ValueError(f"Missing features: {missing}")

    target = (df["fatigue"].to_numpy(dtype=float) >= 4).astype(int)
    groups = df["participant"].to_numpy()
    rows = []
    for train, validation in LeaveOneGroupOut().split(df, target, groups):
        for label, selected_columns in columns.items():
            train_x = df.iloc[train][selected_columns].replace(
                [np.inf, -np.inf], np.nan
            )
            val_x = df.iloc[validation][selected_columns].replace(
                [np.inf, -np.inf], np.nan
            )
            model = build_model()
            model.fit(train_x, target[train])
            predicted = model.predict_proba(val_x)[:, 1]
            for row_idx, risk in zip(validation, predicted):
                rows.append({
                    "model": label,
                    "participant": str(groups[row_idx]),
                    "date": df.at[row_idx, "date"].strftime("%Y-%m-%d"),
                    "high": int(target[row_idx]),
                    "score": float(risk),
                })
    oof = pd.DataFrame(rows)
    results = [
        {"model": name, **scores_for(oof.loc[oof["model"] == name])}
        for name in columns
    ]
    output.mkdir(parents=True, exist_ok=True)
    oof.to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(results).to_csv(output / "metrics.csv", index=False)
    (output / "summary.json").write_text(json.dumps({
        "experiment": "core9_plus_sleep4_exploratory",
        "selected_for_production": False,
        "sealed_test_evaluated": False,
        "frozen_split_sha256": expected_hash,
        "feature_sets": columns,
        "results": results,
        "warnings": [
            "Repeated model/feature search on the same development cohort.",
            "Previous-day survey submission timestamps require verification.",
            "Scores are uncalibrated; not clinical risk probabilities.",
            "Top 10 percent is a retrospective pooled ranking, not a deployed alert threshold.",
            "Does not alter any existing model.",
        ],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path,
                        default=Path("outputs/fatigue_daily.csv"))
    parser.add_argument("--split", type=Path,
                        default=Path("outputs/split.json"))
    parser.add_argument("--protocol", type=Path,
                        default=Path("experiment_protocol.json"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("outputs/experiment_high_fatigue_core9_sleep4"))
    args = parser.parse_args()
    main(args.features, args.split, args.protocol, args.out_dir)
