"""Compare fatigue predictors; apply E2 adoption gate to new group-A candidates.

Legacy candidates remain reference-only. Outputs default to an isolated folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from split_participants import validate_split
from fatigue_models import (
    _candidate_models,
    _continuous_metrics,
    add_personal_features,
    baseline_predictions,
    model_feature_columns,
)


def predictions_for_baselines(train: pd.DataFrame, valid: pd.DataFrame) -> dict[str, np.ndarray]:
    return baseline_predictions(valid, float(train["fatigue"].median()))


def train(features_path: Path, split_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("personafit.grouped_training")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(output_dir / "train.log", mode="w", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.info("Training started; sealed test labels will not be evaluated")
    data = pd.read_csv(features_path, parse_dates=["date"])
    split = json.loads(split_path.read_text(encoding="utf-8"))
    validate_split(split)
    data = data.loc[data["fatigue"].notna()].copy()
    if set(data["participant"].unique()) != set(split["development"]) | set(split["test"]):
        raise ValueError("Participant identities do not match sealed split")
    data = add_personal_features(data.loc[data["participant"].isin(split["development"])].copy())
    columns = model_feature_columns(data)
    candidates = _candidate_models(columns)
    names = list(candidates) + ["baseline_train_median", "baseline_previous_fatigue", "baseline_personal_median"]
    out_of_fold = {name: [] for name in names}
    truths = []
    validation_people = []
    log_rows = []
    for fold in split["folds"]:
        train_frame = data[data["participant"].isin(fold["train"])]
        valid_frame = data[data["participant"].isin(fold["validation"])]
        if train_frame.empty or valid_frame.empty:
            raise ValueError(f"Empty train/validation data in fold {fold['fold']}")
        x_train = train_frame[columns].replace([np.inf, -np.inf], np.nan)
        x_valid = valid_frame[columns].replace([np.inf, -np.inf], np.nan)
        y_train = train_frame["fatigue"]
        y_valid = valid_frame["fatigue"]
        logger.info("Fold %d/7 started | train=%d people/%d days | validation=%d people/%d days", fold["fold"], len(fold["train"]), len(train_frame), len(fold["validation"]), len(valid_frame))
        baseline_predictions = predictions_for_baselines(train_frame, valid_frame)
        fold_predictions = dict(baseline_predictions)
        for name, template in candidates.items():
            started = time.perf_counter()
            logger.info("Fold %d/7 | fitting %-28s ...", fold["fold"], name)
            estimator = clone(template).fit(x_train, y_train)
            fold_predictions[name] = estimator.predict(x_valid)
            logger.info("Fold %d/7 | %-28s fitted in %.1fs", fold["fold"], name, time.perf_counter() - started)
        truths.extend(y_valid.to_list())
        validation_people.extend(valid_frame["participant"].to_list())
        for name, predicted in fold_predictions.items():
            out_of_fold[name].extend(predicted)
            metrics = _continuous_metrics(y_valid, np.asarray(predicted))
            log_rows.append({
                "fold": fold["fold"], "strategy": name,
                "train_people": len(fold["train"]), "validation_people": len(fold["validation"]),
                "train_rows": len(train_frame), "validation_rows": len(valid_frame),
                **metrics,
            })
            logger.info(
                "Fold %d/7 | %-28s | MAE=%.4f RMSE=%.4f R2=%.4f accuracy=%.1f%% balanced=%.1f%% macroF1=%.3f high_recall=%s (n=%d)",
                fold["fold"], name, metrics["mae"], metrics["rmse"], metrics["r2"],
                metrics["rounded_accuracy"] * 100, metrics["rounded_balanced_accuracy"] * 100,
                metrics["rounded_macro_f1"],
                f"{metrics['high_fatigue_recall']:.1%}" if metrics["high_fatigue_recall"] is not None else "N/A",
                metrics["high_fatigue_support"],
            )

    targets = pd.Series(truths, dtype=float)
    scores = {name: _continuous_metrics(targets, np.asarray(values)) for name, values in out_of_fold.items()}
    # The E2 adoption criterion was preregistered for feature-group candidates,
    # not for legacy models. Keep legacy candidates as comparison-only.
    baseline_name = "baseline_personal_median"
    challengers = ["gbm_abs_personal_A", "ridge_personal_A"]
    participant_ids = np.asarray(validation_people, dtype=str)
    people = sorted(split["development"])
    if set(participant_ids) != set(people):
        raise ValueError("OOF validation participants do not match development group.")
    base_errors = np.abs(np.clip(np.asarray(out_of_fold[baseline_name]), 1, 5) - targets.to_numpy())
    base_per_person = {
        p: float(base_errors[participant_ids == p].mean()) for p in people
    }
    eligibility = {}
    person_rows = []
    for name in names:
        errors = np.abs(np.clip(np.asarray(out_of_fold[name]), 1, 5) - targets.to_numpy())
        improved_count = 0
        for p in people:
            mask = participant_ids == p
            person_mae = float(errors[mask].mean())
            improves = person_mae < base_per_person[p]
            improved_count += int(improves)
            person_rows.append({
                "participant": p, "strategy": name,
                "validation_rows": int(mask.sum()),
                "mae": person_mae,
                "personal_median_mae": base_per_person[p],
                "improved_vs_personal_median": bool(improves),
            })
        if name in challengers:
            qualifies = scores[name]["mae"] < scores[baseline_name]["mae"] and improved_count >= 9
            eligibility[name] = {
                "pooled_mae": scores[name]["mae"],
                "personal_median_mae": scores[baseline_name]["mae"],
                "participants_improved": improved_count,
                "participants_total": 14,
                "pooled_mae_lower": bool(scores[name]["mae"] < scores[baseline_name]["mae"]),
                "meets_adoption_rule": bool(qualifies),
            }
    qualifying = [name for name in challengers if eligibility[name]["meets_adoption_rule"]]
    winner = min(qualifying, key=lambda n: (scores[n]["mae"], n)) if qualifying else baseline_name
    best_ml = min(challengers, key=lambda n: (scores[n]["mae"], n))
    unrestricted_lowest_mae = min(scores, key=lambda n: scores[n]["mae"])
    logger.info("E2 selection gate | baseline=%s MAE=%.4f | qualifying=%s | selected=%s",
                baseline_name, scores[baseline_name]["mae"], qualifying or "none", winner)
    for name, item in eligibility.items():
        logger.info("Adoption | %-25s | MAE=%.4f | improved=%d/14 | accepted=%s",
                    name, item["pooled_mae"], item["participants_improved"], item["meets_adoption_rule"])
    logger.info("Combined out-of-fold results (14 development participants):")
    for name in sorted(scores, key=lambda key: scores[key]["mae"]):
        m = scores[name]
        logger.info("%-28s MAE=%.4f RMSE=%.4f R2=%.4f accuracy=%.1f%% balanced=%.1f%% macroF1=%.3f", name, m["mae"], m["rmse"], m["r2"], m["rounded_accuracy"] * 100, m["rounded_balanced_accuracy"] * 100, m["rounded_macro_f1"])
    development = data[data["participant"].isin(split["development"])]
    logger.info("Training final ML challenger (%s) on 14 development participants", best_ml)
    model = clone(candidates[best_ml]).fit(
        development[columns].replace([np.inf, -np.inf], np.nan), development["fatigue"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model, "feature_columns": columns, "prediction_strategy": winner,
        "selected_model": best_ml, "global_median": float(development["fatigue"].median()),
        "development": split["development"], "test": split["test"],
        "split_schema_version": split["schema_version"],
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
    }
    joblib.dump(artifact, output_dir / "fatigue_model.joblib")
    pd.DataFrame(log_rows).to_csv(output_dir / "validation_folds.csv", index=False)
    pd.DataFrame(person_rows).to_csv(output_dir / "participant_metrics.csv", index=False)
    report = {
        "validation": "7 participant-disjoint folds, exactly 12 train and 2 validation per fold",
        "selected_strategy": winner, "best_ml_challenger": best_ml,
        "selection_policy": "E2 group-A candidates: pooled MAE below personal median AND MAE improvement in at least 9/14 participants",
        "selection_policy_scope": "Only new group-A variants may replace the personal median; original models are comparisons",
        "unrestricted_lowest_mae_strategy": unrestricted_lowest_mae,
        "adoption_checks": eligibility, "per_person_metrics_file": "participant_metrics.csv",
        "regression_metrics": ["mae", "rmse", "r2"],
        "protocol_note": "Post-E2 policy implementation; not a preregistered rerun of E2",
        "out_of_fold_metrics": scores, "development_participants": 14,
        "test_participants": 2, "test_evaluated": False,
        "comparability_note": "Legacy and grouped-split metrics are not directly comparable: evaluated participants and protocols differ; a difference is not necessarily a model effect.",
    }
    (output_dir / "validation_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Split SHA256=%s | selected=%s | best ML=%s | validation MAE=%.4f", artifact["split_sha256"], winner, best_ml, scores[winner]["mae"])
    logger.info("Completed. Test participants have NOT been evaluated.")
    for handler in logger.handlers[:]:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("outputs/fatigue_daily.csv"))
    parser.add_argument("--split", type=Path, default=Path("outputs/split.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/train_grouped"))
    args = parser.parse_args()
    train(args.features, args.split, args.out_dir)
