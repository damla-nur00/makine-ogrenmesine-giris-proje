"""E2: Predeclared cumulative feature-family ablation (development participants only).

Outputs under outputs/ablation/:
  train.log, feature_mapping.json, validation_predictions.csv,
  participant_metrics.csv, results.csv, summary.json.

Unlike the original personal GBM/Ridge pipeline, EACH candidate deliberately
receives the full features of its declared A-E set. Original learner hyper-
parameters stay frozen, but these are feature-restricted experimental variants,
not numerically identical to the previously trained full-pipeline models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fatigue_models import (
    PERSONAL_COLUMNS,
    _continuous_metrics,
    _candidate_models,
    add_personal_features,
    baseline_predictions,
    model_feature_columns,
)
from split_participants import validate_split

EXPECTED_GROUPS = ("A", "B", "C", "D", "E")
EXPECTED_MODELS = ("gbm_abs_personal", "ridge_personal")
# The order matters: heart-rate zone activity belongs to E, not C.
GROUP_PREFIXES = {
    "A": ("history_wellness_fatigue_",),
    "B": ("history_sleep_", "morning_sleep_", "history_wellness_sleep_",
          "dv_sleep_", "dw_sleep_", "dv_wellness_sleep_", "dw_wellness_sleep_"),
    "C": ("history_cardio_resting_heart_rate_", "history_cardio_heart_rate_",
          "dv_cardio_", "dw_cardio_"),
    "D": ("history_srpe_", "dv_srpe_", "dw_srpe_"),
    "E": ("history_movement_", "history_activity_", "history_exercise_",
          "history_cardio_zone_", "dv_movement_", "dw_movement_",
          "dv_activity_", "dw_activity_", "dv_exercise_", "dw_exercise_"),
}
A_EXTRA = tuple(PERSONAL_COLUMNS) + ("fat_trend", "fat_vs_pers")
# Deliberately exclude day-of-week and self-reported mood/readiness/stress.
# We test only personal fatigue history + five preregistered feature families.


def feature_mapping(data: pd.DataFrame) -> dict:
    candidates = set(model_feature_columns(data))
    assigned = {}
    for column in sorted(candidates):
        possible = [
            group for group, prefixes in GROUP_PREFIXES.items()
            if column.startswith(prefixes)
        ]
        if column in A_EXTRA:
            possible.append("A")
        if len(possible) > 1:
            raise ValueError(f"Feature mapped to more than one group: {column}: {possible}")
        if possible:
            assigned[column] = possible[0]
    disjoint = {name: sorted(c for c, group in assigned.items() if group == name)
                for name in EXPECTED_GROUPS}
    # Ensure each incremental family has at least one genuine feature.
    for group in EXPECTED_GROUPS:
        if not disjoint[group]:
            raise ValueError(f"Feature family {group} has no columns; check PMData extraction.")
    if not set(PERSONAL_COLUMNS).issubset(disjoint["A"]):
        raise ValueError("Personal fatigue columns are missing.")
    cumulative = {}
    selected = []
    for name in EXPECTED_GROUPS:
        selected.extend(disjoint[name])
        cumulative[name] = list(selected)
    return {
        "disjoint_groups": disjoint,
        "cumulative_sets": cumulative,
        "excluded_model_columns": sorted(candidates - set(assigned)),
        "explanation": {
            "A": "Prior fatigue (shifted history, past personal medians/counts/trends)",
            "B": "A + historical and pre-report morning sleep + sleep deviations",
            "C": "B + historical heart rate and resting HR + HR deviations",
            "D": "C + historical sRPE and training load",
            "E": "D + historical movement, active minutes, exercise and HR zones",
        },
    }


def metrics(frame: pd.DataFrame) -> dict:
    """Regression, rounded class and descriptive high-fatigue metrics."""
    if frame.empty:
        return {"n": 0, "mae": None, "rmse": None, "r2": None}
    actual = frame["fatigue"].to_numpy(dtype=float)
    prediction = np.clip(frame["prediction"].to_numpy(dtype=float), 1, 5)
    # R² is undefined with one observation; pooled results are unaffected.
    if len(frame) == 1:
        basic = _continuous_metrics(pd.Series(np.r_[actual, actual]), np.r_[prediction, prediction])
        basic["r2"] = None
    else:
        basic = _continuous_metrics(pd.Series(actual), prediction)
    positives_predicted = int((prediction >= 3.5).sum())
    true_positives = int(((prediction >= 3.5) & (actual >= 4)).sum())
    p = basic["high_fatigue_precision"]
    r = basic["high_fatigue_recall"]
    f1 = (2 * p * r / (p + r) if p is not None and r is not None and p + r > 0
          else 0.0 if p is not None and r is not None else None)
    return {"n": int(len(frame)), **basic,
            "high_fatigue_predicted_positive": positives_predicted,
            "high_fatigue_true_positive": true_positives,
            "high_fatigue_f1": f1}


def differences_ci(left: pd.DataFrame, right: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Paired cluster bootstrap: ΔMAE = candidate - reference; negative is better."""
    keys = ["participant", "date"]
    merged = left[keys + ["fatigue", "prediction"]].merge(
        right[keys + ["prediction"]], on=keys, validate="one_to_one",
        suffixes=("_candidate", "_reference"),
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("Reference and candidate OOF keys differ.")
    groups = list(merged.groupby("participant", sort=True))
    diff_sums = np.asarray([
        (np.abs(g["fatigue"] - g["prediction_candidate"])
         - np.abs(g["fatigue"] - g["prediction_reference"])).sum()
        for _, g in groups
    ])
    ns = np.asarray([len(g) for _, g in groups])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    draws = diff_sums[indices].sum(axis=1) / ns[indices].sum(axis=1)
    return {
        "delta_mae": float(diff_sums.sum() / ns.sum()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "bootstrap_draws": int(n_boot),
    }


def run(features: Path, split_path: Path, protocol_path: Path, out_dir: Path) -> None:
    proto_bytes = protocol_path.read_bytes()
    proto = json.loads(proto_bytes)
    e2 = next(x for x in proto["experiments"] if x["id"] == "E2_feature_ablation")
    if tuple(x["id"] for x in e2["feature_sets"]) != EXPECTED_GROUPS:
        raise ValueError("E2 feature groups in protocol differ from implementation.")
    if tuple(e2["model_candidates"]) != EXPECTED_MODELS:
        raise ValueError("E2 frozen models differ from implementation.")
    if e2["baseline"] != "baseline_personal_median":
        raise ValueError("E2 baseline differs from implementation.")
    if (e2["adoption_rule"]["participants_with_strictly_lower_MAE_at_least"] != 9
            or e2["adoption_rule"]["of"] != 14):
        raise ValueError("E2 adoption criteria have changed.")
    split_bytes = split_path.read_bytes()
    split_sha = hashlib.sha256(split_bytes).hexdigest()
    if split_sha != proto["provenance"]["previous_grouped_split_sha256"]:
        raise ValueError("Sealed split hash does not match protocol. No training performed.")
    split = json.loads(split_bytes)
    validate_split(split)
    n_boot = int(proto["analysis_common"]["bootstrap"]["iterations"])
    seed = int(proto["analysis_common"]["random_seed"])

    # Restrict FIRST, before any target-derived feature engineering.
    raw = pd.read_csv(features, parse_dates=["date"])
    dev = set(split["development"])
    data = raw.loc[raw["participant"].isin(dev) & raw["fatigue"].notna()].copy()
    del raw
    if set(data["participant"].unique()) != dev:
        raise ValueError("Missing development participants.")
    if data.duplicated(["participant", "date"]).any():
        raise ValueError("Duplicate development participant/date.")

    data = add_personal_features(data)
    mapping = feature_mapping(data)
    columns_by_group = mapping["cumulative_sets"]
    # Take the EXACT learner and hyperparameter configuration from original model.
    full_candidates = _candidate_models(model_feature_columns(data))
    gbm = clone(full_candidates["gbm_abs_personal"].steps[-1][1])
    ridge = clone(full_candidates["ridge_personal"].steps[-1][1])

    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("personafit.feature_ablation")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(out_dir / "train.log", mode="w", encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    try:
        mapping["models"] = list(EXPECTED_MODELS)
        mapping["split_sha256"] = split_sha
        mapping["protocol_sha256"] = hashlib.sha256(proto_bytes).hexdigest()
        (out_dir / "feature_mapping.json").write_text(
            json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        logger.info("E2 started. 7 folds; 14 development participants; sealed test excluded.")
        for key in EXPECTED_GROUPS:
            logger.info("Feature set %s: %d columns (+%d newly assigned)",
                        key, len(columns_by_group[key]), len(mapping["disjoint_groups"][key]))
        records = []
        labels = ["baseline_personal_median"] + [
            f"{name}__{group}" for name in EXPECTED_MODELS for group in EXPECTED_GROUPS
        ]
        for fold in split["folds"]:
            train = data.loc[data["participant"].isin(fold["train"])]
            valid = data.loc[data["participant"].isin(fold["validation"])]
            if train.empty or valid.empty:
                raise ValueError("Empty development fold.")
            baseline = np.clip(
                baseline_predictions(valid, float(train["fatigue"].median()))[
                    "baseline_personal_median"
                ], 1, 5
            )
            pred = {"baseline_personal_median": baseline}
            for group in EXPECTED_GROUPS:
                columns = columns_by_group[group]
                x_train = train[columns].replace([np.inf, -np.inf], np.nan)
                x_valid = valid[columns].replace([np.inf, -np.inf], np.nan)
                for name in EXPECTED_MODELS:
                    if name == "gbm_abs_personal":
                        pipeline = make_pipeline(
                            SimpleImputer(strategy="median", keep_empty_features=True),
                            clone(gbm),
                        )
                    else:
                        pipeline = make_pipeline(
                            SimpleImputer(strategy="median", keep_empty_features=True),
                            StandardScaler(),
                            clone(ridge),
                        )
                    began = time.perf_counter()
                    pipeline.fit(x_train, train["fatigue"])
                    y_pred = np.clip(pipeline.predict(x_valid), 1, 5)
                    label = f"{name}__{group}"
                    pred[label] = y_pred
                    logger.info("Fold %d/7 | %-26s | MAE %.4f | %.1fs",
                                fold["fold"], label,
                                float(np.abs(y_pred - valid["fatigue"].to_numpy()).mean()),
                                time.perf_counter() - began)
            for label in labels:
                for participant, day, truth, prediction in zip(
                    valid["participant"], valid["date"], valid["fatigue"], pred[label]
                ):
                    records.append({
                        "fold": int(fold["fold"]),
                        "participant": str(participant),
                        "date": pd.Timestamp(day).strftime("%Y-%m-%d"),
                        "fatigue": float(truth),
                        "strategy": label,
                        "prediction": float(prediction),
                    })

        oof = pd.DataFrame(records)
        if len(oof) != len(data) * len(labels):
            raise ValueError("OOF row count mismatch.")
        if oof.duplicated(["participant", "date", "strategy"]).any():
            raise ValueError("Duplicate OOF predictions.")
        if set(oof["participant"]) != dev:
            raise ValueError("Unexpected participant in OOF.")
        oof.sort_values(["strategy", "participant", "date"]).to_csv(
            out_dir / "validation_predictions.csv", index=False, float_format="%.10f"
        )

        person_rows = []
        for name, frame in oof.groupby("strategy", sort=False):
            for person, sub in frame.groupby("participant", sort=True):
                person_rows.append({
                    "strategy": name, "participant": person, **metrics(sub)
                })
        person_df = pd.DataFrame(person_rows)
        person_df.to_csv(out_dir / "participant_metrics.csv", index=False)
        baseline = oof.loc[oof["strategy"].eq("baseline_personal_median")]
        baseline_score = metrics(baseline)
        grouped = {name: group for name, group in oof.groupby("strategy")}
        summary_rows = []

        for label in labels:
            frame = grouped[label]
            m = metrics(frame)
            delta = differences_ci(frame, baseline, n_boot, seed)
            current = person_df.loc[person_df["strategy"].eq(label)].set_index("participant")
            reference = person_df.loc[person_df["strategy"].eq("baseline_personal_median")].set_index("participant")
            improved = int((current["mae"] < reference["mae"]).sum())
            if label == "baseline_personal_median":
                group_name = "baseline"
                learner = "personal_median"
                against_A = None
            else:
                learner, group_name = label.split("__")
                against_A = (differences_ci(frame, grouped[f"{learner}__A"], n_boot, seed)
                             if group_name != "A" else None)
            adopt = (label != "baseline_personal_median"
                     and m["mae"] < baseline_score["mae"] and improved >= 9)
            row = {
                "strategy": label, "group": group_name, "model": learner,
                "n": m["n"], "participants": len(current), "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
                "rounded_accuracy": m["rounded_accuracy"],
                "rounded_balanced_accuracy": m["rounded_balanced_accuracy"],
                "rounded_macro_f1": m["rounded_macro_f1"],
                "high_fatigue_recall": m["high_fatigue_recall"],
                "high_fatigue_precision": m["high_fatigue_precision"],
                "high_fatigue_f1": m["high_fatigue_f1"],
                "high_fatigue_support": m["high_fatigue_support"],
                "high_fatigue_predicted_positive": m["high_fatigue_predicted_positive"],
                "high_fatigue_true_positive": m["high_fatigue_true_positive"],
                "participants_improved_vs_personal_median": improved,
                "delta_mae_vs_personal_median": delta["delta_mae"],
                "delta_mae_ci95_low": delta["ci95_low"],
                "delta_mae_ci95_high": delta["ci95_high"],
                "delta_mae_vs_group_A": against_A["delta_mae"] if against_A else None,
                "delta_group_A_ci95_low": against_A["ci95_low"] if against_A else None,
                "delta_group_A_ci95_high": against_A["ci95_high"] if against_A else None,
                "meets_preregistered_adoption_rule": bool(adopt),
                "n_features": len(columns_by_group[group_name]) if group_name != "baseline" else 0,
            }
            summary_rows.append(row)
            logger.info(
                "OOF | %-26s | MAE %.4f RMSE %.4f R2 %s | balanced %.1f%% macroF1 %.3f | high recall=%s precision=%s (n=%d) | improved %d/14 | ΔMAE %.4f [%.4f, %.4f] | adoption=%s",
                label, m["mae"], m["rmse"],
                f'{m["r2"]:.4f}' if m["r2"] is not None else "N/A",
                m["rounded_balanced_accuracy"] * 100, m["rounded_macro_f1"],
                f'{m["high_fatigue_recall"]:.1%}' if m["high_fatigue_recall"] is not None else "N/A",
                f'{m["high_fatigue_precision"]:.1%}' if m["high_fatigue_precision"] is not None else "N/A",
                m["high_fatigue_support"], improved, delta["delta_mae"],
                delta["ci95_low"], delta["ci95_high"], adopt
            )

        pd.DataFrame(summary_rows).to_csv(out_dir / "results.csv", index=False)
        report = {
            "experiment": "E2_feature_ablation",
            "status": "development_only",
            "split_sha256": split_sha,
            "protocol_sha256": hashlib.sha256(proto_bytes).hexdigest(),
            "validation": "7 participant-disjoint folds; 12 training / 2 validation participants",
            "development_participants": len(dev),
            "validation_days": len(data),
            "test_evaluated": False,
            "bootstrap": {"unit": "participant", "iterations": n_boot, "seed": seed, "confidence": 0.95},
            "model_note": "Experimental variants retain original learner hyperparameters but explicitly feed each cumulative feature set into each estimator. Results should NOT be called exact reproductions of original full-feature pipelines.",
            "feature_mapping_file": "feature_mapping.json",
            "adoption_rule": e2["adoption_rule"],
            "metric_note": "Regression: MAE, RMSE, R². Rounded classes: accuracy, balanced accuracy and macro-F1 (1–5, np.rint). Descriptive high fatigue: truth >=4 and predicted regression score >=3.5; recall, precision, F1 and support. The E3 ROC-AUC/AP/Recall@10% task remains separate. E2 adoption still depends only on pooled MAE and >=9/14 people improving.",
            "results": summary_rows,
        }
        (out_dir / "summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8"
        )
        logger.info("E2 complete. Outputs: %s. Sealed test never evaluated.", out_dir)
    finally:
        for handler in logger.handlers[:]:
            handler.flush()
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("outputs/fatigue_daily.csv"))
    parser.add_argument("--split", type=Path, default=Path("outputs/split.json"))
    parser.add_argument("--protocol", type=Path, default=Path("experiment_protocol.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/experiment_ablation"))
    args = parser.parse_args()
    run(args.features, args.split, args.protocol, args.out_dir)
