"""E3 supplementary robustness: compare the frozen 7-fold OOF against 14-fold LOPO.

Creates NEW files under outputs/lopo_comparison. Does not alter any existing
experiment, model, protocol, split or 7-fold result. Never evaluates test people.
No LOPO-based model/hyperparameter/threshold selection is permitted.

Usage:
  python experiment_lopo.py
Requires completed outputs/high_fatigue_e3/validation_predictions.csv.
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

from experiment_ablation import feature_mapping
from experiment_high_fatigue import (
    cluster_bootstrap,
    frozen_models,
    measure,
    person_auc,
    require,
)
from fatigue_models import add_personal_features, baseline_predictions
from split_participants import validate_split


STRATEGIES = (
    "logistic_A", "random_forest_A", "logistic_B",
    "personal_median_reference", "previous_fatigue_reference",
)
KEYS = ["participant", "date", "strategy"]
METRIC_NAMES = (
    "roc_auc", "average_precision", "recall_at_10pct",
    "precision_at_10pct", "fpr_at_10pct", "within_person_mean_roc_auc",
)


def _read_frozen_7fold(path: Path, data: pd.DataFrame) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"7-fold OOF predictions missing: {path}. Run the unchanged "
            "experiment_high_fatigue.py first; this script does not rerun it."
        )
    frame = pd.read_csv(path, dtype={"participant": str, "date": str, "strategy": str})
    expected = {"fold", "participant", "date", "fatigue", "high", "strategy", "score"}
    require(expected.issubset(frame.columns),
            f"7-fold predictions missing fields: {sorted(expected - set(frame.columns))}")
    require(set(frame["strategy"]) == set(STRATEGIES), "7-fold strategy list has changed.")
    require(len(frame) == len(data) * len(STRATEGIES), "7-fold row count differs.")
    require(not frame.duplicated(KEYS).any(), "Duplicate 7-fold OOF keys.")
    require(np.isfinite(frame["score"]).all(), "7-fold scores must be finite.")
    expected_labels = data[["participant", "date", "fatigue"]].copy()
    expected_labels["participant"] = expected_labels["participant"].astype(str)
    expected_labels["date"] = pd.to_datetime(expected_labels["date"]).dt.strftime("%Y-%m-%d")
    expected_labels["high"] = (expected_labels["fatigue"] >= 4).astype(int)
    for strategy, part in frame.groupby("strategy", sort=True):
        matched = part.merge(
            expected_labels, on=["participant", "date"],
            how="outer", validate="one_to_one", indicator=True, suffixes=("", "_source")
        )
        require((matched["_merge"] == "both").all(),
                f"7-fold OOF participants/days differ for {strategy}.")
        require(np.allclose(matched["fatigue"], matched["fatigue_source"], equal_nan=False),
                f"7-fold fatigue labels differ for {strategy}.")
        require((matched["high"].astype(int) == matched["high_source"].astype(int)).all(),
                f"7-fold binary labels differ for {strategy}.")
        require(matched["fold"].between(1, 7).all(), "Unexpected 7-fold index.")
    return frame


def _metrics(frame: pd.DataFrame, *, top_fraction: float) -> tuple[dict, pd.DataFrame]:
    evaluation = frame.reset_index(drop=True).copy()
    evaluation["_draw"] = 0
    pooled = measure(evaluation, top_fraction=top_fraction)
    detail, within = person_auc(evaluation)
    result = {
        **pooled,
        "within_person_mean_roc_auc": within["mean"],
        "within_person_std_roc_auc": within["std"],
        "eligible_people": within["eligible_n"],
        "ineligible_people": within["ineligible_n"],
        "eligible_ids": ",".join(within["eligible_ids"]),
        "ineligible_ids": ",".join(within["ineligible_ids"]),
    }
    return result, detail


def run(features: Path, split_path: Path, protocol_path: Path, config_path: Path,
        seven_fold_predictions: Path, out_dir: Path) -> None:
    # All checks occur before creating output files.
    protocol_bytes = protocol_path.read_bytes()
    config_bytes = config_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    config = json.loads(config_bytes)
    require(config["experiment"] == "E3_high_fatigue", "Unexpected frozen E3 configuration.")
    require(config["feature_count_assertions"] == {"A": 13, "B": 103},
            "Frozen feature counts changed.")
    require(tuple(m["name"] for m in config["models"]) == STRATEGIES[:3],
            "Risk models changed.")
    require(tuple(b["name"] for b in config["reference_scores"]) == STRATEGIES[3:],
            "Baseline strategies changed.")
    protocol_e3 = next(e for e in protocol["experiments"] if e["id"] == "E3_high_fatigue")
    require(protocol["validation"]["secondary"]["selection_allowed"] is False,
            "LOPO was not specified as supplementary-only.")
    split_bytes = split_path.read_bytes()
    digest = hashlib.sha256(split_bytes).hexdigest()
    require(digest == protocol["provenance"]["previous_grouped_split_sha256"],
            "Sealed split checksum differs; refusing to run.")
    split = json.loads(split_bytes)
    validate_split(split)
    n_boot = int(protocol["analysis_common"]["bootstrap"]["iterations"])
    seed = int(protocol["analysis_common"]["random_seed"])
    require(n_boot == int(config["statistics"]["bootstrap_participants"]) == 2000,
            "Frozen bootstrap count changed.")
    require(seed == int(config["statistics"]["bootstrap_seed"]) == 42,
            "Frozen bootstrap seed changed.")
    fraction = float(config["statistics"]["top_fraction"])
    require(fraction == .10, "Frozen alert budget differs.")
    require(protocol_e3["primary_task"] == "binary classification: fatigue >= 4",
            "Target label rule changed.")

    raw = pd.read_csv(features, parse_dates=["date"])
    development_ids = set(split["development"])
    data = raw.loc[raw["participant"].isin(development_ids) & raw["fatigue"].notna()].copy()
    del raw
    require(set(data["participant"]) == development_ids, "Missing development people.")
    require(not data.duplicated(["participant", "date"]).any(),
            "Duplicate development days.")
    data = add_personal_features(data)
    groups = feature_mapping(data)["cumulative_sets"]
    for name, expected in config["feature_count_assertions"].items():
        require(len(groups[name]) == expected,
                f"Feature set {name} must have {expected} columns; got {len(groups[name])}.")
    require(len(data) == 1512, "Development day count differs from frozen E3 evaluation.")
    existing = _read_frozen_7fold(seven_fold_predictions, data)
    # Reuse the identical implementations and fixed hyperparameters from E3.
    model_templates = frozen_models(config)

    if out_dir.resolve() == seven_fold_predictions.parent.resolve():
        raise ValueError("LOPO outputs must never overwrite the 7-fold output directory.")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("personafit.lopo")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(out_dir / "train.log", mode="w", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    try:
        logger.info("Start 14-fold LOPO: %d development users, test excluded.", len(development_ids))
        logger.info("7-fold data read-only from: %s", seven_fold_predictions)
        start = time.perf_counter()
        records = []
        for fold_num, participant in enumerate(sorted(development_ids), start=1):
            train = data.loc[data["participant"] != participant]
            valid = data.loc[data["participant"] == participant]
            require(train["participant"].nunique() == 13 and valid["participant"].nunique() == 1,
                    "LOPO split must use 13 training / 1 validation person.")
            y_train = (train["fatigue"].to_numpy() >= 4).astype(int)
            require(len(np.unique(y_train)) == 2, "One LOPO training fold lacks a risk class.")
            predictions = {}
            for entry in config["models"]:
                name = entry["name"]
                columns = groups[entry["features"]]
                x_train = train[columns].replace([np.inf, -np.inf], np.nan)
                x_valid = valid[columns].replace([np.inf, -np.inf], np.nan)
                # New, unfitted pipeline for each fold.
                estimator = frozen_models(config)[name]
                estimator.fit(x_train, y_train)
                positive_col = list(estimator.classes_).index(1)
                predictions[name] = estimator.predict_proba(x_valid)[:, positive_col]
            baseline = baseline_predictions(valid, float(train["fatigue"].median()))
            predictions["personal_median_reference"] = np.clip(
                baseline["baseline_personal_median"], 1, 5
            )
            predictions["previous_fatigue_reference"] = np.clip(
                baseline["baseline_previous_fatigue"], 1, 5
            )
            for strategy in STRATEGIES:
                for person, date, target, prediction in zip(
                    valid["participant"], valid["date"], valid["fatigue"], predictions[strategy]
                ):
                    records.append({
                        "fold": fold_num, "participant": str(person),
                        "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                        "fatigue": float(target),
                        "high": int(target >= 4),
                        "strategy": strategy, "score": float(prediction),
                    })
            logger.info("LOPO %02d/14 | held-out=%s | valid=%d days, positives=%d | elapsed %.1fs",
                        fold_num, participant, len(valid), int((valid["fatigue"] >= 4).sum()),
                        time.perf_counter() - start)
        lo = pd.DataFrame(records)
        require(len(lo) == len(existing), "7-fold and LOPO row counts differ.")
        require(set(lo["strategy"]) == set(STRATEGIES), "LOPO strategy list differs.")
        require(not lo.duplicated(KEYS).any(), "Duplicated LOPO predictions.")
        require(np.isfinite(lo["score"]).all(), "Non-finite LOPO scores.")
        require(set(lo["participant"]) == development_ids, "LOPO evaluated unexpected person.")
        # Verify paired comparison on exactly the same person-date-label rows.
        merged = lo.merge(
            existing[KEYS + ["fatigue", "high", "score"]],
            on=KEYS, how="outer", validate="one_to_one",
            indicator=True, suffixes=("_lopo", "_7fold")
        )
        require((merged["_merge"] == "both").all(), "Unpaired OOF days across CV schemes.")
        require(np.allclose(merged["fatigue_lopo"], merged["fatigue_7fold"]),
                "Target values differ between CV schemes.")
        require((merged["high_lopo"].astype(int) == merged["high_7fold"].astype(int)).all(),
                "Target class differs between CV schemes.")

        lo.sort_values(KEYS).to_csv(
            out_dir / "lopo_predictions.csv", index=False, float_format="%.10f"
        )

        comparison = []
        per_person = []
        bootstrap = {}
        for strategy in STRATEGIES:
            seven = existing.loc[existing["strategy"] == strategy].copy()
            lopo = lo.loc[lo["strategy"] == strategy].copy()
            seven_metrics, seven_person = _metrics(seven, top_fraction=fraction)
            lopo_metrics, lo_person = _metrics(lopo, top_fraction=fraction)
            for scheme, details in (("7fold", seven_person), ("lopo14", lo_person)):
                details.insert(0, "strategy", strategy)
                details.insert(0, "scheme", scheme)
                per_person.append(details)
            for metric in METRIC_NAMES:
                x, y = seven_metrics[metric], lopo_metrics[metric]
                comparison.append({
                    "strategy": strategy,
                    "metric": metric,
                    "seven_fold": x,
                    "lopo_14": y,
                    "delta_lopo_minus_7fold": y - x if x is not None and y is not None else None,
                    "interpretation": "supplementary robustness only; no model selection",
                })
            logger.info("LOPO bootstrap %s (2000 participant draws) ...", strategy)
            frame = lopo.reset_index(drop=True)
            frame["_draw"] = 0
            intervals = cluster_bootstrap(frame, n_boot, seed, fraction)
            bootstrap[strategy] = intervals
            logger.info("Compare %-29s | ROC-AUC %.3f -> %.3f | AP %.3f -> %.3f | Recall@10 %.3f -> %.3f",
                        strategy, seven_metrics["roc_auc"], lopo_metrics["roc_auc"],
                        seven_metrics["average_precision"], lopo_metrics["average_precision"],
                        seven_metrics["recall_at_10pct"], lopo_metrics["recall_at_10pct"])
        pd.DataFrame(comparison).to_csv(out_dir / "comparison.csv", index=False)
        pd.concat(per_person, ignore_index=True).to_csv(
            out_dir / "participant_metrics.csv", index=False
        )
        (out_dir / "summary.json").write_text(
            json.dumps({
                "experiment": "E3_LOPO_robustness",
                "purpose": "Supplementary 14-fold LOPO versus frozen 7-fold E3; no selection",
                "selection_allowed": False,
                "test_evaluated": False,
                "development_people": len(development_ids),
                "evaluated_days_per_strategy": len(data),
                "split_sha256": digest,
                "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
                "candidate_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "seven_fold_predictions_source": str(seven_fold_predictions),
                "models": list(STRATEGIES),
                "feature_sizes": {"A": len(groups["A"]), "B": len(groups["B"])},
                "metric_note": "7-fold and LOPO predictions use same development days but different training subsets. These results are correlated, not independent external validation.",
                "ranking_note": "Top 10% means ceil(0.10 * 1512) highest out-of-fold scores. Uncalibrated outputs are NOT clinical risk probabilities.",
                "target_thresholds_unchanged": protocol_e3["preregistered_targets"],
                "bootstrap": {"unit": "participant", "draws": n_boot, "seed": seed,
                              "lopo_confidence_intervals": bootstrap},
                "comparison": comparison,
                "no_test_based_tuning": True,
            }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        logger.info("Completed; all outputs under %s. No test participant used.", out_dir)
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
    parser.add_argument("--config", type=Path, default=Path("e3_candidate_config.json"))
    parser.add_argument("--seven-fold-predictions", type=Path,
                        default=Path("outputs/experiment_high_fatigue/validation_predictions.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/experiment_lopo"))
    args = parser.parse_args()
    run(args.features, args.split, args.protocol, args.config,
        args.seven_fold_predictions, args.out_dir)
