"""Cold-start analysis of three frozen strategies on development participants only.

Outputs:
  outputs/cold_start/train.log
  outputs/cold_start/validation_predictions.csv
  outputs/cold_start/segment_metrics.csv
  outputs/cold_start/participant_metrics.csv
  outputs/cold_start/summary.json

No test labels are scored, no split is altered, no model artifact is replaced.
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

from fatigue_models import (
    _candidate_models,
    add_personal_features,
    baseline_predictions,
    model_feature_columns,
)
from split_participants import validate_split

STRATEGIES = (
    "baseline_train_median",
    "baseline_personal_median",
    "gbm_abs_personal",
)


def score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.clip(np.asarray(predicted, dtype=float), 1, 5)
    if len(actual) == 0:
        return {"n": 0, "mae": None, "rmse": None}
    error = actual - predicted
    return {
        "n": int(len(actual)),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
    }


def cluster_bootstrap(frame: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Resample WHOLE participants, preserving each repeated participant cluster."""
    groups = [p for _, p in frame.groupby("participant", sort=True)]
    if not groups:
        return {"mae_ci95": [None, None], "rmse_ci95": [None, None], "valid_draws": 0}
    abs_sums = np.asarray([np.abs(p["fatigue"] - p["prediction"]).sum() for p in groups])
    sq_sums = np.asarray([np.square(p["fatigue"] - p["prediction"]).sum() for p in groups])
    counts = np.asarray([len(p) for p in groups])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    denominators = counts[samples].sum(axis=1)
    mae_draws = abs_sums[samples].sum(axis=1) / denominators
    rmse_draws = np.sqrt(sq_sums[samples].sum(axis=1) / denominators)
    return {
        "mae_ci95": [float(x) for x in np.quantile(mae_draws, [0.025, 0.975])],
        "rmse_ci95": [float(x) for x in np.quantile(rmse_draws, [0.025, 0.975])],
        "valid_draws": int(n_boot),
    }


def main(features: Path, split_path: Path, protocol_path: Path, out_dir: Path) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    experiment = next(x for x in protocol["experiments"] if x["id"] == "E1_cold_start")
    bootstrap_config = protocol["analysis_common"]["bootstrap"]
    if tuple(experiment["strategies"]) != STRATEGIES:
        raise ValueError("The frozen cold-start strategy list has changed.")
    if float(experiment["personal_median_shrinkage_k"]) != 5.0:
        raise ValueError("The frozen personal median shrinkage k must remain 5.")
    if [(x["prior_labeled_observations_min"], x["prior_labeled_observations_max"])
            for x in experiment["bins"]] != [(0, 6), (7, 13), (14, None)]:
        raise ValueError("The frozen cold-start bins have changed.")
    n_boot = int(bootstrap_config["iterations"])
    seed = int(protocol["analysis_common"]["random_seed"])

    split_bytes = split_path.read_bytes()
    split_digest = hashlib.sha256(split_bytes).hexdigest()
    if split_digest != protocol["provenance"]["previous_grouped_split_sha256"]:
        raise ValueError("Sealed split checksum differs from the frozen protocol.")
    split = json.loads(split_bytes.decode("utf-8"))
    validate_split(split)

    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("personafit.cold_start")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(out_dir / "train.log", mode="w", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    try:
        logger.info("Cold-start experiment E1 started. Test participants excluded; k=5 unchanged.")
        # Filter to development participants BEFORE any label-based feature engineering.
        raw = pd.read_csv(features, parse_dates=["date"])
        development_ids = set(split["development"])
        data = raw.loc[raw["participant"].isin(development_ids) & raw["fatigue"].notna()].copy()
        del raw
        if set(data["participant"].unique()) != development_ids:
            raise ValueError("One or more development participants have no labeled records.")
        if data.duplicated(["participant", "date"]).any():
            raise ValueError("Duplicate participant/date records found.")

        data = add_personal_features(data)
        columns = model_feature_columns(data)
        model = _candidate_models(columns)["gbm_abs_personal"]
        results = []

        for fold in split["folds"]:
            train = data.loc[data["participant"].isin(fold["train"])].copy()
            valid = data.loc[data["participant"].isin(fold["validation"])].copy()
            if train.empty or valid.empty:
                raise ValueError("Empty fold.")
            x_train = train[columns].replace([np.inf, -np.inf], np.nan)
            x_valid = valid[columns].replace([np.inf, -np.inf], np.nan)
            global_median = float(train["fatigue"].median())
            baseline = baseline_predictions(valid, global_median)
            start = time.perf_counter()
            regressor = clone(model).fit(x_train, train["fatigue"])
            predictions = {
                "baseline_train_median": baseline["baseline_train_median"],
                "baseline_personal_median": baseline["baseline_personal_median"],
                "gbm_abs_personal": regressor.predict(x_valid),
            }
            logger.info("Fold %d/7 | trained on %d participants; validated on %d participants, %d days; GBM %.1fs",
                        fold["fold"], len(fold["train"]), len(fold["validation"]), len(valid), time.perf_counter() - start)
            for name in STRATEGIES:
                predicted = np.clip(np.asarray(predictions[name], dtype=float), 1, 5)
                for participant, date, prior_count, truth, pred in zip(
                        valid["participant"], valid["date"], valid["p_n"], valid["fatigue"], predicted):
                    n = int(prior_count)
                    period = "1-7" if n <= 6 else ("8-14" if n <= 13 else "15+")
                    results.append({
                        "fold": fold["fold"], "participant": participant,
                        "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                        "prior_labeled_days": n,
                        "labeled_day_number": n + 1,
                        "period": period, "strategy": name,
                        "fatigue": float(truth), "prediction": float(pred),
                    })

        oof = pd.DataFrame(results).sort_values(["strategy", "participant", "date"])
        expected_rows = len(data) * len(STRATEGIES)
        if len(oof) != expected_rows or oof.duplicated(["participant", "date", "strategy"]).any():
            raise ValueError("Out-of-fold predictions are missing or duplicated.")
        if set(oof["participant"]) != development_ids:
            raise ValueError("Unexpected participant in out-of-fold predictions.")

        oof.to_csv(out_dir / "validation_predictions.csv", index=False, float_format="%.8f")
        summary_rows, per_person_rows = [], []
        for period in ["1-7", "8-14", "15+", "ALL"]:
            for name in STRATEGIES:
                subset = oof.loc[oof["strategy"].eq(name)]
                if period != "ALL":
                    subset = subset.loc[subset["period"].eq(period)]
                metrics = score(subset["fatigue"], subset["prediction"])
                ci = cluster_bootstrap(subset, n_boot, seed)
                row = {
                    "period": period, "strategy": name,
                    "participants": int(subset["participant"].nunique()),
                    **metrics,
                    "mae_ci95_low": ci["mae_ci95"][0],
                    "mae_ci95_high": ci["mae_ci95"][1],
                    "rmse_ci95_low": ci["rmse_ci95"][0],
                    "rmse_ci95_high": ci["rmse_ci95"][1],
                    "bootstrap_valid_draws": ci["valid_draws"],
                }
                summary_rows.append(row)
                if metrics["n"]:
                    logger.info("%-4s | %-26s | n=%4d people=%2d MAE=%.4f [%.4f, %.4f] RMSE=%.4f",
                                period, name, metrics["n"], row["participants"], metrics["mae"],
                                row["mae_ci95_low"], row["mae_ci95_high"], metrics["rmse"])
                for person, group in subset.groupby("participant", sort=True):
                    per_person_rows.append({
                        "period": period, "strategy": name, "participant": person,
                        **score(group["fatigue"], group["prediction"]),
                    })

        pd.DataFrame(summary_rows).to_csv(out_dir / "segment_metrics.csv", index=False)
        pd.DataFrame(per_person_rows).to_csv(out_dir / "participant_metrics.csv", index=False)
        report = {
            "experiment": "E1_cold_start",
            "status": "development_only",
            "split_sha256": split_digest,
            "protocol_path": str(protocol_path),
            "strategies": list(STRATEGIES),
            "shrinkage_k": 5,
            "bootstrap": {"iterations": n_boot, "seed": seed, "cluster": "participant", "confidence_level": 0.95},
            "validation": "7 participant-disjoint folds; 12 train and 2 validation people per fold",
            "total_validation_days": len(data),
            "test_evaluated": False,
            "metrics": summary_rows,
            "interpretation": "Descriptive only; no thresholds or hyperparameters changed. Early history uses prior labeled observations (online follow-up), not absence of all user labels.",
        }
        (out_dir / "summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        logger.info("Done. Results saved in %s ; sealed test NOT evaluated.", out_dir)
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
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/experiment_cold_start"))
    args = parser.parse_args()
    main(args.features, args.split, args.protocol, args.out_dir)
