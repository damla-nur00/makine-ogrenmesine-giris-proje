"""E3: high-fatigue (fatigue >= 4) participant-disjoint OOF classification.

Candidates are frozen in e3_candidate_config.json BEFORE local E3 execution.
This experiment does NOT alter level-prediction policy, saved models, or split.
Logistic/RF predict_proba is uncalibrated: interpret as ranking scores only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiment_ablation import feature_mapping
from fatigue_models import add_personal_features, baseline_predictions
from split_participants import validate_split


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def frozen_models(config: dict) -> dict:
    result = {}
    for entry in config["models"]:
        name = entry["name"]
        p = entry["hyperparameters"]
        if name == "logistic_A" or name == "logistic_B":
            require(entry["algorithm"] == "LogisticRegression", "Unexpected logistic model.")
            result[name] = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler(),
                LogisticRegression(**p),
            )
        elif name == "random_forest_A":
            require(entry["algorithm"] == "RandomForestClassifier", "Unexpected forest model.")
            result[name] = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                RandomForestClassifier(**p),
            )
        else:
            raise ValueError(f"Unfrozen candidate: {name}")
    require(set(result) == {"logistic_A", "random_forest_A", "logistic_B"}, "Candidate list changed.")
    return result


def top_indices(frame: pd.DataFrame, fraction: float = .10) -> np.ndarray:
    """Select exactly ceil(fraction*n), score descending, participant/date ascending."""
    n = len(frame)
    k = math.ceil(fraction * n)
    order = frame.sort_values(
        ["score", "participant", "date", "_draw"],
        ascending=[False, True, True, True], kind="mergesort"
    )
    return order.index.to_numpy()[:k]


def measure(frame: pd.DataFrame, *, top_fraction: float = .10) -> dict:
    """All metrics are calculated on OOF scores; may not be calibrated probabilities."""
    n = len(frame)
    positives = int(frame["high"].sum())
    negatives = n - positives
    result = {
        "n": n, "positive_support": positives,
        "negative_support": negatives,
        "prevalence": float(positives / n) if n else None,
        "roc_auc": None, "average_precision": None,
        "recall_at_10pct": None, "precision_at_10pct": None,
        "fpr_at_10pct": None, "top_k": math.ceil(n * top_fraction),
        "top_k_true_positive": None, "top_k_false_positive": None,
    }
    if not n:
        return result
    if positives and negatives:
        result["roc_auc"] = float(roc_auc_score(frame["high"], frame["score"]))
        result["average_precision"] = float(average_precision_score(frame["high"], frame["score"]))
    selected = frame.loc[top_indices(frame, top_fraction)]
    tp = int(selected["high"].sum())
    fp = len(selected) - tp
    result.update({
        "top_k_true_positive": tp,
        "top_k_false_positive": fp,
        "recall_at_10pct": float(tp / positives) if positives else None,
        "precision_at_10pct": float(tp / len(selected)) if len(selected) else None,
        "fpr_at_10pct": float(fp / negatives) if negatives else None,
    })
    return result


def person_auc(frame: pd.DataFrame, *, min_each: int = 3) -> tuple[pd.DataFrame, dict]:
    records = []
    for person, sample in frame.groupby("participant", sort=True):
        positive = int(sample["high"].sum())
        negative = len(sample) - positive
        eligible = positive >= min_each and negative >= min_each
        records.append({
            "participant": str(person), "n": len(sample),
            "positive_support": positive, "negative_support": negative,
            "eligible": bool(eligible),
            "roc_auc": float(roc_auc_score(sample["high"], sample["score"])) if eligible else None,
        })
    details = pd.DataFrame(records)
    eligible = details.loc[details["eligible"]]
    vals = eligible["roc_auc"]
    return details, {
        "mean": float(vals.mean()) if len(vals) else None,
        "std": float(vals.std(ddof=1)) if len(vals) > 1 else None,
        "eligible_n": int(len(eligible)),
        "ineligible_n": int(len(details) - len(eligible)),
        "eligible_ids": eligible["participant"].tolist(),
        "ineligible_ids": details.loc[~details["eligible"], "participant"].tolist(),
    }


def cluster_bootstrap(frame: pd.DataFrame, n_boot: int, seed: int, top_fraction: float) -> dict:
    """Resample participants WITH replacement; duplicated clusters remain duplicated.

    For the pooled ranking each duplicate is its own sampled cluster. Every draw
    reranks all sampled rows, retaining the exact 10% alert capacity.
    """
    groups = [part.copy() for _, part in frame.groupby("participant", sort=True)]
    require(len(groups) == 14, "Expected 14 participant clusters for bootstrap.")
    rng = np.random.default_rng(seed)
    names = [
        "roc_auc", "average_precision", "recall_at_10pct", "precision_at_10pct",
        "fpr_at_10pct", "within_person_mean_roc_auc",
    ]
    draws = {name: [] for name in names}
    invalid_counts = {name: 0 for name in names}
    # Within-person AUCs need no recalculation on resampled duplicates.
    individual, _ = person_auc(frame)
    auc_by_person = {row["participant"]: row["roc_auc"] for _, row in individual.iterrows()}
    person_keys = [str(g["participant"].iloc[0]) for g in groups]
    for _ in range(n_boot):
        ids = rng.integers(0, len(groups), size=len(groups))
        sampled = []
        for draw_index, idx in enumerate(ids):
            sub = groups[idx].copy()
            sub["_draw"] = draw_index  # deterministic tie-break among duplicated people
            sampled.append(sub)
        combined = pd.concat(sampled, ignore_index=True)
        pooled = measure(combined, top_fraction=top_fraction)
        for name in names[:-1]:
            value = pooled[name]
            if value is None:
                invalid_counts[name] += 1
            else:
                draws[name].append(float(value))
        au = [auc_by_person[person_keys[idx]] for idx in ids
              if pd.notna(auc_by_person[person_keys[idx]])]
        if au:
            draws["within_person_mean_roc_auc"].append(float(np.mean(au)))
        else:
            invalid_counts["within_person_mean_roc_auc"] += 1
    return {
        name: {
            "ci95": [float(x) for x in np.quantile(draws[name], [0.025, 0.975])]
            if draws[name] else [None, None],
            "valid_draws": len(draws[name]),
            "invalid_draws": invalid_counts[name],
        }
        for name in names
    }


def main(features: Path, split_path: Path, protocol_path: Path,
         config_path: Path, out_dir: Path) -> None:
    protocol_bytes = protocol_path.read_bytes()
    config_bytes = config_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    config = json.loads(config_bytes)
    e3 = next(e for e in protocol["experiments"] if e["id"] == "E3_high_fatigue")
    require(config["experiment"] == "E3_high_fatigue", "Incorrect experiment config.")
    require(config["label_rule"] == "fatigue >= 4; fatigue < 4 is negative, including 3.5",
            "Target definition changed.")
    require(config["feature_count_assertions"] == {"A": 13, "B": 103},
            "Frozen feature counts changed.")
    require(len(config["reference_scores"]) == 2, "Baseline count differs from frozen scope.")
    require([x["name"] for x in config["models"]] ==
            ["logistic_A", "random_forest_A", "logistic_B"], "Candidate order changed.")
    require(float(e3["preregistered_targets"]["pooled_ROC_AUC_min"]) == .65,
            "Unexpected E3 AUC target.")
    require(float(e3["preregistered_targets"]["average_precision_min"]) == .15,
            "Unexpected E3 AP target.")
    require(float(e3["preregistered_targets"]["recall_at_10pct_min"]) == .30,
            "Unexpected E3 recall target.")
    require(float(e3["preregistered_targets"]["within_person_mean_ROC_AUC_min"]) == .60,
            "Unexpected within-person AUC target.")

    split_bytes = split_path.read_bytes()
    split_sha = hashlib.sha256(split_bytes).hexdigest()
    require(split_sha == protocol["provenance"]["previous_grouped_split_sha256"],
            "Sealed split checksum changed; refusing to run.")
    split = json.loads(split_bytes)
    validate_split(split)
    require(set(split["development"]).isdisjoint(split["test"]), "Test contamination.")
    n_boot = int(protocol["analysis_common"]["bootstrap"]["iterations"])
    seed = int(protocol["analysis_common"]["random_seed"])
    require(n_boot == int(config["statistics"]["bootstrap_participants"]) == 2000,
            "Bootstrap count differs from frozen configuration.")
    require(seed == int(config["statistics"]["bootstrap_seed"]) == 42,
            "Bootstrap random seed differs from frozen configuration.")
    top_fraction = float(config["statistics"]["top_fraction"])
    require(top_fraction == .1, "Top fraction must stay 10%.")
    require(int(config["statistics"]["within_person_min_positives"]) == 3
            and int(config["statistics"]["within_person_min_negatives"]) == 3,
            "Within-person eligibility changed.")

    raw = pd.read_csv(features, parse_dates=["date"])
    dev = set(split["development"])
    data = raw.loc[raw["participant"].isin(dev) & raw["fatigue"].notna()].copy()
    del raw
    require(set(data["participant"].unique()) == dev, "Missing development participant.")
    require(not data.duplicated(["participant", "date"]).any(),
            "Duplicate participant/day records.")
    data = add_personal_features(data)
    mapping = feature_mapping(data)
    groups = mapping["cumulative_sets"]
    for name, count in config["feature_count_assertions"].items():
        require(len(groups[name]) == count,
                f"Expected {count} {name} columns but got {len(groups[name])}; check data generation.")
    models = frozen_models(config)

    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("personafit.e3_high_fatigue")
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
        columns_report = {
            "A": groups["A"], "B": groups["B"],
            "sizes": {g: len(groups[g]) for g in ("A", "B")},
            "method": "Same feature_mapping as E2; no feature changes after observing E3 outcomes.",
        }
        (out_dir / "feature_columns.json").write_text(
            json.dumps(columns_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        require((data["fatigue"] >= 4).sum() > 0, "No positive labels in development data.")
        logger.info("E3 high fatigue risk started. Participants=%d days=%d positives=%d | test excluded",
                    len(dev), len(data), int((data["fatigue"] >= 4).sum()))
        logger.info("Candidates: logistic_A(13), random_forest_A(13), logistic_B(103); 2 score baselines.")
        records = []
        strategies = [m["name"] for m in config["models"]] + [
            r["name"] for r in config["reference_scores"]
        ]
        for fold in split["folds"]:
            training = data.loc[data["participant"].isin(fold["train"])]
            validation = data.loc[data["participant"].isin(fold["validation"])]
            require(not training.empty and not validation.empty, "Empty fold.")
            y_train = (training["fatigue"].to_numpy() >= 4).astype(int)
            require(len(np.unique(y_train)) == 2, "Training fold lacks one risk class.")
            started = time.perf_counter()
            predictions = {}
            for entry in config["models"]:
                name = entry["name"]
                cols = groups[entry["features"]]
                x_train = training[cols].replace([np.inf, -np.inf], np.nan)
                x_valid = validation[cols].replace([np.inf, -np.inf], np.nan)
                estimator = models[name]
                estimator.fit(x_train, y_train)
                probability_index = list(estimator.classes_).index(1)
                predictions[name] = estimator.predict_proba(x_valid)[:, probability_index]
            global_median = float(training["fatigue"].median())
            baselines = baseline_predictions(validation, global_median)
            predictions["personal_median_reference"] = np.clip(
                baselines["baseline_personal_median"], 1, 5
            )
            predictions["previous_fatigue_reference"] = np.clip(
                baselines["baseline_previous_fatigue"], 1, 5
            )
            logger.info("Fold %d/7: train=%d valid=%d high_valid=%d | %.1fs",
                        fold["fold"], len(training), len(validation),
                        int((validation["fatigue"] >= 4).sum()), time.perf_counter() - started)
            for strategy in strategies:
                for p, day, fatigue, score in zip(
                    validation["participant"], validation["date"],
                    validation["fatigue"], predictions[strategy]
                ):
                    records.append({
                        "fold": int(fold["fold"]), "participant": str(p),
                        "date": pd.Timestamp(day).strftime("%Y-%m-%d"),
                        "fatigue": float(fatigue),
                        "high": int(fatigue >= 4),
                        "strategy": strategy, "score": float(score),
                    })

        oof = pd.DataFrame(records)
        require(len(oof) == len(data) * len(strategies), "OOF sample count mismatch.")
        require(not oof.duplicated(["strategy", "participant", "date"]).any(),
                "Repeated OOF predictions.")
        require(set(oof["participant"]) == dev, "Unexpected participant in OOF.")
        require(np.isfinite(oof["score"]).all(), "Non-finite model predictions.")
        oof.sort_values(["strategy", "participant", "date"]).to_csv(
            out_dir / "validation_predictions.csv", index=False, float_format="%.10f"
        )
        summary_rows = []
        people_rows = []
        thresholds = e3["preregistered_targets"]
        for strategy in strategies:
            frame = oof.loc[oof["strategy"] == strategy].copy().reset_index(drop=True)
            frame["_draw"] = 0
            pooled = measure(frame, top_fraction=top_fraction)
            detail, within = person_auc(frame)
            detail.insert(0, "strategy", strategy)
            people_rows.append(detail)
            logger.info("Bootstrap %s (%d participant draws) ...", strategy, n_boot)
            intervals = cluster_bootstrap(frame, n_boot, seed, top_fraction)
            target_pass = {
                "roc_auc": pooled["roc_auc"] is not None
                    and pooled["roc_auc"] >= float(thresholds["pooled_ROC_AUC_min"]),
                "average_precision": pooled["average_precision"] is not None
                    and pooled["average_precision"] >= float(thresholds["average_precision_min"]),
                "recall_at_10pct": pooled["recall_at_10pct"] is not None
                    and pooled["recall_at_10pct"] >= float(thresholds["recall_at_10pct_min"]),
                "within_person_mean_roc_auc": within["mean"] is not None
                    and within["mean"] >= float(thresholds["within_person_mean_ROC_AUC_min"]),
            }
            row = {
                "strategy": strategy, "strategy_type":
                    "classifier" if strategy in models else "reference_ranking_score",
                **pooled,
                "within_person_mean_roc_auc": within["mean"],
                "within_person_std_roc_auc": within["std"],
                "eligible_people": within["eligible_n"],
                "ineligible_people": within["ineligible_n"],
                "eligible_ids": ",".join(within["eligible_ids"]),
                "ineligible_ids": ",".join(within["ineligible_ids"]),
                "targets_passed": int(sum(target_pass.values())),
                "all_targets_pass": bool(all(target_pass.values())),
            }
            for metric, result in intervals.items():
                row[f"{metric}_ci95_low"] = result["ci95"][0]
                row[f"{metric}_ci95_high"] = result["ci95"][1]
                row[f"{metric}_bootstrap_valid"] = result["valid_draws"]
                row[f"{metric}_bootstrap_invalid"] = result["invalid_draws"]
            summary_rows.append(row)
            logger.info(
                "OOF %-29s | ROC-AUC=%s AP=%s Recall@10=%s Precision@10=%s FPR@10=%s "
                "within-person AUC=%s (%d eligible) | all targets=%s",
                strategy,
                f'{pooled["roc_auc"]:.4f}' if pooled["roc_auc"] is not None else "undefined",
                f'{pooled["average_precision"]:.4f}' if pooled["average_precision"] is not None else "undefined",
                f'{pooled["recall_at_10pct"]:.3f}' if pooled["recall_at_10pct"] is not None else "undefined",
                f'{pooled["precision_at_10pct"]:.3f}' if pooled["precision_at_10pct"] is not None else "undefined",
                f'{pooled["fpr_at_10pct"]:.3f}' if pooled["fpr_at_10pct"] is not None else "undefined",
                f'{within["mean"]:.4f}' if within["mean"] is not None else "undefined",
                within["eligible_n"], all(target_pass.values())
            )

        pd.DataFrame(summary_rows).to_csv(out_dir / "results.csv", index=False)
        pd.concat(people_rows, ignore_index=True).to_csv(
            out_dir / "participant_metrics.csv", index=False
        )
        report = {
            "experiment": "E3_high_fatigue",
            "status": "development_only",
            "post_e2_candidate_freeze": True,
            "selection_bias_note": config["selection_context"],
            "split_sha256": split_sha,
            "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "validation": "7 disjoint folds: 12 training people, 2 validation people",
            "development_participants": len(dev), "validation_days": len(data),
            "positive_support": int((data["fatigue"] >= 4).sum()),
            "test_evaluated": False,
            "target": "fatigue >= 4, others negative",
            "feature_counts": {"A": len(groups["A"]), "B": len(groups["B"])},
            "model_config": config["models"],
            "bootstrap": {"n": n_boot, "unit": "participant", "seed": seed,
                          "ci": .95, "invalid_draws_reported": True},
            "preregistered_targets": thresholds,
            "threshold_note": "10% of pooled OOF days ranked by score; this is an alert budget, not a fitted classification threshold.",
            "risk_score_note": "Baseline values are fatigue-level ranking scores; ML predict_proba is not validated as calibrated probabilities. No clinical/safety claims.",
            "no_combined_score_or_threshold_tuning": True,
            "results": summary_rows,
        }
        (out_dir / "summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        logger.info("E3 done. Outputs=%s. Sealed test NOT evaluated.", out_dir)
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
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/experiment_high_fatigue"))
    args = parser.parse_args()
    main(args.features, args.split, args.protocol, args.config, args.out_dir)
