"""Build, evaluate, and train a leakage-aware daily PMData fatigue model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pmdata_pipeline import build_daily_dataset


def _candidate_models() -> dict[str, Any]:
    def imputer() -> SimpleImputer:
        return SimpleImputer(
            strategy="median",
            add_indicator=True,
            keep_empty_features=True,
        )

    return {
        "ridge_alpha_1": make_pipeline(imputer(), StandardScaler(), Ridge(alpha=1.0)),
        "ridge_alpha_10": make_pipeline(imputer(), StandardScaler(), Ridge(alpha=10.0)),
        "ridge_alpha_50": make_pipeline(imputer(), StandardScaler(), Ridge(alpha=50.0)),
        "random_forest": make_pipeline(
            imputer(),
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=8,
                max_features=0.8,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        "extra_trees": make_pipeline(
            imputer(),
            ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=8,
                max_features=0.8,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        "gradient_boosting": make_pipeline(
            imputer(),
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.03,
                max_depth=2,
                loss="huber",
                random_state=42,
            ),
        ),
        "hist_gradient_boosting": make_pipeline(
            imputer(),
            HistGradientBoostingRegressor(
                max_iter=150,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=42,
            ),
        ),
    }


def _continuous_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    bounded = np.clip(predictions, 1, 5)
    rounded = np.rint(bounded).astype(int)
    truth = y_true.to_numpy(dtype=float)
    rounded_truth = np.rint(truth).astype(int)
    return {
        "mae": float(mean_absolute_error(truth, bounded)),
        "rmse": float(np.sqrt(mean_squared_error(truth, bounded))),
        "r2": float(r2_score(truth, bounded)),
        "rounded_accuracy": float(accuracy_score(rounded_truth, rounded)),
        "rounded_balanced_accuracy": float(
            balanced_accuracy_score(rounded_truth, rounded)
        ),
        "rounded_macro_f1": float(
            f1_score(rounded_truth, rounded, average="macro", zero_division=0)
        ),
    }


def _temporal_candidate_scores(
    features: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,
    dates: pd.Series,
    candidates: dict[str, Any],
    folds: int,
) -> dict[str, dict[str, float]]:
    unique_dates = np.sort(dates.unique())
    split_count = min(folds, max(len(unique_dates) - 1, 0))
    if split_count < 2:
        raise ValueError("At least three unique dates are needed for rolling validation")
    first_validation_index = max(int(len(unique_dates) * 0.5), 1)
    validation_dates = np.array_split(unique_dates[first_validation_index:], split_count)
    validation_dates = [chunk for chunk in validation_dates if len(chunk)]
    if len(validation_dates) < 2:
        raise ValueError("Not enough dates for rolling validation")

    scores: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {
        name: np.full(len(labels), np.nan) for name in candidates
    }
    predictions.update(
        {
            "baseline_train_median": np.full(len(labels), np.nan),
            "baseline_previous_fatigue": np.full(len(labels), np.nan),
            "baseline_participant_median": np.full(len(labels), np.nan),
        }
    )
    date_values = dates.to_numpy()
    previous_fatigue = features["history_wellness_fatigue_last7"].to_numpy(dtype=float)

    for name, candidate in candidates.items():
        for date_chunk in validation_dates:
            train_indices = np.flatnonzero(date_values < date_chunk[0])
            test_indices = np.flatnonzero(np.isin(date_values, date_chunk))
            if len(train_indices) < 100 or not len(test_indices):
                continue
            model = clone(candidate)
            model.fit(features.iloc[train_indices], labels.iloc[train_indices])
            predictions[name][test_indices] = model.predict(features.iloc[test_indices])

    for date_chunk in validation_dates:
        train_indices = np.flatnonzero(date_values < date_chunk[0])
        test_indices = np.flatnonzero(np.isin(date_values, date_chunk))
        if len(train_indices) < 100 or not len(test_indices):
            continue
        fold_median = float(labels.iloc[train_indices].median())
        predictions["baseline_train_median"][test_indices] = fold_median
        predictions["baseline_previous_fatigue"][test_indices] = np.where(
            np.isfinite(previous_fatigue[test_indices]),
            previous_fatigue[test_indices],
            fold_median,
        )
        personal_medians = labels.iloc[train_indices].groupby(
            groups.iloc[train_indices]
        ).median()
        predictions["baseline_participant_median"][test_indices] = (
            groups.iloc[test_indices].map(personal_medians).fillna(fold_median)
        )

    scored_rows = np.isfinite(predictions["baseline_train_median"])
    if not scored_rows.any():
        raise ValueError("Rolling validation produced no predictions")
    for name, prediction in predictions.items():
        scores[name] = _continuous_metrics(
            labels.iloc[np.flatnonzero(scored_rows)],
            prediction[scored_rows],
        )
    return scores


def _grouped_model_score(
    features: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,
    model: Any,
    folds: int,
) -> dict[str, dict[str, float]]:
    split_count = min(folds, int(groups.nunique()))
    if split_count < 2:
        raise ValueError("At least two participants are required for grouped validation")
    predictions = np.full(len(labels), np.nan)
    median_baseline = np.full(len(labels), np.nan)
    previous_baseline = np.full(len(labels), np.nan)
    previous_fatigue = features["history_wellness_fatigue_last7"].to_numpy(dtype=float)
    splitter = GroupKFold(n_splits=split_count)
    for train_indices, test_indices in splitter.split(features, labels, groups):
        fold_median = float(labels.iloc[train_indices].median())
        fold_model = clone(model)
        fold_model.fit(features.iloc[train_indices], labels.iloc[train_indices])
        predictions[test_indices] = fold_model.predict(features.iloc[test_indices])
        median_baseline[test_indices] = fold_median
        previous_baseline[test_indices] = np.where(
            np.isfinite(previous_fatigue[test_indices]),
            previous_fatigue[test_indices],
            fold_median,
        )
    return {
        "selected_model": _continuous_metrics(labels, predictions),
        "baseline_train_median": _continuous_metrics(labels, median_baseline),
        "baseline_previous_fatigue": _continuous_metrics(labels, previous_baseline),
    }


def train(
    data_root: Path,
    model_path: Path,
    metrics_path: Path | None,
    features_path: Path | None,
    folds: int,
    include_reporting: bool,
) -> None:
    dataset, audit = build_daily_dataset(data_root, include_reporting=include_reporting)
    labeled = dataset.loc[dataset["fatigue"].notna()].copy()
    if labeled.empty:
        raise ValueError(f"No valid fatigue labels found under {data_root}")

    feature_columns = [
        column
        for column in dataset.columns
        if column.startswith(("history_", "morning_"))
    ]
    features = labeled[feature_columns].replace([np.inf, -np.inf], np.nan)
    labels = labeled["fatigue"].astype(float)
    groups = labeled["participant"]
    participant_count = int(groups.nunique())
    if participant_count < 2:
        raise ValueError("At least two participants are required for grouped validation")

    unique_dates = np.sort(labeled["date"].unique())
    if len(unique_dates) < 10:
        raise ValueError("At least ten unique dates are needed for a chronological holdout")
    cutoff_index = min(max(int(len(unique_dates) * 0.8), 1), len(unique_dates) - 1)
    cutoff = pd.Timestamp(unique_dates[cutoff_index])
    development_mask = labeled["date"] < cutoff
    holdout_mask = ~development_mask
    development = labeled.loc[development_mask]
    temporal_holdout = labeled.loc[holdout_mask]
    if development.empty or temporal_holdout.empty:
        raise ValueError("Unable to create a non-empty chronological holdout")

    development_features = features.loc[development.index]
    development_labels = labels.loc[development.index]
    development_groups = groups.loc[development.index]
    development_dates = development["date"]
    candidate_scores = _temporal_candidate_scores(
        development_features,
        development_labels,
        development_groups,
        development_dates,
        _candidate_models(),
        folds,
    )
    templates = _candidate_models()
    best_model_name = min(
        templates, key=lambda name: candidate_scores[name]["mae"]
    )
    selected_strategy = min(
        candidate_scores, key=lambda name: candidate_scores[name]["mae"]
    )
    selected_template = templates[best_model_name]

    temporal_model = clone(selected_template)
    temporal_model.fit(development_features, development_labels)
    temporal_model_predictions = np.clip(
        temporal_model.predict(features.loc[holdout_mask]), 1, 5
    )
    holdout_labels = labels.loc[holdout_mask]
    median_prediction = np.full(len(holdout_labels), development_labels.median())
    previous_fatigue = features.loc[holdout_mask][
        "history_wellness_fatigue_last7"
    ].to_numpy(dtype=float)
    previous_fatigue = np.where(
        np.isfinite(previous_fatigue),
        previous_fatigue,
        development_labels.median(),
    )
    personal_medians = development_labels.groupby(
        development_groups
    ).median()
    participant_median_prediction = (
        groups.loc[holdout_mask]
        .map(personal_medians)
        .fillna(development_labels.median())
        .to_numpy(dtype=float)
    )
    temporal_baselines = {
        "median": _continuous_metrics(holdout_labels, median_prediction),
        "previous_fatigue": _continuous_metrics(holdout_labels, previous_fatigue),
        "participant_median": _continuous_metrics(
            holdout_labels, participant_median_prediction
        ),
    }
    temporal_metrics = {
        "selected_strategy": _continuous_metrics(
            holdout_labels,
            {
                "baseline_train_median": median_prediction,
                "baseline_previous_fatigue": previous_fatigue,
                "baseline_participant_median": participant_median_prediction,
            }.get(selected_strategy, temporal_model_predictions),
        ),
        "best_model_challenger": _continuous_metrics(
            holdout_labels, temporal_model_predictions
        ),
    }

    grouped_metrics = _grouped_model_score(
        development_features,
        development_labels,
        development_groups,
        selected_template,
        folds,
    )

    print(
        f"Daily rows: {len(dataset)} | labeled: {len(labeled)} | "
        f"participants: {participant_count} | features: {len(feature_columns)}"
    )
    print(
        f"Excluded invalid fatigue reports: {audit['invalid_fatigue_reports']} | "
        f"invalid 1-5 wellness values: {audit['invalid_wellness_values']} | "
        f"duplicate report days: {audit['duplicate_report_days']}"
    )
    print(f"Chronological holdout: {cutoff.date()} onward ({len(temporal_holdout)} rows)")
    print(f"Best ML challenger on rolling validation: {best_model_name}")
    print(f"Selected prediction strategy: {selected_strategy}")
    if selected_strategy.startswith("baseline_"):
        print(
            "No ML candidate beats the selected baseline on rolling validation; "
            "the artifact will use that baseline for predictions."
        )
    print("Development rolling-origin MAE:")
    for name, score in sorted(
        candidate_scores.items(), key=lambda item: item[1]["mae"]
    ):
        print(f"  {name}: {score['mae']:.3f}")
    print(
        "Participant-held-out development metrics: "
        f"{json.dumps(grouped_metrics, sort_keys=True)}"
    )
    print(
        "Chronological holdout metrics: "
        f"{json.dumps(temporal_metrics, sort_keys=True)}"
    )
    print(
        "Temporal baselines: "
        f"{json.dumps(temporal_baselines, sort_keys=True)}"
    )

    if features_path is not None:
        features_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(features_path, index=False)
        print(f"Saved daily feature table: {features_path}")

    final_model = clone(selected_template)
    final_model.fit(features, labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": final_model,
        "feature_columns": feature_columns,
        "selected_model": best_model_name,
        "prediction_strategy": selected_strategy,
        "global_median": float(labels.median()),
        "participant_medians": {
            str(participant): float(median)
            for participant, median in labels.groupby(groups).median().items()
        },
        "target": "daily_fatigue_1_to_5",
        "include_reporting": include_reporting,
        "training_rows": int(len(labeled)),
        "training_participants": participant_count,
        "temporal_holdout_cutoff": str(cutoff.date()),
    }
    joblib.dump(artifact, model_path)
    print(f"Saved model: {model_path}")

    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "dataset": {key: value for key, value in audit.items() if key != "participants_detail"},
            "best_model_challenger": best_model_name,
            "selected_prediction_strategy": selected_strategy,
            "development_rolling_origin": candidate_scores,
            "development_participant_held_out": grouped_metrics,
            "temporal_holdout_cutoff": str(cutoff.date()),
            "temporal_holdout": temporal_metrics,
            "temporal_baselines": temporal_baselines,
            "include_reporting": include_reporting,
        }
        metrics_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Saved evaluation report: {metrics_path}")


def predict_latest(
    data_root: Path,
    model_path: Path,
    participant: str,
    as_of: str | None,
    features_path: Path | None,
) -> None:
    artifact = joblib.load(model_path)
    if features_path is None:
        dataset, _ = build_daily_dataset(
            data_root, include_reporting=bool(artifact["include_reporting"])
        )
    else:
        dataset = pd.read_csv(features_path, parse_dates=["date"])
    rows = dataset.loc[dataset["participant"] == participant].copy()
    if as_of is not None:
        requested_day = pd.Timestamp(as_of)
        rows = rows.loc[rows["date"] == requested_day]
    else:
        rows = rows.loc[rows["date"] == rows["date"].max()]
    if rows.empty:
        raise ValueError(f"No daily row found for {participant} at the requested date")
    missing_columns = sorted(
        set(artifact["feature_columns"]) - set(dataset.columns)
    )
    if missing_columns:
        raise ValueError(
            "Daily feature table does not match the model; missing columns: "
            + ", ".join(missing_columns)
        )
    feature_frame = rows.reindex(columns=artifact["feature_columns"])
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    strategy = artifact["prediction_strategy"]
    global_median = float(artifact["global_median"])
    row = rows.iloc[0]
    if strategy == "baseline_train_median":
        prediction = global_median
    elif strategy == "baseline_previous_fatigue":
        value = row.get("history_wellness_fatigue_last7", np.nan)
        prediction = float(value) if pd.notna(value) else global_median
    elif strategy == "baseline_participant_median":
        history = dataset.loc[
            (dataset["participant"] == participant)
            & (dataset["date"] < row["date"]),
            "fatigue",
        ].dropna()
        prediction = float(history.median()) if not history.empty else global_median
    else:
        prediction = float(artifact["model"].predict(feature_frame)[0])
    print(
        f"{participant} {rows.iloc[0]['date'].date()}: "
        f"predicted fatigue {np.clip(prediction, 1, 5):.2f}/5 "
        f"(strategy: {strategy})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing participant folders such as p01 and p02",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("outputs/fatigue_model.joblib"),
        help="Path for the trained model artifact",
    )
    parser.add_argument(
        "--features-out",
        type=Path,
        help="Optional path for the daily person-day feature table CSV",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        help="Optional path for the evaluation metrics JSON",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--include-reporting",
        action="store_true",
        help="Use previous-day Google Forms features for an explicit ablation",
    )
    parser.add_argument(
        "--predict-latest",
        metavar="PARTICIPANT",
        help="Predict the latest reported day using an existing model artifact",
    )
    parser.add_argument("--model-in", type=Path, help="Existing artifact for prediction")
    parser.add_argument(
        "--features-in",
        type=Path,
        help="Previously generated daily feature CSV; avoids rereading large sensor JSON",
    )
    parser.add_argument(
        "--as-of",
        help="Optional YYYY-MM-DD date for --predict-latest",
    )
    args = parser.parse_args()

    if args.predict_latest:
        if args.model_in is None:
            parser.error("--predict-latest requires --model-in")
        predict_latest(
            args.data_root,
            args.model_in,
            args.predict_latest,
            args.as_of,
            args.features_in,
        )
        return
    if args.as_of or args.features_in:
        parser.error("--as-of and --features-in are only valid with --predict-latest")

    train(
        args.data_root,
        args.model_out,
        args.metrics_out,
        args.features_out,
        args.folds,
        args.include_reporting,
    )


if __name__ == "__main__":
    main()
