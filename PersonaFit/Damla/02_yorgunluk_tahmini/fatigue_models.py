"""Model definitions and metrics shared by grouped training/testing."""
from typing import Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Personal (within-person) features. Everything is computed from the same
# participant's *earlier* labeled days only (shift(1)), so no future leakage and
# no information crosses participants.
# ---------------------------------------------------------------------------
PERSONAL_COLUMNS = ["p_n", "p_mean", "p_median", "p_mean14", "p_std", "p_high_rate", "p_low_rate"]
FATIGUE_HISTORY_COLUMNS = [
    "history_wellness_fatigue_lag1",
    "history_wellness_fatigue_last7",
    "history_wellness_fatigue_mean7",
    "history_wellness_fatigue_mean28",
]
_DEVIATION_BASES = [
    "sleep_score_overall_score", "sleep_sleep_minutesAsleep", "sleep_sleep_efficiency",
    "cardio_resting_heart_rate", "srpe_load", "wellness_mood", "wellness_stress",
    "wellness_soreness", "wellness_readiness", "wellness_sleep_quality",
    "cardio_heart_rate_mean", "movement_steps_total", "activity_very_active_minutes",
]
_EXTRA_DEVIATION = [
    "history_srpe_acute_chronic_ratio", "history_srpe_load_sum7",
    "morning_sleep_score_overall_score_delta28", "morning_sleep_sleep_minutesAsleep_delta28",
    "morning_sleep_sleep_efficiency_delta28",
]


def add_personal_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add past-only personal fatigue stats and "vs. own baseline" deviation features.

    Call on the rows that have a fatigue label (any set of whole participants).
    """
    frame = data.sort_values(["participant", "date"]).copy()
    grouped = frame.groupby("participant")["fatigue"]
    frame["p_n"] = grouped.cumcount().astype(float)
    frame["p_mean"] = grouped.transform(lambda s: s.shift().expanding().mean())
    frame["p_median"] = grouped.transform(lambda s: s.shift().expanding().median())
    frame["p_mean14"] = grouped.transform(lambda s: s.shift().rolling(14, min_periods=3).mean())
    frame["p_std"] = grouped.transform(lambda s: s.shift().expanding(min_periods=5).std())
    frame["p_high_rate"] = grouped.transform(lambda s: (s >= 4).astype(float).shift().expanding().mean())
    frame["p_low_rate"] = grouped.transform(lambda s: (s <= 2).astype(float).shift().expanding().mean())
    frame["dow"] = pd.to_datetime(frame["date"]).dt.dayofweek.astype(float)
    frame["fat_trend"] = frame["history_wellness_fatigue_mean7"] - frame["history_wellness_fatigue_mean28"]
    frame["fat_vs_pers"] = frame["history_wellness_fatigue_last7"] - frame["p_mean"]
    for base in _DEVIATION_BASES:
        lag, m7, m28 = (f"history_{base}_{k}" for k in ("lag1", "mean7", "mean28"))
        if lag in frame and m28 in frame:
            frame[f"dv_{base}"] = frame[lag] - frame[m28]
        if m7 in frame and m28 in frame:
            frame[f"dw_{base}"] = frame[m7] - frame[m28]
    return frame


def deviation_columns(data: pd.DataFrame) -> list[str]:
    cols = [c for c in data if c.startswith(("dv_", "dw_"))]
    return cols + [c for c in _EXTRA_DEVIATION if c in data]


def model_feature_columns(data: pd.DataFrame) -> list[str]:
    """All columns a model pipeline may select from (original + personal)."""
    base = [c for c in data if c.startswith(("history_", "morning_"))]
    extra = PERSONAL_COLUMNS + ["dow", "fat_trend", "fat_vs_pers"] + [c for c in data if c.startswith(("dv_", "dw_"))]
    return base + [c for c in extra if c in data]


def shrunk_personal_median(frame: pd.DataFrame, global_median: float, k: float = 5.0) -> np.ndarray:
    """Personal past median, pulled toward the global median when history is short."""
    n = frame["p_n"].to_numpy(dtype=float)
    personal = frame["p_median"].fillna(global_median).to_numpy(dtype=float)
    weight = n / (n + k)
    return weight * personal + (1.0 - weight) * global_median


def baseline_predictions(frame: pd.DataFrame, global_median: float) -> dict[str, np.ndarray]:
    """Baselines shared by training (validation folds) and the sealed test."""
    previous = frame["history_wellness_fatigue_last7"].to_numpy(dtype=float)
    return {
        "baseline_train_median": np.full(len(frame), global_median),
        "baseline_previous_fatigue": np.where(np.isfinite(previous), previous, global_median),
        "baseline_personal_median": shrunk_personal_median(frame, global_median),
    }


def _select(columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer([("cols", "passthrough", columns)], remainder="drop")


def _candidate_models(columns: list[str] | None = None) -> dict[str, Any]:
    """Candidate regressors. `columns` (all available feature names) enables the
    personal-feature candidates; without it only the original set is returned."""
    def imputer() -> SimpleImputer:
        return SimpleImputer(
            strategy="median",
            add_indicator=True,
            keep_empty_features=True,
        )

    candidates = {
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
    if columns is not None:
        # Original candidates keep seeing exactly the original feature set.
        base = [c for c in columns if c.startswith(("history_", "morning_"))]
        candidates = {
            name: Pipeline([("select", _select(base))] + list(model.steps))
            for name, model in candidates.items()
        }

        def plain_imputer() -> SimpleImputer:
            return SimpleImputer(strategy="median", keep_empty_features=True)

        fat_dev = [c for c in PERSONAL_COLUMNS + FATIGUE_HISTORY_COLUMNS + ["fat_trend", "fat_vs_pers", "dow"]
                   + [c for c in columns if c.startswith(("dv_", "dw_"))] + [c for c in _EXTRA_DEVIATION if c in columns]
                   if c in columns]
        core_pers = [c for c in PERSONAL_COLUMNS + FATIGUE_HISTORY_COLUMNS if c in columns]
        # Smooth expected-value score; strongest RMSE/R2, not MAE-optimal.
        candidates["ridge_personal"] = make_pipeline(
            _select(fat_dev), plain_imputer(), StandardScaler(), Ridge(alpha=1000.0)
        )
        # Median-targeting (absolute-error) boosting on a small, person-relative feature set.
        candidates["gbm_abs_personal"] = make_pipeline(
            _select(core_pers), plain_imputer(),
            GradientBoostingRegressor(
                loss="absolute_error", n_estimators=150, learning_rate=0.05, max_depth=2,
                subsample=0.7, min_samples_leaf=30, random_state=0,
            ),
        )
        # E2 feature group A: 7 personal statistics + 4 fatigue-history
        # values + 2 fatigue deviations. Identical feature composition to
        # experiment_ablation.py group A, but distinct candidate names preserve
        # all previously evaluated original models.
        group_a = PERSONAL_COLUMNS + FATIGUE_HISTORY_COLUMNS + ["fat_trend", "fat_vs_pers"]
        if not all(c in columns for c in group_a):
            missing = [c for c in group_a if c not in columns]
            raise ValueError(f"Missing group-A features: {missing}")
        candidates["ridge_personal_A"] = make_pipeline(
            _select(group_a), plain_imputer(), StandardScaler(), Ridge(alpha=1000.0)
        )
        candidates["gbm_abs_personal_A"] = make_pipeline(
            _select(group_a), plain_imputer(),
            GradientBoostingRegressor(
                loss="absolute_error", n_estimators=150, learning_rate=0.05,
                max_depth=2, subsample=0.7, min_samples_leaf=30, random_state=0,
            ),
        )
    return candidates


def _continuous_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    bounded = np.clip(predictions, 1, 5)
    rounded = np.rint(bounded).astype(int)
    truth = y_true.to_numpy(dtype=float)
    rounded_truth = np.rint(truth).astype(int)
    # Average recall across classes that actually appear in y_true; avoids
    # warnings when a small participant-held-out fold lacks a particular class.
    present_classes = np.unique(rounded_truth)
    class_recalls = [float(np.mean(rounded[rounded_truth == label] == label)) for label in present_classes]
    high_true = truth >= 4
    high_pred = bounded >= 3.5
    return {
        "mae": float(mean_absolute_error(truth, bounded)),
        "rmse": float(np.sqrt(mean_squared_error(truth, bounded))),
        "r2": float(r2_score(truth, bounded)),
        "rounded_accuracy": float(accuracy_score(rounded_truth, rounded)),
        "rounded_balanced_accuracy": float(np.mean(class_recalls)),
        "rounded_macro_f1": float(
            f1_score(rounded_truth, rounded, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0)
        ),
        "high_fatigue_recall": float(np.mean(high_pred[high_true])) if high_true.any() else None,
        "high_fatigue_precision": (
            float(np.mean(high_true[high_pred])) if high_pred.any() else None
        ),
        "high_fatigue_support": int(high_true.sum()),
    }
