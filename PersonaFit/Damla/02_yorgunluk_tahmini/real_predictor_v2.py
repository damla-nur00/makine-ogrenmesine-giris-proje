"""Development-only PersonaFit group-A GBM and personal-median RL bridge.

All 13 features use ONLY earlier observed reports from consecutive synthetic
days. PMData history windows are calendar-day windows: this parity applies
only when simulated reporting has NO missing calendar days.
Never load untrusted joblib/pickle files: deserializing them executes code.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from fatigue_models import (
    FATIGUE_HISTORY_COLUMNS, PERSONAL_COLUMNS, shrunk_personal_median,
)

GROUP_A = PERSONAL_COLUMNS + FATIGUE_HISTORY_COLUMNS + ["fat_trend", "fat_vs_pers"]
assert len(GROUP_A) == 13


def features_from_reports(reports) -> dict[str, float]:
    """Generate features of simulated day t using only reports for days [0,t)."""
    r = np.asarray(reports, dtype=float)
    if r.ndim != 1 or not np.isfinite(r).all() or np.any((r < 1) | (r > 5)):
        raise ValueError("Expected one-dimensional finite 1..5 earlier daily reports")
    n = len(r)
    nan = float("nan")
    if not n:
        return {name: 0.0 if name == "p_n" else nan for name in GROUP_A}
    last = float(r[-1])
    m7 = float(r[-7:].mean())
    m28 = float(r[-28:].mean())
    pmean = float(r.mean())
    return {
        "p_n": float(n),
        "p_mean": pmean,
        "p_median": float(np.median(r)),
        "p_mean14": float(r[-14:].mean()) if n >= 3 else nan,
        "p_std": float(r.std(ddof=1)) if n >= 5 else nan,
        "p_high_rate": float((r >= 4).mean()),
        "p_low_rate": float((r <= 2).mean()),
        "history_wellness_fatigue_lag1": last,
        "history_wellness_fatigue_last7": last,
        "history_wellness_fatigue_mean7": m7,
        "history_wellness_fatigue_mean28": m28,
        "fat_trend": m7 - m28,
        "fat_vs_pers": last - pmean,
    }


class ValidatedPredictor:
    """Synthetic-history level predictor: median or frozen 13-feature GBM-A.

    GBM-A is exploratory for RL: it did NOT qualify for replacing the personal
    median under E2's predeclared level-prediction selection criterion.
    """

    def __init__(self, artifact_path: Path | str, split_path: Path | str,
                 strategy: str = "median"):
        if strategy not in ("median", "model"):
            raise ValueError("strategy must be median or model")
        self.strategy = strategy
        split_raw = Path(split_path).read_bytes()
        split = json.loads(split_raw)
        self.split_sha256 = hashlib.sha256(split_raw).hexdigest()
        expected_sha = "8f053e7dc1ee6a4fd7d34a7a590079a11fe53cc9036328b27a63f2fc1a76437e"
        if self.split_sha256 != expected_sha:
            raise ValueError("Participant split differs from frozen SHA256")
        if set(split["development"]) & set(split["test"]):
            raise ValueError("Sealed test participants overlap with development")
        if len(split["development"]) != 14 or len(split["test"]) != 2:
            raise ValueError("Expected 14 development and 2 sealed test participants")
        self.artifact_path = str(artifact_path)
        artifact = joblib.load(artifact_path)
        if artifact.get("split_sha256") != self.split_sha256:
            raise ValueError("Artifact split SHA256 differs from locked split")
        if set(artifact.get("development", [])) != set(split["development"]):
            raise ValueError("Artifact development participant identities differ")
        if set(artifact.get("test", [])) != set(split["test"]):
            raise ValueError("Artifact sealed-test identities differ")
        self.global_median = float(artifact["global_median"])
        if not np.isfinite(self.global_median) or not 1 <= self.global_median <= 5:
            raise ValueError("Invalid training-only global median")
        self.model = None
        if strategy == "model":
            if artifact.get("selected_model") != "gbm_abs_personal_A":
                raise ValueError("ML integration strictly requires GBM-A")
            if artifact.get("prediction_strategy") != "gbm_abs_personal_A":
                raise ValueError("Artifact strategy is not the GBM-A experimental model")
            model = artifact.get("model")
            if not isinstance(model, Pipeline):
                raise ValueError("Expected sklearn Pipeline")
            selector = model.steps[0][1]
            if not hasattr(selector, "transformers_"):
                raise ValueError("ML selector is not fitted")
            actual = list(selector.transformers_[0][2])
            if actual != GROUP_A:
                raise ValueError(f"Model column order differs from frozen group A: {actual}")
            self.model = model

    def __call__(self, reports) -> float:
        feats = features_from_reports(reports)
        if self.strategy == "median":
            row = pd.DataFrame([{"p_n": feats["p_n"], "p_median": feats["p_median"]}])
            out = float(shrunk_personal_median(row, self.global_median)[0])
        else:
            x = pd.DataFrame([feats], columns=GROUP_A)
            out = float(self.model.predict(x)[0])
        if not np.isfinite(out):
            raise ValueError("Predictor returned nonfinite output")
        return float(np.clip(out, 1.0, 5.0))
