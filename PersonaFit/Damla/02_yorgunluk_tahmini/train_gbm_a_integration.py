"""Train experimental group-A GBM on ONLY 14 development participants.

This GBM-A was not adopted over personal median in frozen E2. The saved
artifact is local-only, under outputs/, and sealed-test labels are not read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from fatigue_models import _candidate_models, add_personal_features, model_feature_columns
from real_predictor_v2 import GROUP_A
from split_participants import validate_split

FROZEN_SPLIT_HASH = "8f053e7dc1ee6a4fd7d34a7a590079a11fe53cc9036328b27a63f2fc1a76437e"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("outputs/fatigue_daily.csv"))
    ap.add_argument("--split", type=Path, default=Path("outputs/split.json"))
    ap.add_argument("--out", type=Path, default=Path("outputs/gbm_a_integration/fatigue_model.joblib"))
    args = ap.parse_args()

    split_raw = args.split.read_bytes()
    digest = hashlib.sha256(split_raw).hexdigest()
    if digest != FROZEN_SPLIT_HASH:
        raise ValueError("Split SHA256 changed; refusing to fit")
    split = json.loads(split_raw)
    validate_split(split)
    if len(split["development"]) != 14 or len(split["test"]) != 2:
        raise ValueError("Unexpected development/test participant counts")
    data = pd.read_csv(args.features, parse_dates=["date"])
    development = set(split["development"])
    data = data.loc[data["participant"].isin(development) & data["fatigue"].notna()].copy()
    if set(data["participant"]) != development:
        raise ValueError("Missing development participants")
    if data.duplicated(["participant", "date"]).any():
        raise ValueError("Repeated participant/date rows")
    if len(data) != 1512:
        raise ValueError("Expected original 1512 development rows")
    data = add_personal_features(data)
    full_cols = model_feature_columns(data)
    estimator = clone(_candidate_models(full_cols)["gbm_abs_personal_A"])
    declared = list(estimator.steps[0][1].transformers[0][2])
    if declared != GROUP_A:
        raise ValueError("Unexpected 13-feature column order")
    estimator.fit(data[full_cols].replace([np.inf, -np.inf], np.nan), data["fatigue"])
    artifact = {
        "model": estimator,
        "feature_columns": full_cols,
        "selected_model": "gbm_abs_personal_A",
        "prediction_strategy": "gbm_abs_personal_A",
        "global_median": float(data["fatigue"].median()),
        "development": split["development"],
        "test": split["test"],
        "split_sha256": digest,
        "status": "post_E2_exploratory_RL_integration_not_adopted",
        "test_evaluated": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.out)
    print(f"GBM-A trained on 14 development participants, {len(data)} labeled days.")
    print(f"Local artifact: {args.out}; sealed test untouched.")


if __name__ == "__main__":
    main()
