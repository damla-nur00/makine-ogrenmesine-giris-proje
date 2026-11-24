"""Run the sealed participant test once, writing metrics and predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from split_participants import validate_split
from fatigue_models import _continuous_metrics, add_personal_features, baseline_predictions


def evaluate(features_path: Path, split_path: Path, model_path: Path, out_dir: Path) -> None:
    if (out_dir / "test_metrics.json").exists():
        raise FileExistsError("Sealed test has already been recorded; refusing to overwrite")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    validate_split(split)
    artifact = joblib.load(model_path)
    if artifact.get("split_sha256") != hashlib.sha256(split_path.read_bytes()).hexdigest():
        raise ValueError("Sealed split file changed after training")
    if artifact["test"] != split["test"] or artifact["development"] != split["development"]:
        raise ValueError("Model participant split differs from sealed split")
    data = pd.read_csv(features_path, parse_dates=["date"])
    # Personal stats use only each participant's own earlier labeled days.
    labeled = add_personal_features(data.loc[data["fatigue"].notna()].copy())
    heldout = labeled.loc[labeled["participant"].isin(split["test"])].copy()
    if heldout.empty or heldout["participant"].nunique() != 2:
        raise ValueError("Missing labels from one or both sealed test participants")
    heldout = heldout.sort_values(["participant", "date"])
    features = heldout[artifact["feature_columns"]].replace([np.inf, -np.inf], np.nan)
    global_median = artifact["global_median"]
    model_predictions = np.asarray(artifact["model"].predict(features))
    baselines = baseline_predictions(heldout, global_median)
    strategy = artifact["prediction_strategy"]
    predicted = baselines.get(strategy, model_predictions)
    heldout["prediction"] = np.clip(predicted, 1, 5)
    heldout["ml_challenger_prediction"] = np.clip(model_predictions, 1, 5)
    heldout["high_fatigue_actual"] = heldout["fatigue"] >= 4
    heldout["high_fatigue_predicted"] = heldout["prediction"] >= 3.5
    actual_high = int(heldout["high_fatigue_actual"].sum())
    detected_high = int((heldout["high_fatigue_actual"] & heldout["high_fatigue_predicted"]).sum())
    report = {
        "selected_strategy": strategy,
        "selected_metrics": _continuous_metrics(heldout["fatigue"], heldout["prediction"].to_numpy()),
        "ml_challenger_metrics": _continuous_metrics(heldout["fatigue"], heldout["ml_challenger_prediction"].to_numpy()),
        "baseline_metrics": {name: _continuous_metrics(heldout["fatigue"], vals) for name, vals in baselines.items()},
        "test_rows": len(heldout), "test_participants": split["test"],
        "high_fatigue_days": actual_high,
        "high_fatigue_recall": detected_high / actual_high if actual_high else None,
        "comparability_note": "Do not directly compare legacy and new results: different evaluated participants and evaluation protocols.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save predictions before the single-use marker (metrics file).
    heldout[["participant", "date", "fatigue", "prediction", "ml_challenger_prediction", "high_fatigue_actual", "high_fatigue_predicted"]].to_csv(out_dir / "test_predictions.csv", index=False)
    (out_dir / "test_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "test.log").write_text(
        f"UTC: {datetime.now(timezone.utc).isoformat()}\n"
        f"Test people: {', '.join(split['test'])}; rows: {len(heldout)}\n"
        f"Selected strategy: {strategy}; MAE: {report['selected_metrics']['mae']:.6f}\n"
        f"High fatigue recall: {report['high_fatigue_recall']}\n"
        "Final sealed test completed. Do not tune on these results.\n", encoding="utf-8"
    )
    print(f"Final sealed test recorded: {len(heldout)} days, MAE={report['selected_metrics']['mae']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("outputs/fatigue_daily.csv"))
    parser.add_argument("--split", type=Path, default=Path("outputs/split.json"))
    parser.add_argument("--model", type=Path, default=Path("outputs/grouped/fatigue_model.joblib"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/grouped"))
    args = parser.parse_args()
    evaluate(args.features, args.split, args.model, args.out_dir)
