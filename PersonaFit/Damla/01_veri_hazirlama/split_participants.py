"""Freeze a participant-level 2-test / seven (12-train, 2-validation) split."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def create_split(data: pd.DataFrame, *, exclude_test: tuple[str, ...] = ("p15",)) -> dict:
    labeled = data.dropna(subset=["fatigue"])
    summary = labeled.groupby("participant")["fatigue"].agg(
        days="count", high=lambda s: int((s >= 4).sum())
    )
    if len(summary) != 16:
        raise ValueError(f"Expected 16 participants, found {len(summary)}")
    overall_high = float((labeled["fatigue"] >= 4).mean())
    mean_days = float(summary["days"].mean())
    candidates = []
    for pair in itertools.combinations(sorted(summary.index), 2):
        if any(p in exclude_test for p in pair):
            continue
        group = summary.loc[list(pair)]
        if group["high"].sum() < 3 or (group["days"] < 30).any():
            continue
        score = abs(group["days"].mean() - mean_days) / mean_days
        score += abs(group["high"].sum() / group["days"].sum() - overall_high)
        candidates.append((float(score), pair))
    if not candidates:
        raise ValueError("No eligible test pair; inspect participant coverage and revise criteria explicitly")
    _, test_pair = min(candidates)
    development = sorted(set(summary.index) - set(test_pair))
    # Deterministic, exactly two participants per fold; distribute large groups.
    ordered = sorted(development, key=lambda p: (-int(summary.at[p, "days"]), p))
    pairs = [[ordered[i], ordered[-i - 1]] for i in range(7)]
    folds = [
        {"fold": i + 1, "train": sorted(set(development) - set(pair)), "validation": sorted(pair)}
        for i, pair in enumerate(pairs)
    ]
    result = {
        "schema_version": 1,
        "selection": "predeclared coverage/high-fatigue matching; test identities frozen before training",
        "test": list(test_pair),
        "development": development,
        "folds": folds,
    }
    validate_split(result)
    return result


def validate_split(split: dict) -> None:
    test, development = set(split["test"]), set(split["development"])
    if len(test) != 2 or len(development) != 14 or test & development:
        raise ValueError("Split must have 2 distinct test and 14 distinct development participants")
    if len(split["folds"]) != 7:
        raise ValueError("Expected seven folds")
    seen = []
    for fold in split["folds"]:
        training, validation = set(fold["train"]), set(fold["validation"])
        if len(training) != 12 or len(validation) != 2 or training & validation:
            raise ValueError("Every fold must be exactly 12 train / 2 validation")
        if training | validation != development:
            raise ValueError("Fold is inconsistent with development participants")
        seen.extend(validation)
    if sorted(seen) != sorted(development):
        raise ValueError("Each development participant must be validated once")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/split.json"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Split is sealed at {args.out}; will not overwrite it")
    data = pd.read_csv(args.features)
    result = create_split(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Sealed {len(result['test'])} test participants and seven 12/2 folds: {args.out}")


if __name__ == "__main__":
    main()
