"""Prepare daily PMData features without training a model or inspecting test scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pmdata_pipeline import build_daily_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--features-out", type=Path, default=Path("outputs/fatigue_daily.csv"))
    parser.add_argument("--audit-out", type=Path, default=Path("outputs/data_audit.json"))
    parser.add_argument("--include-reporting", action="store_true")
    args = parser.parse_args()
    data, audit = build_daily_dataset(args.data_root, include_reporting=args.include_reporting)
    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.features_out, index=False)
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Prepared {len(data)} participant-days with {audit['feature_count']} features")
