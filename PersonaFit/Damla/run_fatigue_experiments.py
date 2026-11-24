"""Development-only fatigue experiment runner. Run from PersonaFit/Damla.

All experimental outputs go under outputs/<experiment_script_stem>/.
Requires an ORIGINAL sealed outputs/split.json with the expected SHA256.
This runner never creates a split, evaluates test subjects, or chooses a winner.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
HASH = "8f053e7dc1ee6a4fd7d34a7a590079a11fe53cc9036328b27a63f2fc1a76437e"
STEPS = [
    ("train_grouped", "02_yorgunluk_tahmini/train_grouped.py", ("validation_metrics.json", "validation_folds.csv", "participant_metrics.csv")),
    ("experiment_cold_start", "06_deneyler/experiment_cold_start.py", ("summary.json", "segment_metrics.csv", "validation_predictions.csv")),
    ("experiment_ablation", "06_deneyler/experiment_ablation.py", ("summary.json", "results.csv", "validation_predictions.csv")),
    ("experiment_high_fatigue", "06_deneyler/experiment_high_fatigue.py", ("summary.json", "results.csv", "validation_predictions.csv")),
    ("experiment_lopo", "06_deneyler/experiment_lopo.py", ("summary.json", "comparison.csv")),
    ("experiment_fatigue_change_errors", "06_deneyler/experiment_fatigue_change_errors.py", ("summary.json", "error_by_change.csv", "error_by_person.csv")),
    ("experiment_high_fatigue_new_13_features", "02_yorgunluk_tahmini/experiment_high_fatigue_new_13_features.py", ("summary.json", "metrics.csv", "oof_predictions.csv")),
    ("experiment_high_fatigue_core9_sleep4", "02_yorgunluk_tahmini/experiment_high_fatigue_core9_sleep4.py", ("summary.json", "metrics.csv", "oof_predictions.csv")),
]


def verify_inputs():
    prepared = ROOT / "outputs/fatigue_daily.csv"
    split = ROOT / "outputs/split.json"
    if not prepared.is_file():
        raise SystemExit("Missing outputs/fatigue_daily.csv: first prepare PMData locally.")
    if not split.is_file():
        raise SystemExit("Missing ORIGINAL outputs/split.json: do not regenerate.")
    digest = hashlib.sha256(split.read_bytes()).hexdigest()
    if digest != HASH:
        raise SystemExit("Frozen split SHA256 differs from experiment_protocol.json. Aborting.")
    return digest


def check(selected):
    results = {}
    for name, _, required in STEPS:
        if name not in selected:
            continue
        folder = ROOT / "outputs" / name
        missing = [f for f in required if not (folder / f).is_file() or (folder / f).stat().st_size == 0]
        state = "OK" if not missing else "MISSING"
        print(f"[{state}] {name} : " + (", ".join(missing) if missing else "all main files present"))
        results[name] = {"status": state, "missing": missing}
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="Comma-separated script names; default: all")
    ap.add_argument("--check", action="store_true", help="Check outputs without running scripts")
    args = ap.parse_args()
    all_names = {s[0] for s in STEPS}
    selected = set(args.only.split(",")) if args.only else all_names
    unknown = selected - all_names
    if unknown:
        ap.error("Unknown script names: " + ", ".join(sorted(unknown)))
    if args.check:
        states = check(selected)
        if any(x["status"] != "OK" for x in states.values()):
            sys.exit(1)
        return
    digest = verify_inputs()
    env = os.environ.copy()
    pypath = [str(ROOT / part) for part in ("01_veri_hazirlama", "02_yorgunluk_tahmini", "06_deneyler")]
    env["PYTHONPATH"] = os.pathsep.join(pypath + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    status_path = ROOT / "outputs/experiment_run_status.json"
    report = {
        "start_time_utc": datetime.now(timezone.utc).isoformat(),
        "split_sha256": digest, "sealed_test_evaluated": False,
        "experiments": {},
    }
    for name, script, needed in STEPS:
        if name not in selected:
            continue
        command = [sys.executable, str(ROOT / script), "--out-dir", str(ROOT / "outputs" / name)]
        if name == "experiment_lopo":
            command += ["--seven-fold-predictions", str(ROOT / "outputs/experiment_high_fatigue/validation_predictions.csv")]
        if name == "experiment_fatigue_change_errors":
            command += ["--predictions", str(ROOT / "outputs/experiment_cold_start/validation_predictions.csv")]
        print("\n>>>", name, flush=True)
        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
        missing = [f for f in needed if not (ROOT / "outputs" / name / f).is_file() or (ROOT / "outputs" / name / f).stat().st_size == 0]
        success = result.returncode == 0 and not missing
        report["experiments"][name] = {"status": "passed" if success else "failed", "exit_code": result.returncode, "missing_outputs": missing}
        status_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if not success:
            raise SystemExit(f"Stopped: {name} failed or results missing. Check {status_path}")
    check(selected)
    print("\nFinished; sealed test was not evaluated.")


if __name__ == "__main__":
    main()
