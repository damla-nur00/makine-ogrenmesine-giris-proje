"""Compare GBM-A/median/current report on IDENTICAL held-fixed simulator trajectories.

This is a diagnostic of synthetic-data prediction, NOT a real PMData
held-out MAE result, nor an off-policy RL evaluation. The rule policy chooses
actions using only report/sleep/injury. Both predictors see the exact same
past-only reported-fatigue history and true latent F is used only for scoring.

Example:
python evaluate_fixed_fatigue_v1.py --split outputs/split.json \
  --gbm-artifact outputs/gbm_a_integration/fatigue_model.joblib \
  --median-artifact outputs/grouped_policy_v2/fatigue_model.joblib \
  --features outputs/fatigue_daily.csv --seeds 3 --episodes 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from real_predictor_v2 import ValidatedPredictor
from sim_env_integrated import FatigueEnv, default_profiles, profiles_from_data
from run_rl_ablation_v2 import rule_policy


def ci95(x):
    a = np.asarray(x, dtype=float)
    mean = float(a.mean())
    half = float(stats.t.ppf(0.975, len(a)-1) * a.std(ddof=1) / np.sqrt(len(a))) if len(a)>1 else np.nan
    return mean, half


def run(seed, episodes, profiles, median, gbm):
    rows = []
    env = FatigueEnv(profiles, compute_pred=False)
    for ep in range(episodes):
        # Same trajectories for all predictors: neither controls actions.
        o, _ = env.reset(seed=2_000_000 + seed * 10_000 + ep,
                         options={"profile": ep % len(profiles)})
        for day in range(env.horizon):
            truth = env.true_fatigue_for_oracle()
            history = env.reports
            pred_gbm = gbm(history)
            pred_median = median(history)
            rows.append({
                "seed": seed, "episode": ep, "day": day, "profile": env.p.name,
                "abs_error_report": abs(float(o[0]) - truth),
                "abs_error_gbm": abs(pred_gbm - truth),
                "abs_error_median": abs(pred_median - truth),
                "sq_error_report": (float(o[0]) - truth)**2,
                "sq_error_gbm": (pred_gbm - truth)**2,
                "sq_error_median": (pred_median - truth)**2,
                "true_high": int(truth >= 4.0),
            })
            a = rule_policy(o)
            o, _, _, _, _ = env.step(a)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--gbm-artifact", type=Path, required=True)
    p.add_argument("--median-artifact", type=Path, required=True)
    p.add_argument("--features", type=Path)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/fixed_fatigue_v1"))
    args = p.parse_args()
    if args.seeds < 2 or args.episodes < 3 or args.episodes % 3:
        p.error("Use >=2 seeds and an episode count divisible by 3 for balanced profiles")
    if (args.out_dir / "summary.csv").exists():
        raise FileExistsError("Refusing to overwrite existing fixed-trajectory results")
    gbm = ValidatedPredictor(args.gbm_artifact, args.split, strategy="model")
    median = ValidatedPredictor(args.median_artifact, args.split, strategy="median")
    if abs(gbm.global_median - median.global_median) > 1e-9:
        raise ValueError("Different development training medians; check model provenance")
    profiles = profiles_from_data(args.features, args.split) if args.features else default_profiles()
    frames = [run(seed, args.episodes, profiles, median, gbm) for seed in range(args.seeds)]
    data = pd.concat(frames, ignore_index=True)
    byseed = data.groupby("seed", sort=True).mean(numeric_only=True).reset_index()
    comparisons = [
        ("GBM-A vs report", "abs_error_gbm", "abs_error_report"),
        ("GBM-A vs personal median", "abs_error_gbm", "abs_error_median"),
        ("personal median vs report", "abs_error_median", "abs_error_report"),
    ]
    summary = []
    for title, a, b in comparisons:
        paired = byseed[a] - byseed[b]
        mean, half = ci95(paired)
        summary.append({
            "comparison": title,
            "delta_MAE": mean,
            "ci95_low": mean - half,
            "ci95_high": mean + half,
            "n_seeds": args.seeds,
            "note": "same fixed rule-policy synthetic trajectories; negative is better",
        })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    byseed.to_csv(args.out_dir / "results_per_seed.csv", index=False)
    pd.DataFrame(summary).to_csv(args.out_dir / "summary.csv", index=False)
    byprofile = data.groupby("profile", sort=True)[
        ["abs_error_gbm","abs_error_median","abs_error_report","true_high"]
    ].mean()
    byprofile.to_csv(args.out_dir / "profile_summary.csv")
    print("Synthetic fixed-rule-policy MAE (shared trajectories; NOT real held-out PMData):")
    print(pd.DataFrame(summary).to_string(index=False))
    print("By profile:")
    print(byprofile.to_string())


if __name__ == "__main__":
    main()
