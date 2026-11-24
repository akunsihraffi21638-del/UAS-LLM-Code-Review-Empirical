import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt


def summarize_resolution(baseline: Path, output: Path) -> dict:
    df = pd.read_csv(baseline)
    df.columns = [c.strip() for c in df.columns]
    numeric = ["project_age_month", "no_commits", "PR_size(SLOC)", "no_changed_files", "time_to_merge_h", "is_gpt"]
    df = df.dropna(subset=numeric)
    df["is_gpt"] = df["is_gpt"].astype(int)

    gpt = df.loc[df["is_gpt"] == 1, "time_to_merge_h"]
    control = df.loc[df["is_gpt"] == 0, "time_to_merge_h"]
    test = stats.mannwhitneyu(gpt, control, alternative="two-sided")

    covariates = ["project_age_month", "no_commits", "PR_size(SLOC)", "no_changed_files"]
    treated = df[df["is_gpt"] == 1].copy()
    controls = df[df["is_gpt"] == 0].copy()
    scaler = StandardScaler()
    scaler.fit(df[covariates])
    neighbors = NearestNeighbors(n_neighbors=1)
    neighbors.fit(scaler.transform(controls[covariates]))
    distances, indices = neighbors.kneighbors(scaler.transform(treated[covariates]))
    matched_control = controls.iloc[indices[:, 0]].copy()
    matched = pd.concat([treated, matched_control], ignore_index=True)

    x = sm.add_constant(matched[covariates + ["is_gpt"]])
    model = sm.OLS(np.log1p(matched["time_to_merge_h"]), x).fit(cov_type="HC3")

    rng = np.random.default_rng(20241124)
    boot = []
    for _ in range(2000):
        sample = matched.sample(len(matched), replace=True, random_state=rng)
        boot.append(np.median(sample[sample["is_gpt"] == 1]["time_to_merge_h"]) -
                    np.median(sample[sample["is_gpt"] == 0]["time_to_merge_h"]))

    result = {
        "n_total": int(len(df)),
        "n_gpt": int(len(gpt)),
        "n_control": int(len(control)),
        "median_gpt_hours": float(np.median(gpt)),
        "median_control_hours": float(np.median(control)),
        "median_difference_gpt_minus_control": float(np.median(gpt) - np.median(control)),
        "mann_whitney_U": float(test.statistic),
        "mann_whitney_p": float(test.pvalue),
        "bootstrapped_median_difference_ci95": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "matched_pairs": int(len(treated)),
        "regression_log1p_is_gpt_coefficient": float(model.params["is_gpt"]),
        "regression_log1p_is_gpt_p": float(model.pvalues["is_gpt"]),
        "model_r_squared": float(model.rsquared),
    }
    (output / "rq1_resolution.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([control, gpt], tick_labels=["Non-assisted", "GPT-assisted"], showfliers=False)
    ax.set_ylabel("Time to merge (hours)")
    ax.set_title("Pull Request Resolution Time")
    fig.tight_layout()
    fig.savefig(output / "rq1_resolution.png", dpi=180)
    plt.close(fig)
    return result


def summarize_phases(baseline: Path, output: Path) -> dict:
    df = pd.read_csv(baseline)
    df = df.rename(columns={
        "duration_gpt": "duration_gpt",
        "duration_no_gpt": "duration_control",
    })
    phase_result = {}
    for phase in sorted(df["phase"].dropna().unique()):
        subset = df[df["phase"] == phase].dropna(subset=["duration_gpt", "duration_control"])
        gpt = subset["duration_gpt"]
        control = subset["duration_control"]
        if len(gpt) and len(control):
            test = stats.mannwhitneyu(gpt, control, alternative="two-sided")
            phase_result[phase] = {
                "n_gpt": int(len(gpt)),
                "n_control": int(len(control)),
                "median_gpt_hours": float(np.median(gpt)),
                "median_control_hours": float(np.median(control)),
                "median_difference_gpt_minus_control": float(np.median(gpt) - np.median(control)),
                "mann_whitney_p": float(test.pvalue),
            }
    (output / "rq2_phases.json").write_text(json.dumps(phase_result, indent=2), encoding="utf-8")

    names = list(phase_result)
    gpt_medians = [phase_result[n]["median_gpt_hours"] for n in names]
    control_medians = [phase_result[n]["median_control_hours"] for n in names]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(names))
    width = 0.36
    ax.bar(x - width / 2, control_medians, width, label="Non-assisted")
    ax.bar(x + width / 2, gpt_medians, width, label="GPT-assisted")
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylabel("Median duration (hours)")
    ax.set_title("PR Phase Durations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "rq2_phases.png", dpi=180)
    plt.close(fig)
    return phase_result


def summarize_usage(baseline: Path, output: Path) -> dict:
    df = pd.read_csv(baseline)
    event_counts = df.groupby(["is_gpt_for", "event"], dropna=False).size()
    result = {
        "note": "Event-frequency proxy; the original paper requires manual coding of task purposes",
        "event_counts": {
            f"{str(label)}::{str(event)}": int(count)
            for (label, event), count in event_counts.items()
        },
        "n_events": int(len(df)),
    }
    (output / "rq3_usage.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", default=Path("outputs"), type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rq1 = summarize_resolution(
        args.baseline / "R" / "RQ1" / "at_resolution.csv", args.output
    )
    rq2 = summarize_phases(
        args.baseline / "R" / "RQ2" / "distance.csv", args.output
    )
    rq3 = summarize_usage(
        args.baseline / "scripts" / "data" / "events_gpt.csv", args.output
    )
    summary = {"RQ1": rq1, "RQ2": rq2, "RQ3": rq3}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
