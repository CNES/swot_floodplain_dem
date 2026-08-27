"""
Relative PIXC Quality Analysis:
    Detection of Anomalous Pixel Cloud Products Using Quality Flags, Sigma0 Statistics and Robust MAD Scores
"""

import os
from glob import glob
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# Robust z-score using median and MAD

def robust_zscore(x, max_z=5.0):
    x = np.asarray(x, dtype=float)
    result = np.full_like(x, np.nan)
    valid = np.isfinite(x)

    if not np.any(valid):
        return result

    values = x[valid]
    median = np.median(values)
    mad = np.median(np.abs(values - median))

    if mad > 1e-10:
        z = 0.6745 * (values - median) / mad
    else:
        z = np.zeros_like(values)
        different = values != median

        if np.any(different):
            z[different] = np.sign(values[different] - median) * max_z

    result[valid] = np.clip(z, -max_z, max_z)
    return result



# Quality flags

BITS_DEF = {
    "suspect": range(0, 16),
    "degraded": range(16, 25),
    "bad": range(25, 32)
}


def has_any_bit(value, bits):
    return any(int(value) & (1 << b) for b in bits)


def evaluate_flag(values):
    values = np.asarray(values)

    if np.issubdtype(values.dtype, np.floating):
        values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "ok": np.nan,
            "suspect": np.nan,
            "degraded": np.nan,
            "bad": np.nan
        }

    values = values.astype(np.int64)

    return {
        "ok": np.mean(values == 0),
        "suspect": np.mean([has_any_bit(v, BITS_DEF["suspect"]) for v in values]),
        "degraded": np.mean([has_any_bit(v, BITS_DEF["degraded"]) for v in values]),
        "bad": np.mean([has_any_bit(v, BITS_DEF["bad"]) for v in values])
    }


# PIXC analysis

def pixc_analysis(pixc_file):
    name = os.path.basename(pixc_file)
    parts = name.split("_")

    metrics = {
        "pixc_file": name,
        "cycle": parts[4] if len(parts) > 4 else "unknown",
        "pass": parts[5] if len(parts) > 5 else "unknown",
        "tile": parts[6][:-1] if len(parts) > 6 else "unknown",
        "side": parts[6][-1] if len(parts) > 6 else "unknown"
    }

    try:
        with xr.open_dataset(pixc_file, group="pixel_cloud", decode_times=False) as ds:
            df = ds[
                [
                    "classification",
                    "classification_qual",
                    "geolocation_qual",
                    "sig0_qual",
                    "sig0"
                ]
            ].to_dataframe()

        df = df[df["classification"] > 1]
        metrics["n_pixels"] = len(df)

        # Quality flag statistics
        for var in ["classification_qual", "geolocation_qual", "sig0_qual"]:
            result = evaluate_flag(df[var].values)

            for key, value in result.items():
                metrics[f"{var}_{key}"] = value

        # Sigma0 indicator
        sig0 = df["sig0"].to_numpy(dtype=float)
        sig0 = sig0[np.isfinite(sig0)]

        metrics["sig0_p05"] = np.percentile(sig0, 5) if len(sig0) > 0 else np.nan
        metrics["error"] = ""

    except Exception as exc:
        print(f"[ERROR] {pixc_file}: {exc}")

        metrics["n_pixels"] = 0
        metrics["sig0_p05"] = np.nan
        metrics["error"] = str(exc)

        for var in ["classification_qual", "geolocation_qual", "sig0_qual"]:
            for key in ["ok", "suspect", "degraded", "bad"]:
                metrics[f"{var}_{key}"] = np.nan

    return metrics


# Relative PIXC quality

def compute_quality(df):
    df = df.copy()

    indicators = {
        "geolocation_suspect_z": "geolocation_qual_suspect",
        "geolocation_bad_z": "geolocation_qual_bad",
        "sig0_suspect_z": "sig0_qual_suspect",
        "sig0_bad_z": "sig0_qual_bad",
        "sig0_p05_z": "sig0_p05"
    }

    # Compute robust relative scores
    for z_col, column in indicators.items():
        df[z_col] = robust_zscore(df[column].to_numpy(dtype=float))

    df["quality_score"] = (
        0.50 * df["sig0_p05_z"]
        - 0.10 * df["geolocation_suspect_z"]
        - 0.10 * df["sig0_suspect_z"]
        - 0.15 * df["geolocation_bad_z"]
        - 0.15 * df["sig0_bad_z"]
    )

    # Relative classification
    df["suspect"] = df["quality_score"] < -0.75
    df["very_bad"] = df["quality_score"] < -1.25
    df["extremely_bad"] = df["quality_score"] < -1.75

    return df


# Labels

def get_labels(df):
    return (
        df["cycle"].astype(str) + "_" +
        df["pass"].astype(str) + "_" +
        df["tile"].astype(str) +
        df["side"].astype(str)
    )


# Quality score plot

def plot_quality(df, outdir):
    x = np.arange(len(df))
    labels = get_labels(df)

    colors = np.where(
        df["extremely_bad"],
        "#A50026",
        np.where(
            df["very_bad"],
            "#D55E00",
            np.where(df["suspect"], "#E69F00", "#0072B2")
        )
    )

    fig, ax = plt.subplots(figsize=(max(12, len(df) * 0.3), 5))
    ax.bar(x, df["quality_score"], color=colors)

    for value, color, style, label in [
        (-0.75, "#E69F00", "--", "Suspect"),
        (-1.25, "#D55E00", ":", "Very bad"),
        (-1.75, "#A50026", "-.", "Extremely bad")
    ]:
        ax.axhline(value, color=color, linestyle=style, label=f"{label} = {value}")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Relative quality score")
    ax.set_title("Relative PIXC quality")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pixc_quality_score.png"), dpi=150, bbox_inches="tight")
    plt.close()


# Component plots

def plot_components(df, outdir):
    components = [
        ("geolocation_suspect_z", "Geolocation suspect"),
        ("geolocation_bad_z", "Geolocation bad"),
        ("sig0_suspect_z", "Sigma0 suspect"),
        ("sig0_bad_z", "Sigma0 bad"),
        ("sig0_p05_z", "Sigma0 P05")
    ]

    x = np.arange(len(df))
    labels = get_labels(df)

    fig, axes = plt.subplots(5, 1, figsize=(max(12, len(df) * 0.3), 12), sharex=True)

    for ax, (column, title) in zip(axes, components):
        ax.bar(x, df[column], color="#0072B2")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(2, color="#D55E00", linestyle=":")
        ax.axhline(-2, color="#D55E00", linestyle=":")
        ax.set_ylabel("Quality score")
        ax.set_title(title)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=45, ha="right", fontsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "quality_components.png"), dpi=150, bbox_inches="tight")
    plt.close()


# Raw quality flag plot

def plot_flag_fractions(df, outdir):
    components = [
        ("geolocation_qual_suspect", "Geolocation suspect"),
        ("geolocation_qual_bad", "Geolocation bad"),
        ("sig0_qual_suspect", "Sigma0 suspect"),
        ("sig0_qual_bad", "Sigma0 bad")
    ]

    x = np.arange(len(df))
    labels = get_labels(df)

    fig, axes = plt.subplots(4, 1, figsize= (max(12, len(df)* 0.3), 10), sharex=True)

    for ax, (column, title) in zip(axes, components):
        ax.bar(x, 100 * df[column], color="#0072B2")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Fraction (%)")
        ax.set_title(title)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=45, ha="right", fontsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "quality_flag_fractions.png"), dpi=150, bbox_inches="tight")
    plt.close()


# Sigma0 plot

def plot_sig0(df, outdir):
    x = np.arange(len(df))
    labels = get_labels(df)

    fig, ax = plt.subplots(figsize=(max(12, len(df) * 0.3), 5))
    ax.bar(x, df["sig0_p05"], color="tab:purple")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Sigma0 P05 (dB)")
    ax.set_title("Sigma0 P05 per PIXC")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "sig0_p05.png"), dpi=150, bbox_inches="tight")
    plt.close()



# Main

def main(pixc_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plots_dir = os.path.join(output_dir, "quality_plots")
    os.makedirs(plots_dir, exist_ok=True)

    files = sorted(glob(os.path.join(pixc_dir, "SWOT_L2_HR_PIXC_*.nc")))

    if not files:
        raise FileNotFoundError(f"No PIXC files found in {pixc_dir}")

    print(f"Found {len(files)} PIXC files.")

    workers = min(len(files), multiprocessing.cpu_count())
    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(pixc_analysis, f) for f in files]

        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"[ERROR] {exc}")

    if not results:
        raise RuntimeError("No PIXC could be processed.")

    df = pd.DataFrame(results)
    df = df.sort_values(["pass", "tile", "side", "cycle"], na_position="last").reset_index(drop=True)

    # Compute relative quality
    df = compute_quality(df)

    # Save complete table
    summary_file = os.path.join(output_dir, "pixc_analysis_summary.csv")
    df.to_csv(summary_file, index=False)

    # Console summary
    print("=" * 60)
    print("RELATIVE PIXC QUALITY")
    print("=" * 60)
    print(f"Total PIXC          : {len(df)}")
    print(f"Suspect (< -0.75)  : {df['suspect'].sum()}")
    print(f"Very bad (< -1.25) : {df['very_bad'].sum()}")
    print(f"Extreme (< -1.75)  : {df['extremely_bad'].sum()}")
    print("=" * 60)

    # bad PIXC
    bad = df[df["very_bad"]].sort_values("quality_score")

    if len(bad) > 0:
        columns = [
            "cycle",
            "pass",
            "tile",
            "side",
            "quality_score"
        ]

        print("Very bad PIXC:")
        print(bad[columns].to_string(index=False))

    # Generate plots
    plot_quality(df, plots_dir)
    plot_components(df, plots_dir)
    plot_flag_fractions(df, plots_dir)
    plot_sig0(df, plots_dir)

    print(f"Results : {summary_file}")
    print(f"Plots   : {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Relative PIXC quality analysis "
            "using quality flags and MAD"
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        dest="pixc_dir",
        required=True
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="output_dir",
        required=True
    )

    args = parser.parse_args()
    main(args.pixc_dir, args.output_dir)
