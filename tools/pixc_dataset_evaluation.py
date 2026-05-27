import os
from glob import glob
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed


# Utils

def zscore(x):
    """
    Compute a z-score using median and MAD.
    """
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return (x - med) / (mad + 1e-6)

# Quality flag handling

BITS_DEF = {
    "suspect": range(0, 16),
    "degraded": range(16, 25),
    "bad": range(25, 32),
}


def has_any_bit(value, bits):
    """Check if any bit from a set is present in the flag value"""
    for b in bits:
        if (value & (1 << b)) != 0:
            return True
    return False


def evaluate_flag(values):
    """
    Compute proportions of OK / suspect / degraded / bad
    """
    values = values.astype(np.int64)

    if len(values) == 0:
        return dict(ok=np.nan, suspect=np.nan,
                    degraded=np.nan, bad=np.nan)

    return {
        "ok": np.mean(values == 0),
        "suspect": np.mean([has_any_bit(v, BITS_DEF["suspect"]) for v in values]),
        "degraded": np.mean([has_any_bit(v, BITS_DEF["degraded"]) for v in values]),
        "bad": np.mean([has_any_bit(v, BITS_DEF["bad"]) for v in values]),
    }

# PIXC extraction

def pixc_analysis(pixc_file):
    """
    Extract quality metrics from a PIXC file
    """
    ds = xr.open_dataset(pixc_file, group="pixel_cloud", decode_times=False)

    df = ds[
        ["classification",
         "classification_qual",
         "geolocation_qual",
         "sig0_qual",
         "sig0"]
    ].to_dataframe()

    # Keep only water-like pixels
    df = df[df["classification"] > 1]

    name = os.path.basename(pixc_file)
    parts = name.split("_")

    metrics = {
        "pixc_file": name,
        "short_name": parts[4] if len(parts) > 4 else parts[0],
        "cycle": parts[2] if len(parts) > 2 else "unknown",
        "n_pixels": len(df),
    }

    # Compute flag statistics
    for var in ["classification_qual", "geolocation_qual", "sig0_qual"]:
        res = evaluate_flag(df[var].fillna(0).values)
        for k, v in res.items():
            metrics[f"{var}_{k}"] = v

    # Sigma0 low fraction
    sig0 = df["sig0"].dropna().values
    metrics["sig0_low_frac"] = np.mean(sig0 < 0) if len(sig0) > 0 else np.nan

    ds.close()
    return metrics

# SCORING

def compute_quality_score(df):
    """
    Compute quality score and detection flags.
    """

    # Base score
    base_score = (
        0.2 * df["classification_qual_ok"]
        + 0.4 * df["geolocation_qual_ok"]
        + 0.4 * df["sig0_qual_ok"]
    )

    #  Geoloc penalty
    geo_raw = (
        df["geolocation_qual_suspect"]
        + 2 * df["geolocation_qual_bad"]
    )

    geo_z = zscore(geo_raw)

    # Penalize only anomalous values (z > 0.5)
    geo_penalty = np.clip(geo_z - 0.5, 0, None)

    # Sigma0 low frac penalty
    sig0_z = zscore(df["sig0_low_frac"])
    sig0_penalty = np.clip(sig0_z - 0.5, 0, None)

    # Final score
    df["quality_score"] = (
        base_score
        - 0.3 * geo_penalty
        - 0.3 * sig0_penalty
    )

    df["quality_score"] = np.clip(df["quality_score"], 0, 1)

    # Bad PIXC detection
    mean = df["quality_score"].mean()
    std = df["quality_score"].std()

    threshold_bad = mean - 0.5 * std

    df["very_bad"] = df["quality_score"] < threshold_bad

    return df, threshold_bad



# PLOTS

def get_labels(df):
    return df["cycle"].astype(str) + "_" + df["short_name"]


def plot_global_score(df, threshold_bad, outdir):
    """
    Plot quality score with color-coded bad PIXC.
    """

    x = np.arange(len(df))
    labels = get_labels(df)

    fig, ax = plt.subplots(figsize=(max(12, len(df)*0.3), 5))

    # Color-blind friendly palette
    color_good = "#0072B2"
    color_bad = "#D55E00"

    colors = [
        color_bad if v else color_good
        for v in df["very_bad"]
    ]

    ax.bar(x, df["quality_score"], color=colors)

    median_score = df["quality_score"].median()

    ax.axhline(median_score, color="black", linestyle="--",
               label=f"Median = {median_score:.2f}")

    ax.axhline(threshold_bad, color=color_bad, linestyle=":",
               label=f"Very bad = {threshold_bad:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Quality score")
    ax.set_title("PIXC quality score (robust data-driven)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "quality_score.png"), dpi=150)
    plt.close()


def plot_sig0_low_fraction(df, outdir):
    """
    Plot sigma0 low fraction.
    """

    x = np.arange(len(df))
    labels = get_labels(df)

    fig, ax = plt.subplots(figsize=(max(12, len(df)*0.3), 5))

    ax.bar(x, df["sig0_low_frac"], color="tab:purple")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction sigma0 < 0 dB")
    ax.set_title("Low sigma0 fraction per PIXC")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "sig0_low_fraction.png"), dpi=150)
    plt.close()

# MAIN

def main(pixc_dir, output_dir):

    plots_dir = os.path.join(output_dir, "quality_plots")
    os.makedirs(plots_dir, exist_ok=True)

    pixc_files = sorted(glob(os.path.join(pixc_dir, "SWOT_L2_HR_PIXC_*.nc")))

    n_workers = min(len(pixc_files), multiprocessing.cpu_count())

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(pixc_analysis, p) for p in pixc_files]
        for f in as_completed(futures):
            results.append(f.result())

    df = pd.DataFrame(results)
    df = df.sort_values(["cycle", "short_name"]).reset_index(drop=True)

    # Save table
    csv_file = os.path.join(output_dir, "pixc_analysis_summary.csv")
    df.to_csv(csv_file, index=False)

    print(f"Quality table written to {csv_file}")

    # Compute scores
    df, threshold_bad = compute_quality_score(df)

    # Plot results
    plot_global_score(df, threshold_bad, plots_dir)
    plot_sig0_low_fraction(df, plots_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PIXC quality analysis")

    parser.add_argument("-i", "--input", dest="pixc_dir", required=True)
    parser.add_argument("-o", "--output", dest="output_dir", required=True)

    args = parser.parse_args()
    main(args.pixc_dir, args.output_dir)