
import os
from glob import glob
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed


# Bit definitions
BITS_DEF = {
    "suspect": range(0, 16),    # bits 0-15
    "degraded": range(16, 25),  # bits 16-24
    "bad": range(25, 32),       # bits 25-31
}


def has_any_bit(value, bits):
    for b in bits:
        if (value & (1 << b)) != 0:
            return True
    return False


def evaluate_flag(values):
    values = values.astype(np.int64)

    if len(values) == 0:
        return dict(ok=np.nan, suspect=np.nan, degraded=np.nan, bad=np.nan)

    return {
        "ok": np.mean(values == 0),
        "suspect": np.mean([has_any_bit(v, BITS_DEF["suspect"]) for v in values]),
        "degraded": np.mean([has_any_bit(v, BITS_DEF["degraded"]) for v in values]),
        "bad": np.mean([has_any_bit(v, BITS_DEF["bad"]) for v in values]),
    }


# PIXC evaluation
def pixc_analysis(pixc_file):
    ds = xr.open_dataset(pixc_file, group="pixel_cloud", decode_times=False)

    df = ds[
        [
            "classification",
            "classification_qual",
            "geolocation_qual",
            "sig0_qual",
            "sig0",
        ]
    ].to_dataframe()

    # Keep only water pixels
    df = df[df["classification"] > 1]

    metrics = {
        "pixc_file": os.path.basename(pixc_file),
        "short_name": os.path.basename(pixc_file).split("_")[4],
        "n_pixels": len(df),
    }

    # Quality flags analysis
    for var in ["classification_qual", "geolocation_qual", "sig0_qual"]:
        res = evaluate_flag(df[var].fillna(0).values)
        for k, v in res.items():
            metrics[f"{var}_{k}"] = v

    # Low sigma0 fraction analysis (sigma0 < 0 dB)
    sig0 = df["sig0"].dropna().values
    metrics["sig0_low_frac"] = np.mean(sig0 < 0) if len(sig0) > 0 else np.nan

    ds.close()
    return metrics


# Plot functions
def plot_quality_bars(df, flag, outdir):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    w = 0.2

    ax.bar(x - 1.5*w, df[f"{flag}_ok"],       w, label="OK",       color="tab:green")
    ax.bar(x - 0.5*w, df[f"{flag}_suspect"],  w, label="Suspect",  color="tab:orange")
    ax.bar(x + 0.5*w, df[f"{flag}_degraded"], w, label="Degraded", color="tab:red")
    ax.bar(x + 1.5*w, df[f"{flag}_bad"],      w, label="Bad",      color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(df["short_name"], rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion of water pixels")
    ax.set_title(f"{flag} quality per PIXC")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{flag}_quality_bars.png"), dpi=150)
    plt.close()


# Plot 1 : quality score OK only
def plot_global_score(df, outdir, threshold=0.7):
    df["quality_score"] = (
        df["classification_qual_ok"]
        * df["geolocation_qual_ok"]
        * df["sig0_qual_ok"]
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["short_name"], df["quality_score"], color="tab:blue")
    ax.axhline(threshold, color="red", linestyle="--", label="Threshold")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Quality score (OK only)")
    ax.set_title("PIXC quality score  OK pixels only")
    ax.legend()

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "quality_score_ok_only.png"), dpi=150)
    plt.close()


# Plot 2 : quality score OK + DEGRADED
def plot_global_score_ok_degraded(df, outdir, threshold=0.7):

    df["quality_score_ok_degraded"] = (
        (df["classification_qual_ok"] + df["classification_qual_degraded"])
        * (df["geolocation_qual_ok"]   + df["geolocation_qual_degraded"])
        * (df["sig0_qual_ok"]           + df["sig0_qual_degraded"])
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["short_name"], df["quality_score_ok_degraded"], color="tab:green")
    ax.axhline(threshold, color="red", linestyle="--", label="Threshold")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Quality score (OK + Degraded)")
    ax.set_title("PIXC quality score OK + Degraded pixels")
    ax.legend()

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "quality_score_ok_degraded.png"), dpi=150)
    plt.close()


def plot_sig0_low_fraction(df, outdir, threshold=0.05):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(df["short_name"], df["sig0_low_frac"], color="tab:purple")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1,
               label="Suspicious threshold")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of water pixels with sigma0 < 0 dB")
    ax.set_title("Low sigma0 fraction per PIXC")
    ax.legend()

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "sig0_low_fraction.png"), dpi=150)
    plt.close()


def main(pixc_dir, output_dir):

    plots_dir = os.path.join(output_dir, "quality_plots")
    os.makedirs(plots_dir, exist_ok=True)

    pixc_files = sorted(glob(os.path.join(pixc_dir, "SWOT_L2_HR_PIXC_*.nc")))
    if not pixc_files:
        raise RuntimeError(f"No PIXC files found in {pixc_dir}")

    n_workers = min(len(pixc_files), multiprocessing.cpu_count())
    print(f"Processing {len(pixc_files)} PIXC files using {n_workers} processes")

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(pixc_analysis, p) for p in pixc_files]
        for f in as_completed(futures):
            results.append(f.result())

    df = pd.DataFrame(results)
    df = df.sort_values("pixc_file").reset_index(drop=True)

    csv_file = os.path.join(output_dir, "pixc_analysis_summary.csv")
    df.to_csv(csv_file, index=False)
    print(f"Quality table written to {csv_file}")

    # Plots
    plot_quality_bars(df, "classification_qual", plots_dir)
    plot_quality_bars(df, "geolocation_qual", plots_dir)
    plot_quality_bars(df, "sig0_qual", plots_dir)
    plot_global_score(df, plots_dir)              # score strict
    plot_global_score_ok_degraded(df, plots_dir)  # score élargi
    plot_sig0_low_fraction(df, plots_dir)

# Main
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="PIXC quality analysis"
    )

    parser.add_argument(
        "-i", "--input",
        dest="pixc_dir",
        required=True,
        help="Directory containing PIXC netCDF files"
    )

    parser.add_argument(
        "-o", "--output",
        dest="output_dir",
        required=True,
        help="Output directory for CSV and plots"
    )

    args = parser.parse_args()
    main(args.pixc_dir, args.output_dir)