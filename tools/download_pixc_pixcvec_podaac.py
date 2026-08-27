"""
Download PODAAC SWOT Level-2 products PIXC or PIXCVec from Earthdata interface.
using NASA Earthdata services via the earthaccess library.

The script allows:
- Searching for SWOT PIXC / PIXCVec granules (versions C and D)
- Filtering by time range, bounding box, pass, tile and swath side, and CRID
- Skipping already-downloaded and valid NetCDF files

This script is designed to run on local Linux systems
and requires a valid NASA Earthdata account.

Requirements:
- An Earthdata login (configured via ~/.netrc or interactive login)

NB:
- Tested on local Linux systems with internet access and Earthdata credentials.
- Not tested on CNES TREX server
"""

import os
import sys
import argparse
import time
import socket
import earthaccess
import xarray as xr
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from requests.exceptions import (
    ReadTimeout,
    ConnectionError,
    ChunkedEncodingError
)

# print(earthaccess.search_data.__doc__)
# Retry download function
def download_with_retries(granule, outdir,
                          max_retries=5,
                          sleep_minutes=5):
    for attempt in range(1, max_retries + 1):
        try:
            earthaccess.download(granule, outdir)
            return True
        except (ReadTimeout, ConnectionError, ChunkedEncodingError, socket.timeout, TimeoutError) as e:
            print(f"Retry {attempt}/{max_retries} failed ({type(e).__name__})")
            if attempt < max_retries:
                print(f"Sleeping {sleep_minutes} minutes...")
                time.sleep(sleep_minutes * 60)
            else:
                print("Maximum retries reached.")
        except Exception as e:
            print("Fatal error:", repr(e))
            return False
    return False


# Argument parser
parser = argparse.ArgumentParser(
    description="""
Download SWOT PIXC / PIXCVec granules from PODAAC.

Usage examples:

  1) Download a version C granule with all options:
     python download_pixc_pixcvec_podaac.py -d ./downloads -pd SWOT_L2_HR_PIXC -pts 497 199 L -st 2024-01-01T00:00:00 -et 2024-01-02T00:00:00 -crid PIC0

  2) Download a version D granule with all options:
     python download_pixc_pixcvec_podaac.py -d ./downloads -pd SWOT_L2_HR_PIXCVec_D -pts 196 185 R -st 2024-02-01T00:00:00 -et 2024-02-02T00:00:00 -crid PGD0
""",
    formatter_class=argparse.RawTextHelpFormatter
)

# Arguments
parser.add_argument("-d", required=True, metavar="DIR", help="Output directory")
parser.add_argument("-pd", metavar="PRODUCT", required=True, dest="product",
                    help=(
                        "Version C products: SWOT_L2_HR_PIXC or SWOT_L2_HR_PIXCVec\n"
                        "Version D products: SWOT_L2_HR_PIXC_D or SWOT_L2_HR_PIXCVec_D"
                         )
                    )
parser.add_argument("-pts", nargs=3, metavar=("PASS", "TILE", "SIDE"),
                    help="Filter by pass, tile, side (e.g., 5 12 L)"
                    )
parser.add_argument(
    "-box",
    nargs=4,
    type=float,
    metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
    help="Bounding box: lon_min lat_min lon_max lat_max"
                    )
parser.add_argument("-st", metavar="START_TIME", type=str,
                    help="Start time (YYYY-MM-DDTHH:MM:SS)"
                    )
parser.add_argument("-et", metavar="END_TIME", type=str,
                    help="End time (YYYY-MM-DDTHH:MM:SS)"
                    )
parser.add_argument("-crid", metavar="CRID",
                    help="CRID to filter (e.g., PIC0)")

args = parser.parse_args()

if args.pts and args.box:
    parser.error("Use either -pts or -box, not both.")

# Earthdata login
earthaccess.login()

# Search granules using concept_id
results = []

concept_ids = {
    # Version C
    "SWOT_L2_HR_PIXC": "C2799438266-POCLOUD",
    "SWOT_L2_HR_PIXCVec": "C2799438260-POCLOUD",

    # Version D
    "SWOT_L2_HR_PIXC_D": "C3233944986-POCLOUD",
    "SWOT_L2_HR_PIXCVec_D": "C3233944988-POCLOUD"
}

# Single product type only
prod = args.product
cid = concept_ids.get(prod)
if not cid:
    print(f"No concept_id defined for {prod}, exiting")
    sys.exit(1)

print(f"Searching product: {prod}")

search_kwargs = {
    "concept_id": cid,
    "count": -1
}

# Temporal filter
if args.st or args.et:
    search_kwargs["temporal"] = (
        args.st + "Z" if args.st else None,
        args.et + "Z" if args.et else None
    )

# Spatial filter (bbox)
if args.box:
    lon_min, lat_min, lon_max, lat_max = args.box

    search_kwargs["bounding_box"] = (
        lon_min,
        lat_min,
        lon_max,
        lat_max
    )

# Granule name filter
if args.pts:
    p, t, s = args.pts
    p = f"{int(p):03d}"
    t = f"{int(t):03d}"
    s = s[0].upper()

    if args.crid:
        search_kwargs["granule_name"] = (
            f"SWOT*_{p}_{t}{s}_*_{args.crid}_*"
        )
    else:
        search_kwargs["granule_name"] = (
            f"SWOT*_{p}_{t}{s}_*"
        )

elif args.crid:
    search_kwargs["granule_name"] = (
        f"SWOT*_{args.crid}_*"
    )

print("CMR request:", search_kwargs)

granules = earthaccess.search_data(**search_kwargs)

print(f"Found {len(granules)} granules")
results = []

for g in granules:

    name = g["umm"]["GranuleUR"]

    # Safety check on pass/tile/side
    if args.pts:
        if f"_{p}_" not in name or f"_{t}{s}_" not in name:
            continue

    # Safety check on CRID
    if args.crid:
        if f"_{args.crid}_" not in name:
            continue

    results.append(g)

print(f"Granules retained after validation: {len(results)}")

for g in results[:10]:
    print("  ", g["umm"]["GranuleUR"])

# Confirmation
resp = input(f"Download {len(results)} products? [y/n] ")
if resp.lower() != "y":
    sys.exit(0)

# Download loop (parallelized)
os.makedirs(args.d, exist_ok=True)

max_workers = 5  # Adjust according to your network


def download_granule(g):
    """Download a single granule with retries, skip if already downloaded."""
    name = g["umm"]["GranuleUR"]
    outdir = os.path.join(args.d, name.replace(".nc", ""))
    os.makedirs(outdir, exist_ok=True)

    # Check if file already exists and is readable
    existing = [f for f in os.listdir(outdir) if f.endswith(".nc")]
    if existing:
        try:
            xr.open_dataset(os.path.join(outdir, existing[0]), decode_times=False)
            print(f"Already present: {name}")
            return
        except Exception:
            print("Corrupted NetCDF detected, re-downloading")

    # Download
    print(f"Downloading {name}")
    success = download_with_retries(g, outdir)
    if not success:
        print(f"FAILED: {name}")


with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(download_granule, g): g for g in results}
    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            g = futures[future]
            print(f"ERROR downloading {g['umm']['GranuleUR']}: {e}")

print("All downloads completed.")