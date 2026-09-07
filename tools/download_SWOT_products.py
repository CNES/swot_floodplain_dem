"""
Download SWOT products (mainly PIXC and/or PIXCVec) from hydroweb.next
! An API key must be set up on hydroweb.next website !
"""

import os
import argparse
import getopt
import time
from pystac_client import Client
import requests
from requests.exceptions import ReadTimeout, ChunkedEncodingError
from urllib3.exceptions import ReadTimeoutError, ProtocolError, IncompleteRead
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

########################################################################################################################

# Some default parameters if wanted

workdir = f"{os.getcwd()}/FPDEM_products"

# apikey = ""

product = ["SWOT_L2_HR_PIXC", "SWOT_L2_HR_PIXCVEC"]
crid = ['PIC0', 'PIC2', 'PID0', 'PGD0']

########################################################################################################################

try:
    # Initialize parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    # Adding optional argument
    parser.add_argument("-api", "--apikey",
                        type=str,
                        help="API key obtained from Hydroweb Next website (in user settings)")
    parser.add_argument("-dir", "--outdir",
                        type=str,
                        help="Directory where to download the products")
    parser.add_argument("-prod", "--product",
                        nargs='*',
                        type=str,
                        help='''List of products to download (ex: SWOT_L2_HR_PIXC SWOT_L2_HR_PIXCVec)
                        NB: SWOT_L2_HR_PIXCVec with provider Swot; SWOT_L2_HR_PIXCVEC with provider hydroweb_next
                        ''')
    parser.add_argument("-box", "--boxregion",
                        nargs=4,
                        type=float,
                        metavar=('LON_MIN', 'LAT_MIN', 'LON_MAX', 'LAT_MAX'),
                        help='Region box: lon_min, lat_min, lon_max, lat_max. ex: -box " -16." "11." " -15.5" "11.5"')
    parser.add_argument("-st", "--starttime",
                        type=str,
                        help="Select the period of time : start time (ex: 2022-01-01T00:00:00Z)")
    parser.add_argument("-et", "--endtime",
                        type=str,
                        help="Select the period of time : end time (ex: 2023-01-01T00:00:00Z)")
    parser.add_argument("-pass", "--passtile",
                        nargs=3,
                        metavar=('pass', 'tile_number', 'tile_side'),
                        help="Number of the pass, number of tile and tile side -- ex: 264 69 Right")
    parser.add_argument("-c", "--crid",
                        nargs='*',
                        type=str,
                        help="List of all CRID to use (ex: PIC0 PGC0)")
    parser.add_argument("-cy", "--cycle",
                        nargs='*',
                        type=int,
                        help="List of cycle numbers to use")
    args = parser.parse_args()

except getopt.error as err:
    # Output error, and return with an error code
    print("Exception occurred", str(err))


########################################################################################################################
# Download function
def download_file(url_address, product_filename):
    with requests.get(url_address, headers=headers, stream=True) as request:
        request.raise_for_status()
        file_size = int(request.headers.get("content-length", 0))
        with open(product_filename, "wb") as file, tqdm(
            desc=product_filename,
            total=file_size if file_size > 0 else None,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            leave=True,
        ) as bar:
            for chunk in request.iter_content(chunk_size=8192):
                if chunk:
                    size = file.write(chunk)
                    bar.update(size)


def get_item_info(prod_item):
    e_prod = prod_item.properties['hydro:data_type']
    pixc_cycle = int(prod_item.properties['spatial:cycle_id'])
    pass_number = int(prod_item.properties['spatial:pass_id'])
    tile_number = int(prod_item.properties['spatial:tile_id'])
    tile_side = prod_item.properties['spatial:tile_side']
    crid = prod_item.properties['processing:software']['chain_version']
    item_info = f'{e_prod}_{pixc_cycle:03d}_{pass_number:03d}_{tile_number:03d}{tile_side[0]}_{crid}'
    return item_info

########################################################################################################################

# Parameters setup
if args.apikey:
    apikey = args.apikey

if args.outdir:
    workdir = args.outdir
# Create workdir if it does not exist
if not os.path.isdir(workdir):
    os.mkdir(workdir)

if args.product:
    product = args.product

if args.passtile:
    pass_number = int(args.passtile[0])
    tile_number = str(args.passtile[1])
    tile_side = str(args.passtile[2])

if args.boxregion:
    lat_min = float(args.boxregion[1])
    lon_min = float(args.boxregion[0])
    lat_max = float(args.boxregion[3])
    lon_max = float(args.boxregion[2])

if args.crid:
    crid = args.crid
else:
    crid = []

########################################################################################################################

headers = {
    "X-API-Key": f"{apikey}",
}

client = Client.open(
    url="https://hydroweb.next.theia-land.fr/api/catalog/stac",
    headers=headers
)

# List all collections available in the STAC API
# for collection in client.get_collections():
#     summaries_dict = collection.summaries.to_dict()
#     print(f"{collection.id:<40} {summaries_dict.get('total_items','-')}")

# Creating search request
search_parameters = {
    "collections": product,
}

if args.boxregion:
    search_parameters["bbox"] = [lon_min, lat_min, lon_max, lat_max]

if args.starttime or args.endtime:
    if 'query' not in search_parameters:
        search_parameters["query"] = {}
    if args.starttime:
        if "end_datetime" not in search_parameters["query"]:
            search_parameters["query"]["end_datetime"] = {}
        search_parameters["query"]["end_datetime"]["gte"] = args.starttime
    if args.endtime:
        if "start_datetime" not in search_parameters["query"]:
            search_parameters["query"]["start_datetime"] = {}
        search_parameters["query"]["start_datetime"]["lte"] = args.endtime

if args.passtile:
    if 'query' not in search_parameters:
        search_parameters["query"] = {}

    # pass_id
    if "spatial:pass_id" not in search_parameters["query"]:
        search_parameters["query"]["spatial:pass_id"] = {}
    search_parameters["query"]["spatial:pass_id"]["eq"] = pass_number
    # tile_id
    if "spatial:tile_id" not in search_parameters["query"]:
        search_parameters["query"]["spatial:tile_id"] = {}
    search_parameters["query"]["spatial:tile_id"]["eq"] = tile_number
    # tile_side
    if "spatial:tile_side" not in search_parameters["query"]:
        search_parameters["query"]["spatial:tile_side"] = {}
    search_parameters["query"]["spatial:tile_side"]["eq"] = tile_side

# Does not filter anything at the moment !
if args.crid:
    if 'query' not in search_parameters:
        search_parameters["query"] = {}
    if "processing:software" not in search_parameters["query"]:
        search_parameters["query"]["processing:software"] = {}
    search_parameters["query"]["processing:software"]["chain_version"] = crid

search_parameters["max_items"] = 200

print('Search request :', search_parameters)

# Search the client
search = client.search(**search_parameters)
if search.matched() > 200:
    print("! Search parameter max_items set to 200 => You might want to increase it !")
items = list(search.items())

# Filtering CRID
new_items1 = []
if crid != []:
    for item in items:
        if item.properties["processing:software"]["chain_version"] in crid:
            new_items1.append(item)
else:
    new_items1 = items.copy()

# Filtering cycles
new_items2 = []
if args.cycle:
    for item1 in new_items1:
        if item1.properties["spatial:cycle_id"] in args.cycle:
            new_items2.append(item1)
else:
    new_items2 = new_items1.copy()

# Count the items found and print their name
new_items = new_items2.copy()
print('Number of items found  : ', len(new_items))
# List all remaining products
for item in new_items:
    print(get_item_info(item))

# Downloading files
def download_item(item):
    for name, asset in item.assets.items():
        # Only the netcdf will be downloaded
        if name[-3:] == '.nc':
            filename = name[:-3]
            filepath = os.path.join(workdir, filename)

            for itry in range(5):
                try:
                    download_file(asset.href, f"{filepath}.nc")
                    print(f"\nDownloaded : {filepath}.nc")
                    break

                except ReadTimeout as e:
                    print("ReadTimeout ERROR hydroweb.next request attempt number %i unsuccessful" % (itry + 1))
                    time.sleep(60 * 5)

                except ReadTimeoutError as e:
                    print("ReadTimeoutError ERROR hydroweb.next request attempt number %i unsuccessful" % (itry + 1))
                    time.sleep(60 * 5)

                except ChunkedEncodingError as e:
                    print("ChunkedEncodingError ERROR hydroweb.next request attempt number %i unsuccessful" % (itry + 1))
                    time.sleep(60 * 5)

                except ProtocolError as e:
                    print("ProtocolError ERROR hydroweb.next request attempt number %i unsuccessful" % (itry + 1))
                    time.sleep(60 * 5)

                except IncompleteRead as e:
                    print("IncompleteRead ERROR hydroweb.next request attempt number %i unsuccessful" % (itry + 1))
                    time.sleep(60 * 5)

                except TimeoutError as e:
                    print("TimeoutError ERROR hydroweb.next request attempt number %i unsuccessful" % (itry + 1))
                    time.sleep(60 * 5)

            break


if input(f"Do you want to continue and download products? [y/n] ") == "y":

    # Number of simultaneous downloads
    max_workers = 8

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_item, item)
            for item in new_items
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"\nDownload ERROR: {e}")