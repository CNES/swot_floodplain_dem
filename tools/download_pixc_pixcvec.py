"""
Download SWOT products (mainly PIXC and/or PIXCVec)
using providers swot and/or hydroweb_next

!!! This script works only on the CNES server Trex at the moment !!!

NB: A file eodag.yml is necessary for this script
    The providers to be defined in it are swot and hydroweb_next
"""

import sys
import os
import argparse
import getopt
import xarray as xr
import time

import eodag
from eodag import EODataAccessGateway, SearchResult
from eodag import __version__ as eodag_version
#from eodag import setup_logging

from eodag.utils.exceptions import RequestError, AuthenticationError, TimeOutError, MisconfiguredError
from requests.exceptions import ReadTimeout, ChunkedEncodingError
from urllib3.exceptions import ReadTimeoutError, ProtocolError, IncompleteRead

########################################################################################################################

# Defining the scratch directory
user = os.getenv("USER")
workdir = f"/work/scratch/data/{user}/FPDEM/pixc_pixcvec_products"

providers = ['swot', 'hydroweb_next']

product = ["SWOT_L2_HR_PIXC", "SWOT_L2_HR_PIXCVec"] # SWOT_L2_HR_PIXCVec with provider Swot; SWOT_L2_HR_PIXCVEC with provider hydroweb_next
crid = [] #['PIC0'] #, 'PGC0']

########################################################################################################################

try:
    # Initialize parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    # Adding optional argument
    parser.add_argument("-d", "--dirs",
                        type=str,
                        help="Directory where to download the products")
    parser.add_argument("-prov", "--provider",
                        nargs='*',
                        type=str,
                        help="List of providers to use (ex: swot hydroweb_next)")
    parser.add_argument("-prod", "--product",
                        nargs='*',
                        type=str,
                        help='''List of products to download (ex: SWOT_L2_HR_PIXC SWOT_L2_HR_PIXCVec)
                        NB: SWOT_L2_HR_PIXCVec with provider Swot; SWOT_L2_HR_PIXCVEC with provider hydroweb_next
                        ''')
    parser.add_argument("-pass", "--pass_tile",
                        nargs=3,
                        metavar=('pass', 'tile_number', 'tile_side'),
                        help="Number of the pass, number of tile and tile side -- ex: 264 69 Right -- !!! Not to be given with boxregion !!!")
    parser.add_argument("-box", "--boxregion",
                        nargs=4,
                        type=float,
                        metavar=('LAT_CENTER', 'LON_CENTER', 'LAT_BOX', 'LON_BOX'),
                        help="Region box: Center (lat/lon) and dimension (lat/lon) -- !!! Not to be given with pass_tile !!!")
    parser.add_argument("-c", "--crid",
                        nargs='*',
                        type=str,
                        help="List of all CRID to use (ex: PIC0 PGC0)")
    parser.add_argument("-st", "--starttime",
                        type=str,
                        help="Select the period of time : start time (ex: 2022-01-01T00:00:00)")
    parser.add_argument("-et", "--endtime",
                        type=str,
                        help="Select the period of time : end time (ex: 2023-01-01T00:00:00)")
    args = parser.parse_args()

except getopt.error as err:
    # Output error, and return with an error code
    print("Exception occurred", str(err))

if args.pass_tile and args.boxregion:
    print('You cannot provide a pass number and a box region at the same time.')
    sys.exit()

########################################################################################################################

# Parameters setup
if args.dirs:
    workdir = args.dirs

if args.product:
    product = args.product
# Change the name for PIXCVEC to fit provider Swot
# (otherwise the name of the different attributes below have to be changed)
for i, e in enumerate(product):
    if e == "SWOT_L2_HR_PIXCVEC":
        product[i] = 'SWOT_L2_HR_PIXCVec'

if args.pass_tile:
    pass_number = args.pass_tile[0]
    tile_number = args.pass_tile[1]
    tile_side = args.pass_tile[2]

if args.boxregion:
    lat_c = args.boxregion[0] # Center latitude
    lon_c = args.boxregion[1] # Center longitude
    lat_b = args.boxregion[2] # Latitude height
    lon_b = args.boxregion[3] # Longitude width

if args.crid:
    crid = args.crid
else:
    crid = []

if args.provider:
    providers = args.providers

########################################################################################################################
# EODAG setup
eodag.setup_logging(2) # 0: nothing, 1: only progress bars, 2: INFO, 3: DEBUG
dag = eodag.EODataAccessGateway()

# providers = dag.available_providers()
# print('Available providers: ', providers)

# Get list of results/products found with provided criteria
search_results = ([])
for provider in providers:
    for i_prod, e_prod in enumerate(product):
        print('Product selected: ', e_prod)

        search_criteria = {
            "productType": e_prod,
        }

        if args.starttime:
            search_criteria["start"] = args.starttime

        if args.endtime:
            search_criteria["end"] = args.endtime

        if args.pass_tile:
            if provider == 'swot':
                search_criteria['swot:pass_number'] = int(pass_number)
                search_criteria['swot:tile_number'] = int(tile_number)
                search_criteria['swot:tile_side'] = tile_side
            elif provider == 'hydroweb_next':
                search_criteria['spatial:tile_side'] = tile_side
                search_criteria['spatial:pass_id'] = int(pass_number)
                search_criteria['spatial:tile_id'] = int(tile_number)

        if args.boxregion:
            search_criteria["geom"] = {'lonmin': lon_c-lon_b,
                                       'latmin': lat_c-lat_b,
                                       'lonmax': lon_c+lon_b,
                                       'latmax': lat_c+lat_b}

        nb_items_per_page = 500
        print('Search_criteria: ', search_criteria)
        if provider == 'swot':
            if eodag_version[0] == '2':
                search_res, total_count = dag.search(**search_criteria,
                                                     provider=provider,
                                                     items_per_page=nb_items_per_page)
            else:
                search_res = dag.search(**search_criteria,
                                        provider=provider,
                                        items_per_page=nb_items_per_page)
        elif provider == 'hydroweb_next':
            search_res = dag.search_iter_page(**search_criteria,
                                              provider=provider,
                                              items_per_page=nb_items_per_page)
        search_results.extend(search_res)

        # TODO: For the moment the search by tile_pass and crid do not work with hydroweb_next!
        if len(crid) >= 1:
            print('    CRID will be selected')
            for i_crid, e_crid in enumerate(crid):
                print('    => CRID selected: ', e_crid)
                if provider == 'swot':
                    search_criteria["swot:crid"] = e_crid
                # elif provider == 'hydroweb_next':
                #     # search_criteria["processing:software"]["chain_version"] = e_crid
                #     search_criteria["swot:crid"] = e_crid
                print('Search_criteria: ', search_criteria)
                if provider == 'swot':
                    if eodag_version[0] == '2':
                        search_res, total_count = dag.search(**search_criteria,
                                                             provider=provider,
                                                             items_per_page=nb_items_per_page)
                    else:
                        search_res = dag.search(**search_criteria,
                                                provider=provider,
                                                items_per_page=nb_items_per_page)
                elif provider == 'hydroweb_next':
                    search_res = dag.search_iter_page(**search_criteria,
                                                      provider=provider,
                                                      items_per_page=nb_items_per_page)
                    # search_res = dag.search(**search_criteria,
                    #                         provider=provider,
                    #                         items_per_page=nb_items_per_page)
                search_results.extend(search_res)
        else:
            print('Search_criteria: ', search_criteria)
            if provider == 'swot':
                if eodag_version[0] == '2':
                    search_res, total_count = dag.search(**search_criteria,
                                                         provider=provider,
                                                         items_per_page=nb_items_per_page)
                else:
                    search_res = dag.search(**search_criteria,
                                            provider=provider,
                                            items_per_page=nb_items_per_page)
            elif provider == 'hydroweb_next':
                search_res = dag.search_iter_page(**search_criteria,
                                                  provider=provider,
                                                  items_per_page=nb_items_per_page)
                # search_res = dag.search(**search_criteria,
                #                         provider=provider,
                #                         items_per_page=nb_items_per_page)
            search_results.extend(search_res)

search_results = [x for sub in search_results for x in sub]
print('Total count: ', len(search_results), '\n\n')
if len(search_results) == 0:
    sys.exit()

# Filter products, if provider is hydroweb_next, by tile_pass and crid
if 'hydroweb_next' in providers:
    new_search_results = []
    for res in search_results:
        if res.properties["spatial:pass_id"] == int(pass_number):
            if crid != [] and res.properties["processing:software"]["chain_version"] in crid:
                new_search_results.append(res)
    search_results = new_search_results.copy()
    print('New total count : ', len(search_results))

# List all the products to be downloaded
new_search_results = []
prods = []
for i_res, e_res in enumerate(search_results):

    e_prod = search_results[i_res].properties['productType']
    try:
        pixc_cycle = "{:03d}".format(search_results[i_res].properties['swot:cycle_number'])
        pass_number = "{:03d}".format(search_results[i_res].properties['swot:pass_number'])
        tile_number = "{:03d}".format(search_results[i_res].properties['swot:tile_number'])
        tile_side = search_results[i_res].properties['swot:tile_side']
        crid = search_results[i_res].properties['swot:crid']
    except:
        pixc_cycle = "{:03d}".format(search_results[i_res].properties['spatial:cycle_id'])
        pass_number = "{:03d}".format(search_results[i_res].properties['spatial:pass_id'])
        tile_number = "{:03d}".format(search_results[i_res].properties['spatial:tile_id'])
        tile_side = search_results[i_res].properties['spatial:tile_side']
        crid = search_results[i_res].properties['processing:software']['chain_version']
    #
    products_path = f'{e_prod}_{pixc_cycle}_{pass_number}_{tile_number}{tile_side[0]}_{crid}'
    # Remove duplicates if there are any
    if products_path not in prods:
        prods.append(products_path)
        new_search_results.append(e_res)
        print('products: ', i_res + 1, products_path)
search_results = new_search_results.copy()

#
if input(f"Do you want to continue and download all {len(search_results)} products? [y/n]") == "y":

    # Download PIXC/PIXCVEC in the netcdf format
    for i_res, e_res in enumerate(search_results):

        print(f'\nPreparing to download product {i_res+1}/{len(search_results)}')

        # Preparing the name of the downloading directory
        e_prod = search_results[i_res].properties['productType']
        try:
            pixc_cycle = "{:03d}".format(search_results[i_res].properties['swot:cycle_number'])
            pass_number = "{:03d}".format(search_results[i_res].properties['swot:pass_number'])
            tile_number = "{:03d}".format(search_results[i_res].properties['swot:tile_number'])
            tile_side = search_results[i_res].properties['swot:tile_side']
            crid = search_results[i_res].properties['swot:crid']
        except:
            pixc_cycle = "{:03d}".format(search_results[i_res].properties['spatial:cycle_id'])
            pass_number = "{:03d}".format(search_results[i_res].properties['spatial:pass_id'])
            tile_number = "{:03d}".format(search_results[i_res].properties['spatial:tile_id'])
            tile_side = search_results[i_res].properties['spatial:tile_side']
            crid = search_results[i_res].properties['processing:software']['chain_version']

        #
        products_path = f'{workdir}/{e_prod}_{pixc_cycle}_{pass_number}_{tile_number}{tile_side[0]}_{crid}'
        print('Path to product: ', products_path)

        if os.path.isdir(f'{products_path}') is False:
            os.mkdir(f'{products_path}')
        else:
            # Check if the product needs to be downloaded again due to error, file corruption...
            if len(os.listdir(f'{products_path}')) > 1:
                ncfile = [i for i in os.listdir(f'{products_path}') if i[-3:] == '.nc']
                if ncfile != []:
                    if os.path.getsize(f'{products_path}/{ncfile[0]}') > 32:
                        try:
                            if 'PIXCVec' in ncfile[0]:
                                dnc = xr.open_dataset(f'{products_path}/{ncfile[0]}', decode_times=False)
                            elif 'PIXC_' in ncfile[0]:
                                dnc = xr.open_dataset(f'{products_path}/{ncfile[0]}', group="pixel_cloud", decode_times=False)
                                dnc = dnc.drop_dims('complex_depth') 
                            try:
                                dnc = dnc.drop_dims('num_pixc_lines')
                            except:
                                dnc = dnc
                            #import netCDF4
                            #cdf_dataset = netCDF4.Dataset(f'{products_path}/{ncfile[0]}')
                            #print(cdf_dataset.variables )
                            if 'PIXCVec' in ncfile[0]:
                                var = 'height_vectorproc'
                            elif 'PIXC_' in ncfile[0]:
                                var = 'height'
                            if var in list(dnc.keys()):
                                print('Directory exists already and data (netcdf) present inside.')
                                continue
                        except:
                            print('Netcdf present in directory but seems corrupted. Download done again.')

        # Downloading product
        for i in range(5):
            try:
                outputpath = dag.download(search_results[i_res], outputs_prefix=products_path)
                break
            except ReadTimeout as e:
                print("ReadTimeout ERROR REGARDS request attempt number %i unsuccessful" % (i + 1))
                time.sleep(60 * 5)
            except RequestError as e:
                print("RequestError ERROR REGARDS request attempt number %i unsuccessful" % (i + 1))
                time.sleep(60 * 5)
            except AuthenticationError as e:
                print("AuthenticationError ERROR REGARDS request attempt number %i unsuccessful" % (i + 1))
                time.sleep(60 * 5)
            except TimeOutError as e:
                print("TimeOutError ERROR REGARDS request attempt number %i unsuccessful" % (i + 1))
                time.sleep(60 * 5)
            except MisconfiguredError as e:
                print("MisconfiguredError ERROR REGARDS request attempt number %i unsuccessful" % (i + 1))
                time.sleep(60 * 5)
            except ReadTimeoutError as e:
                print("ReadTimeoutError ERROR REGARDS request attempt number %i unsuccessful" % (i + 1))
                time.sleep(60 * 5)
            except ChunkedEncodingError as e:
                print("ChunkedEncodingError ERROR REGARDS request attempt number %i unsuccessful" % (i + 1))
                time.sleep(60 * 5)
            except ProtocolError as e:
                print.info("ProtocolError ERROR REGARDS request attempt number %i unsuccessful" % (i + 1))
                time.sleep(60 * 5)
            except IncompleteRead as e:
                print.info("IncompleteRead ERROR REGARDS request attempt number %i unsuccessful" % (i + 1))
                time.sleep(60 * 5)
            except TimeoutError as e:
                print.info("TimeoutError ERROR REGARDS request attempt number %i unsuccessful" % (i + 1))
                time.sleep(60 * 5)

        # Move netdcf file to workdir
        listfiles = os.listdir(f'{outputpath}')
        nc_file = [listfiles[i] for i in range(len(listfiles)) if listfiles[i][-3:] == '.nc'][0]
        os.system(f'mv {outputpath}/{nc_file} {products_path}/')
