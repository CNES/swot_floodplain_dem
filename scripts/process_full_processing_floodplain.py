import argparse
import os
import logging
import my_rdf_file as my_rdf
import my_timer as my_timer

from names import (FPDEM_BASENAME, FPDEM_POINTCLOUD_BASENAME,
                   MASK_SUFFIX, FPDEM_RASTER_BASENAME, MRF_RASTER_BASENAME, compute_name)

# FPDEM processes
from process_floodplain import Floodplain
from process_extract_area import Extract_Area
from process_raster import FPDEM_Raster

# MRF process
from process_mrf import MRF_run

# Main program
if __name__ == "__main__":

    timer = my_timer.Timer()
    timer.start()

    # Get parameters
    descr = "Compute fpdem or mrf intermediate product from multiple tiles of PIXC products (and their associated PIXCVecRiver products)."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument("parameter_file", help="parameter_file (*.rdf)")
    args = parser.parse_args()

    level = getattr(logging, "INFO")
    logging.basicConfig(filename=None, format='%(asctime)s [%(levelname)s] %(message)s', level=level)

    parameters = my_rdf.myRdfReader(args.parameter_file)

    method = parameters.getValue("method to use")

    # Prepare output names
    output_directory = parameters.getValue("output directory")
    # Creating output directory
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    if method == 'fpdem':
        print("===== Floodplain DEM processing = BEGIN =====")
        print("")

        # Compute floodplain DEM pixels boundaries extraction
        fpdem = Floodplain(parameters)
        fpdem.compute_fpdem_pointcloud_boundaries()
        fpdem.write_fpdem_pointcloud_output()
        logging.info('Calculation of FPDEM was performed')

        # Compute region of intersect where raster floodplain dem product in computed (between min and max water extent)
        output_fpdem_pointcloud_name = compute_name(output_directory, FPDEM_POINTCLOUD_BASENAME,
                                                    parameters.getValue("tile name"),
                                                    parameters.getValue("first date name"),
                                                    parameters.getValue("last date name"))
        output_mask_name = os.path.join(output_directory, FPDEM_BASENAME + MASK_SUFFIX)
        extract_area = Extract_Area(parameters, input_file=output_fpdem_pointcloud_name, output_file=output_mask_name)
        extract_area.load_input_extract_area()
        extract_area.extract_area_polygons()
        logging.info("Extraction of the area was performed")

        # Compute rasterization of pixel cloud boundaries and produce final FPDEM product
        output_fpdem_raster_name = compute_name(output_directory, FPDEM_RASTER_BASENAME,
                                                parameters.getValue("tile name"),
                                                parameters.getValue("first date name"),
                                                parameters.getValue("last date name"))
        fpdem_raster = FPDEM_Raster(parameters, input_file=output_fpdem_pointcloud_name,
                                    output_file=output_fpdem_raster_name,
                                    mask=output_mask_name)
        fpdem_raster.load_input_fpdem_raster()
        fpdem_raster.compute_fpdem_raster()
        fpdem_raster.write_fpdem_raster()
        logging.info('Rasterization of FPDEM product was performed')
   
        print("")
        print("===== Floodplain DEM processing  = END =====")

    elif method == 'mrf':

        print("===== Markov Random Field processing = BEGIN =====")
        print("")

        output_mrf_raster_name = compute_name(output_directory, MRF_RASTER_BASENAME,
                                              parameters.getValue("tile name"),
                                              parameters.getValue("first date name"),
                                              parameters.getValue("last date name"))
        mrf = MRF_run(parameters, output_file=output_mrf_raster_name)
        mrf.mrf_compute()
        mrf.mrf_raster()

        print("")
        print("===== MRF processing  = END =====")

    print(timer.stop())
