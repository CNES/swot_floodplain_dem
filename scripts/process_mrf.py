# -*- coding: utf8 -*-
'''
Run MRF method

Copyright (c) 2018, CNES
'''

import os
import logging
import argparse

import my_rdf_file as my_rdf

from names import MRF_RASTER_BASENAME, compute_name

from mrf_method import (get_grid_lat_lon_ref, get_grid_lat_lon_ref2, extract_params_from_pixc, apply_mrf_method,
                        get_labels, get_mean_dem, write_mrf_fpdem_file)

class MRF_run(object):
    """
    Class MRF_run
    Process the MRF method (Markov Random Field)
    """

    def __init__(self, param, output_file=None):

        if param:
            # AOI
            self.lat_max = float(param.getValue("latitude max").split(" ")[0])
            self.lon_max = float(param.getValue("longitude max").split(" ")[0])
            self.lat_min = float(param.getValue("latitude min").split(" ")[0])
            self.lon_min = float(param.getValue("longitude min").split(" ")[0])

            # PIXC directory
            self.pixcfiles = param.getValue("PIXC file list").split(" ")
            inputpixcfiles = os.popen("ls "+self.pixcfiles[0]).readlines()
            self.inputpixcfiles = []
            for i in range(0, len(inputpixcfiles)):
                self.inputpixcfiles.append(inputpixcfiles[i].rstrip('\n'))

            # Output
            self.output_path = param.getValue("output directory").split(" ")[0]
            if output_file is None:
                self.output_file = compute_name(self.output_path, MRF_RASTER_BASENAME,
                                                param.getValue("tile name"),
                                                param.getValue("first date name"),
                                                param.getValue("last date name"))
            else:
                self.output_file = output_file

            self.plot = param.getValue("plot").split(" ")[0]

            # Pre-processing of PIXC
            self.cross_track_min = float(param.getValue("cross track min value").split(" ")[0])
            self.threshold = float(param.getValue("threshold").split(" ")[0])

            # Raster
            self.resolution = float(param.getValue("resolution").split(" ")[0])

            # Reference DEM
            self.refdem_grid = bool(param.getValue("refdem grid").split(" ")[0])
            self.refdem_file = param.getValue("refdem file").split(" ")[0]

            # Margins in lon and lat for reference DEM
            try:
                margin_lon = float(param.getValue("margin lon").split(" ")[0])
                margin_lat = float(param.getValue("margin lat").split(" ")[0])
                self.margin = [margin_lon, margin_lat]
            except:
                pass

            # MRF parameters
            self.pixel_res_m = float(param.getValue("pixel res in m").split(" ")[0])
            self.tile_size_km = float(param.getValue("tile size in km").split(" ")[0])
            self.stride_km = float(param.getValue("stride in km").split(" ")[0])
            self.border_exclude_km = float(param.getValue("border exclude in km").split(" ")[0])
            self.land_law = param.getValue("land law").split(" ")[0]
            self.weight_h = float(param.getValue("weight h").split(" ")[0])
            self.max_iters = int(param.getValue("mrf max iterations").split(" ")[0])
            self.convergence = float(param.getValue("mrf convergence").split(" ")[0])

            # MRF sig0 initialization parameters
            sig0_mu_land = float(param.getValue("sig0 mu land").split(" ")[0])
            sig0_std_land = float(param.getValue("sig0 std land").split(" ")[0])
            sig0_mu_water = float(param.getValue("sig0 mu water").split(" ")[0])
            sig0_std_water = float(param.getValue("sig0 std water").split(" ")[0])

            # MRF h initialization parameters
            h_mu_land = float(param.getValue("h mu land").split(" ")[0])
            h_std_land = float(param.getValue("h std land").split(" ")[0])
            h_mu_water = float(param.getValue("h mu water").split(" ")[0])
            h_std_water = float(param.getValue("h std water").split(" ")[0])
            h_expon_k = float(param.getValue("h expon k").split(" ")[0])
            h_expon_loc = float(param.getValue("h expon lo").split(" ")[0])
            h_expon_scale = float(param.getValue("h expon scale").split(" ")[0])

            # Initial variables for MRF method for sig0 and height
            self.P_SIG0_INIT = {
                'mu_land': sig0_mu_land, 'std_land': sig0_std_land,
                'mu_water': sig0_mu_water, 'std_water': sig0_std_water
            }
            self.P_H_INIT = {
                'mu_land': h_mu_land, 'std_land': h_std_land,
                'mu_water': h_mu_water, 'std_water': h_std_water,
                'expon_k': h_expon_k, 'expon_loc': h_expon_loc, 'expon_scale': h_expon_scale
            }

            # MRF probability thresholds (i.e., higher probabilities than the threshold are kept)
            self.threshold_without_h = float(param.getValue("threshold without h").split(" ")[0])
            self.threshold_with_h = float(param.getValue("threshold with h").split(" ")[0])

            # Creating output directory
            if not os.path.isdir(self.output_path):
                os.mkdir(self.output_path)

    def mrf_compute(self):

        # ref_dem, self.latitude_ref_dem, self.longitude_ref_dem = get_grid_lat_lon_ref(self.refdem_grid,
        #                                                                     path_ref_dem=self.refdem_file,
        #                                                                     pixc=self.inputpixcfiles[0],
        #                                                                     resolution=self.resolution,
        #                                                                     margin=self.margin)

        if self.refdem_grid == False:
            extrema = [self.lon_min, self.lon_max, self.lat_min, self.lat_max]
            ref_dem, self.latitude_ref_dem, self.longitude_ref_dem = get_grid_lat_lon_ref2(self.refdem_grid,
                                                                                lonlat_extrema=extrema,
                                                                                resolution=self.resolution)
        else:
            ref_dem, self.latitude_ref_dem, self.longitude_ref_dem = get_grid_lat_lon_ref2(self.refdem_grid,
                                                                                path_ref_dem=self.refdem_file)

        fpdem = []
        for pathfile in self.inputpixcfiles:
            file = os.path.basename(pathfile)
            root = os.path.dirname(pathfile)
            (grid_h, grid_h_interp,
             grid_inc,
             grid_sig0, grid_sig0_interp,
             grid_coh_interp,
             mask,
             date,
             grid_coh_th_interp) = extract_params_from_pixc(file, root, ref_dem,
                                                            min_crosstrack=self.cross_track_min)

            fpdem.append([grid_h, grid_h_interp, grid_inc, grid_sig0, grid_sig0_interp,
                          grid_coh_interp, mask, date, grid_coh_th_interp])

        fpdem = sorted(fpdem, key=lambda x: x[7])
        proba_map_list = []
        for i in range(len(fpdem)):
            logging.info(f"Applying MRF method on PIXC {i + 1}/{len(fpdem)}")

            (height_cycle_0, sig0_cycle_0,
             proba_smooth_0, labels_smooth_0,
             proba_smooth_h, labels_smooth_h) = apply_mrf_method(
                fpdem[i],
                self.P_SIG0_INIT, self.P_H_INIT, self.land_law, self.max_iters, self.convergence, self.weight_h,
                self.pixel_res_m, self.tile_size_km, self.stride_km, self.border_exclude_km
            )

            proba_map_list.append([proba_smooth_0, labels_smooth_0, proba_smooth_h, labels_smooth_h])

        # Get the labels for each pixel to determine if it is either land (0) or water (1)
        labels_combined_list = []
        for i in range(len(fpdem)):
            labels_combined = get_labels(fpdem, i, proba_map_list,
                                         self.threshold_with_h, self.threshold_without_h)
            labels_combined_list.append(labels_combined)

        # Compute the mean elevation, filtering out bad dates
        self.mean_dem, self.mean_sig0, mean_coh, mean_coh_th, count = get_mean_dem(fpdem.copy(),
                                                                                   labels_combined_list.copy())

    def mrf_raster(self):
        epsg = "4326"
        write_mrf_fpdem_file(self.output_file, self.longitude_ref_dem, self.latitude_ref_dem, self.mean_dem, epsg)


# Main program
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=
                                     '''Compute MRF method bathymetry raster from multiple tiles of PIXC products.''')
    parser.add_argument("parameter_file", help="parameter_file (*.rdf)")
    args = parser.parse_args()

    level = getattr(logging, "INFO")
    logging.basicConfig(filename=None, format='%(asctime)s [%(levelname)s] %(message)s', level=level)

    parameters = my_rdf.myRdfReader(args.parameter_file)
    mrf = MRF_run(parameters)
    mrf.mrf_compute()
    mrf.mrf_raster()
    logging.info('Calculation of MRF method was performed')