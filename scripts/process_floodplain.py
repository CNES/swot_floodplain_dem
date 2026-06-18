"""
Create ply, shp and netcdf with FPDEM points from pixel cloud

Copyright (c) 2018, CNES
"""

import argparse
import os
import logging
import traceback
import copy
import pandas as pd
import geopandas as gpd
import numpy as np
import multiprocessing as mp
import utm
import shapely
from shapely.geometry import Point, MultiPoint
from functools import partial

import my_rdf_file as my_rdf
import my_hull as my_hull
import toply as ply
import toshp as shp

from nc import PixcReader, write_raster_ungridded

from names import FPDEM_BASENAME, FPDEM_POINTCLOUD_BASENAME, compute_name

from Pekel_Sword_PLD_intersect import (get_tiles_geom, get_PLD_mask, get_Pekel_mask, extract_Pekel_polygon,
                                       get_reach_node_info, get_poly_sword_from_reach)

from process import (pre_processing_pixc, get_range_azimuth_extrema, prepare_clustering_image,
                     find_attributes, remove_borders, filter_data_based_on_quality_flag,
                     final_filtering_FPDEM_results, valid_date)

from water_or_land import (compute_label_and_remove_small_object, compute_subwater_extract_from_label,
                           find_body_category)

from clustering_method import Clustering_Method


class Floodplain(object):
    """
    Class Floodplain
    Main class to run many Floodplain processes
    """

    def __init__(self, param):
        """
        Constructor: initialize variables

        :param in_params: input parameters to run the processor
        """

        if param:
            # AOI
            self.lat_max = float(param.getValue("latitude max").split(" ")[0])
            self.lon_max = float(param.getValue("longitude max").split(" ")[0])
            self.lat_min = float(param.getValue("latitude min").split(" ")[0])
            self.lon_min = float(param.getValue("longitude min").split(" ")[0])

            # PIXC and PIXCVEC directory and names
            self.pixcfiles = param.getValue("PIXC file list").split(" ")
            self.vecfiles = param.getValue("PIXCVec file list").split(" ")

            # Output
            self.output_path = param.getValue("output directory").split(" ")[0]
            self.tile_name = param.getValue("tile name").split(" ")[0]
            self.first_date_name = param.getValue("first date name").split(" ")[0]
            self.last_date_name = param.getValue("last date name").split(" ")[0]

            # FPDEM approach
            self.approach = param.getValue("select_approach").split(" ")[0]

            # Pekel
            self.pekel_path = param.getValue("Pekel path").split(" ")[0]

            # SWORD
            self.sword_path = param.getValue("Sword path").split(" ")[0]
            self.hydrobasin_lv1_file = param.getValue("hydrobasin nv1 file").split(" ")[0]
            self.hydrobasin_lv2_file = param.getValue("hydrobasin nv2 file").split(" ")[0]

            # SWOT science tiles: needed to filter SWORD reaches
            self.swot_tiles_file = param.getValue("swot tiles file").split(" ")[0]

            # Mask information
            try:
                self.mask_flag = int(param.getValue("use mask").split(" ")[0])
            except:
                self.mask_flag = 0
            try:
                self.mask_buffer = float(param.getValue("mask buffer size").split(" ")[0])
            except:
                self.mask_buffer = None
            try:
                self.lake_id = int(param.getValue("lake id").split(" ")[0])
                self.pld_path = param.getValue("PLD path").split(" ")[0]
            except:
                self.lake_id = None
                self.pld_path = None

            # Parallel run or not
            self.option = param.getValue("threading option")[0]
            self.nb_cpu = int(param.getValue("number of cpu")[0])

            # Plot
            self.plot = param.getValue("plot").split(" ")[0]

            # Pre-processing of PIXC
            self.cross_track_min = float(param.getValue("cross track min value").split(" ")[0])
            self.threshold = float(param.getValue("threshold").split(" ")[0])

            # Clustering method
            self.cluster_filter_method = param.getValue("clustering outlier filter method").split(" ")[0]
            self.cluster_method_choice = param.getValue("clustering method").split(" ")[0]
            self.clustering_contour = param.getValue("clustering water contour").split(" ")[0]

            # Filtering
            # Pekel filtering
            self.filtering_pekel_val_DW = int(param.getValue("pekel percent for Dark Water").split(" ")[0])
            self.filtering_pekel_end = param.getValue("pekel filtering end").split(" ")[0]
            try:
                self.filtering_pekel_val = int(param.getValue("pekel filtering value").split(" ")[0])
            except:
                self.filtering_pekel_val = None
            # Quality flags
            self.classif_qual = int(param.getValue("classification_qual").split(" ")[0])
            self.geoloc_qual = int(param.getValue("geolocation_qual").split(" ")[0])
            self.sig_qual = int(param.getValue("sig0_qual").split(" ")[0])
            # Parameters for removing isolated points
            self.d_ngbr = float(param.getValue("distance to neighbors").split(" ")[0])
            self.n_ngbr = int(param.getValue("number of neighbors").split(" ")[0])
            # Labelling water bodies parameters
            self.body_min_size = int(param.getValue("water body min size").split(" ")[0])
            self.body_connectivity = int(param.getValue("connectivity").split(" ")[0])

            # Extraction of polygons
            self.ch_ratio = float(param.getValue("concave hull ratio").split(" ")[0])

            #
            # Get the list of PIXC files and PIXCVEC files
            inputpixcfiles = os.popen("ls "+self.pixcfiles[0]).readlines()
            try:
                inputvecfiles = os.popen("ls "+self.vecfiles[0]).readlines()
            except:
                inputvecfiles = []
            self.inputpixcfiles = []
            self.inputvecfiles = []
            list_tile_info = []
            for i in range(0, len(inputpixcfiles)):
                find = False
                for k in range(0, len(inputvecfiles)):
                    if ((inputpixcfiles[i].rstrip('\n').split("_PIXC_")[1].split("_P")[0]
                         == (inputvecfiles[k].rstrip('\n')).split("_PIXCVec_")[1].split("_P")[0])
                            and (find == False)):
                        self.inputpixcfiles.append(inputpixcfiles[i].rstrip('\n'))
                        self.inputvecfiles.append(inputvecfiles[k].rstrip('\n'))
                        find = True
                if not find:
                    date_pixcvec = inputpixcfiles[i].rstrip('\n').split("_PIXC_")[1].split("_PI")[0]
                    logging.info(f"PIXCVec file is missing for date {date_pixcvec} => replaced by PIXC")
                    self.inputpixcfiles.append(inputpixcfiles[i].rstrip('\n'))
                    self.inputvecfiles.append(inputpixcfiles[i].rstrip('\n'))
                if inputpixcfiles[i].split('/')[-1][20:28] not in list_tile_info:
                    list_tile_info.append(inputpixcfiles[i].split('/')[-1][20:28])
            if len(self.inputpixcfiles) != len(inputpixcfiles):
                logging.warning(f"{len(inputpixcfiles)-len(self.inputpixcfiles)} PIXCVec file is missing")

            # Get geometry of pixc tiles
            logging.info('Get the PIXC tiles geometry')
            self.pixc_tiles_geom = get_tiles_geom(list_tile_info, self.swot_tiles_file)

            # Get Pekel info and polygons
            logging.info('Read Pekel info')
            self.data_pekel = get_Pekel_mask(self.pekel_path, self.pixc_tiles_geom.geometry.values[0])
            logging.info('Get Pekel polygon: all occurrences')
            self.pekel_0_100_poly = extract_Pekel_polygon(self.data_pekel,
                                                          occ_min=0, occ_max=100,
                                                          bufsize=0.)

            # Pekel polygon for filtering Dark Water points
            try:
                logging.info(f'Get Pekel polygon for filtering Dark Water in PIXC: occurrence > {self.filtering_pekel_val_DW}%')
                self.pekel_X_100_poly = extract_Pekel_polygon(self.data_pekel,
                                                              occ_min=self.filtering_pekel_val_DW, occ_max=100,
                                                              bufsize=0.)
            except:
                logging.warning(f"    No points in Pekel occurrences > {self.filtering_pekel_val_DW}% for this region")
                self.pekel_X_100_poly = None

            # Pekel polygon for filtering the FPDEM points at the end of the bathy extraction
            self.pekel_X2_100_poly = None
            if self.filtering_pekel_end == 'yes':
                try:
                    logging.info(f'Get Pekel polygon for filtering bathy at the end: occurrence > {self.filtering_pekel_val}%')
                    self.pekel_X2_100_poly = extract_Pekel_polygon(self.data_pekel,
                                                                   occ_min=self.filtering_pekel_val, occ_max=100,
                                                                   bufsize=0.)
                except:
                    logging.warning(f"    No points in Pekel occurrences > {self.filtering_pekel_val}% for this region")
                    self.pekel_X2_100_poly = None

            # SWORD info
            try:
                logging.info('Read SWORD reaches and nodes info')
                self.df_reach_data, self.df_node_data = get_reach_node_info(self.pixc_tiles_geom,
                                                                            self.sword_path, self.hydrobasin_lv1_file,
                                                                            self.hydrobasin_lv2_file)
                logging.info('Get SWORD reaches polygon')
                self.poly_sword = get_poly_sword_from_reach(self.df_reach_data, buffer_size=0.)
            except:
                self.poly_sword = None

            # Create the polygon mask needed for intersection with PIXC to limit the number of points
            if self.mask_flag == 1:
                logging.info('PLD Mask intersection will be performed')
                self.polygon_mask = get_PLD_mask(self.lake_id, self.pld_path, self.inputpixcfiles[0])
            elif self.mask_flag == 2:
                logging.info('Pekel Mask intersection will be performed')
                if self.mask_buffer == 0.:
                    self.polygon_mask = copy.copy(self.pekel_0_100_poly)
                else:
                    self.polygon_mask = extract_Pekel_polygon(self.data_pekel,
                                                              occ_min=0, occ_max=100,
                                                              bufsize=self.mask_buffer)
            elif self.mask_flag == 3:
                logging.info('Sword Mask intersection will be performed')
                self.polygon_mask = get_poly_sword_from_reach(self.df_reach_data, buffer_size=self.mask_buffer)
            else:
                self.polygon_mask = None

            logging.info(f"{len(self.inputvecfiles)} PIXC/PIXCVec couple files will be processed")

            # Creating output directory
            if not os.path.isdir(self.output_path):
                os.mkdir(self.output_path)


    # Start the bathy extraction either with multiprocessing or not
    def compute_fpdem_pointcloud_boundaries(self):
        """
        Extract bathymetry FPDEM
        """

        res_data = pd.DataFrame()

        # Multiprocessing option
        if self.option == "multiprocessing":
            # Create multiprocessing Pool
            logging.info("multiprocessing")
            cpu_number = self.nb_cpu
            logging.info("mp.cpu_count() = ", cpu_number)
            pool = mp.Pool(cpu_number)

            # Multiprocessing arguments list preparation
            args = []
            for pixc_file, vec_file in zip(self.inputpixcfiles, self.inputvecfiles):
                args.append([pixc_file, vec_file])

            res = pool.starmap(partial(self.process_data), args)
            pool.close()
            pool.join()

            # Concatenate all PIXC FPDEM results
            for i in range(len(res)):
                res_data = pd.concat([res_data, res[i]])

        # No multiprocessing
        else:
            for pixc_file, vec_file in zip(self.inputpixcfiles, self.inputvecfiles):
                result_data = self.process_data(pixc_file, vec_file)
                res_data = pd.concat([res_data, result_data])

        # Some final filtering on the FPDEM results
        res_data = final_filtering_FPDEM_results(res_data, self.filtering_pekel_end, self.pekel_X2_100_poly,
                                                 self.d_ngbr, self.n_ngbr, self.output_path)

        self.res_pointcloud = res_data

    #
    def process_data(self, pixc_file, vec_file):
        """
        Main routine, processing the PIXC and extracting the bathymetry points

        :param pixc_file: list of PIXC
        :param vec_file: list of PIXCVec
        """

        try:
            # Get cycle and create directory
            cycle = os.path.basename(pixc_file)[16:28]
            if os.path.isdir(os.path.join(self.output_path, f'cycle{cycle}')) is False:
                os.mkdir(os.path.join(self.output_path, f'cycle{cycle}'))

            pixc_reader = PixcReader(self.polygon_mask, pixc_file, vec_file)

            water, min_range_ind_to_remove = pre_processing_pixc(pixc_reader,
                                                                 self.cross_track_min, self.threshold)

            (label_tab, count, classification_tab,
             height_tab, sig0_tab,
             azimuth_index_tab, range_index_tab,
             latitude_tab, longitude_tab) = compute_label_and_remove_small_object(water,
                                                                                  pixc_reader.range_size,
                                                                                  pixc_reader.azimuth_size,
                                                                                  self.pekel_X_100_poly,
                                                                                  self.body_min_size,
                                                                                  self.body_connectivity,
                                                                                  plot=self.plot,
                                                                                  outpath=self.output_path,
                                                                                  cycle=cycle)
            logging.info(f'Number of water bodies: {count}')

            fpdem_land_pixel = pd.DataFrame()
            for label in range(1, count+1):
                logging.info(f'Water body number {label} / {count}')

                # Return water_extract (-> GeoDataFrame)
                water_extract = compute_subwater_extract_from_label(water, label_tab,
                                                                    azimuth_index_tab, range_index_tab, label)
                logging.info(f'    Number of points in label: {len(water_extract)}')
                if len(water_extract) == 0:
                    logging.warning('No points found in this water body!')
                    continue

                if self.approach == 'mixed':
                    water_extractCateg = water_extract[(water_extract.classification == 3) |
                                                       (water_extract.classification == 4)]
                    water_extractCateg = water_extractCateg.loc[(water_extractCateg.sig0_qual == 0)]
                    if len(water_extractCateg) == 0:
                        logging.warning('No points of classification 3 or 4 found in this water body')
                        logging.warning('    so no categorization possible!')
                        continue

                    water_body_type, categ_params = find_body_category(water_extractCateg,
                                                                       self.pekel_0_100_poly, self.poly_sword,
                                                                       plot=self.plot,
                                                                       choices=['WATER', 'WATER_LAND', 'LAND'],
                                                                       cycle=cycle,
                                                                       label=label, output_path=self.output_path)
                    logging.info(f'Body type: {water_body_type}')

                    # Apply different method depending on body category
                    if water_body_type == "LAND":
                        fpdem_land_pixel = pd.concat([fpdem_land_pixel, water_extract[['longitude', 'latitude',
                                                                                       'height', 'sig0',
                                                                                       'classification',
                                                                                       'azimuth_index',
                                                                                       'range_index']]])
                    elif water_body_type == "WATER":
                        logging.info('Bathtub ring method is applied')
                        res_bathtub = self.compute_bathtub_method(water_extract)
                        fpdem_land_pixel = pd.concat([fpdem_land_pixel, res_bathtub])

                    elif water_body_type == "WATER_LAND":
                        logging.info('Clustering method is applied')

                        # Over one cycle (selected or first one of the list) extract height
                        # to obtain range and azimuth extrema
                        l0, l1, c0, c1 = get_range_azimuth_extrema(water_extract, pixc_reader.range_size,
                                                                   pixc_reader.azimuth_size)
                        # Create 4D matrix with all variables used for clustering (h, sig0, lon, lat)
                        sub_image = prepare_clustering_image(height_tab, sig0_tab, latitude_tab, longitude_tab,
                                                             label_tab, label, l0, l1, c0, c1)

                        clust = Clustering_Method(sub_image, label, self.output_path, cycle, plot=self.plot)
                        clust.normalize_data(filter_method=self.cluster_filter_method) # isolation_forest/zscore
                        if self.cluster_method_choice == 'birch' or self.cluster_method_choice == 'kmeans':
                            clust.clustering_method(method=self.cluster_method_choice)
                        elif self.cluster_method_choice == 'hdbscan_tsne':
                            clust.tsne()
                            clust.clustering_tsne_hdbscan()
                        else:
                            clust.umap()
                            clust.clustering_umap_hdbscan()
                        clust.get_water_soil_labels(water_extract, self.pekel_0_100_poly, self.poly_sword,
                                                    method=self.cluster_method_choice)

                        if clust.sub_img_labeled_water is not None:
                            geom = gpd.points_from_xy(clust.sub_img_labeled_water[:, 2],
                                                      clust.sub_img_labeled_water[:, 3])
                            water_extract_water = water_extract[water_extract.geometry.isin(geom)]
                            if self.clustering_contour == 'bathtub':
                                res_bathtub = self.compute_bathtub_method(water_extract_water)
                            elif self.clustering_contour == 'concavehull':
                                # Alternate contours extraction because bathtub returns weird contours
                                # probably because not a lot of points present for some labels
                                concave_hull = gpd.GeoSeries([
                                                            MultiPoint(water_extract_water['geometry'].values)
                                                            ]).concave_hull(ratio=0.05, allow_holes=True)
                                list_coords = shapely.get_coordinates(concave_hull.exterior).tolist()
                                res_bathtub = gpd.GeoDataFrame(geometry=[Point(i) for i in list_coords], crs=4326)
                                res_bathtub['longitude'] = pd.Series(res_bathtub.geometry.get_coordinates().x.values)
                                res_bathtub['latitude'] = pd.Series(res_bathtub.geometry.get_coordinates().y.values)
                                col_list = ['height', 'sig0', 'azimuth_index', 'range_index', 'classification']
                                res_bathtub[col_list] = res_bathtub.apply(find_attributes, args=(water, 2,), axis=1)
                                res_bathtub = res_bathtub.drop(['geometry'], axis=1)

                            fpdem_land_pixel = pd.concat([fpdem_land_pixel, res_bathtub])

                        if clust.sub_img_labeled_land is not None:
                            geom = [Point(lon, lat) for lon, lat in zip(clust.sub_img_labeled_land[:, 2],
                                                                        clust.sub_img_labeled_land[:, 3])]
                            water_extract_land = water_extract[water_extract.geometry.isin(geom)]
                            fpdem_land_pixel = pd.concat([fpdem_land_pixel,
                                                          water_extract_land[['longitude', 'latitude',
                                                                              'height', 'sig0', 'classification',
                                                                              'azimuth_index', 'range_index']]])


                elif self.approach == 'bathtub':
                    res_bathtub = self.compute_bathtub_method(water_extract)
                    fpdem_land_pixel = pd.concat([fpdem_land_pixel, res_bathtub])

            # Remove borders points
            range_min = np.min(fpdem_land_pixel.range_index)
            range_max = np.max(fpdem_land_pixel.range_index)
            fpdem_land_pixel = remove_borders(fpdem_land_pixel.copy(), range_min, range_max - 1,
                                              pixc_reader.azimuth_min, pixc_reader.azimuth_max - 1,
                                              min_range_ind_to_remove)

            # Filter according to user defined lat-lon box
            try:
                fpdem_land_pixel = fpdem_land_pixel.loc[(fpdem_land_pixel['longitude'] >= self.lon_min) &
                                                        (fpdem_land_pixel['longitude'] <= self.lon_max) &
                                                        (fpdem_land_pixel['latitude'] >= self.lat_min) &
                                                        (fpdem_land_pixel['latitude'] <= self.lat_max)]
            except NameError:
                logging.warning('self.lon_min, self.lon_max, self.lat_min and/or self.lat_max do not exist')

            # Add flags columns + time and elevation
            fpdem_land_pixel[['classification_qual',
                              'geolocation_qual',
                              'sig0_qual',
                              'time',
                              'elevation']] = fpdem_land_pixel.apply(find_attributes, args=(water, 1,), axis=1)

            # Flag filtering
            fpdem_land_pixel = filter_data_based_on_quality_flag(fpdem_land_pixel,
                                                                 self.classif_qual,
                                                                 self.geoloc_qual,
                                                                 self.sig_qual)

            # Get utm coordinates
            logging.info('Converting to UTM')
            lambdafunc = lambda row: pd.Series(
                [*utm.from_latlon(row['latitude'], row['longitude'])[0:2], row['height']])
            fpdem_land_pixel[['x', 'y', 'z']] = fpdem_land_pixel.apply(lambdafunc, axis=1)

            # Get geometry and write file with FPDEM results for each PIXC: intermediary output
            # => TODO: to be deleted later
            logging.info('Create intermediary results (i.e., FPDEM results for each PIXC)')
            geom = gpd.points_from_xy(fpdem_land_pixel.longitude, fpdem_land_pixel.latitude)
            fpdem_land_pixel = gpd.GeoDataFrame(fpdem_land_pixel, geometry=geom, crs=water.crs)
            filename_temp = f'{os.path.basename(pixc_file).replace("_PIXC", "").replace(".nc", "")}_resFPDEM.shp'
            fpdem_land_pixel.to_file(os.path.join(self.output_path, f'cycle{cycle}', f'{filename_temp}'))

        except:
            traceback.print_exc()
            logging.warning(f"Error in files {pixc_file} and {vec_file}")

        # Return
        logging.info(f"Process file: {pixc_file}")
        try:
            return fpdem_land_pixel
        except:
            return pd.DataFrame()

    def compute_bathtub_method(self, water):
        """
        Recover all points present on the water contour obtained with the bathtub ring method

        :param water: dataframe with PIXC points info
        :return: dataframe with contours points, polygons
        """
        in_v_classif = water["classification"].values
        in_height = water["height"].values
        in_sig0 = water["sig0"].values
        in_azimuth = water["azimuth_index"].values.astype('int')
        in_range = water["range_index"].values.astype('int')
        in_v_lat = water["latitude"].values
        in_v_long = water["longitude"].values

        try:
            # TODO: need to debug 2.1 option
            lake_poly, multi_list_of_points = my_hull.get_polygon_from_binar_image(in_range, in_azimuth, in_v_long,
                                                                                   in_v_lat, in_height, in_sig0,
                                                                                   in_v_classif, 2.0)

            if len(multi_list_of_points) > 1:
                list_of_points = [x for xs in multi_list_of_points for x in xs]
            else:
                list_of_points = multi_list_of_points[0]

            df = pd.DataFrame(list_of_points, columns=['longitude', 'latitude', 'height', 'sig0',
                                                       'azimuth_index', 'range_index', 'classification'])

        except:
            logging.warning("Issue with label")
            df = gpd.GeoDataFrame()

        return df

    def write_fpdem_pointcloud_output(self):
        """
        Write the FPDEM results into a netcdf (point clouds format)
        """
        # Combine outputs into on product
        if len(self.res_pointcloud) > 0:
            try:

                logging.info("Writing the ungridded netcdf with FPDEM results")

                # Add valid_date
                res_pointcloud = valid_date(self.res_pointcloud)

                outputfile_root = os.path.join(self.output_path, FPDEM_BASENAME)

                ply.gdf_to_file(f"{outputfile_root}.ply", res_pointcloud, mode="text")

                # Also write shp file
                shp.gdf_to_file(f"{outputfile_root}.shp", res_pointcloud, index=True)

                res_pointcloud_ds = res_pointcloud[['longitude', 'latitude', 'elevation', 'x', 'y', 'z',
                                                    'fpdem_ungridded_qual', 'time', 'lab_dbscan']].to_xarray()

                zone_number = utm.latlon_to_zone_number(res_pointcloud.iloc[0]['latitude'],
                                                        res_pointcloud.iloc[0]['longitude'])
                if np.mean(res_pointcloud.iloc[0]['latitude']) >= 0:
                    res_pointcloud_ds.attrs['epsg'] = int(32600 + zone_number)
                else:
                    res_pointcloud_ds.attrs['epsg'] = int(32700 + zone_number)

                list_pixc = []
                list_pixcvec = []
                for k in self.inputpixcfiles:
                    list_pixc.append(os.path.basename(k))
                for k in self.inputvecfiles:
                    list_pixcvec.append(os.path.basename(k))

                res_pointcloud_ds.attrs['xref_input_l2_hr_pixc_files'] = list_pixc
                res_pointcloud_ds.attrs['xref_input_l2_hr_pixcvec_files'] = list_pixcvec

                output_fpdem_pointcloud_name = compute_name(self.output_path, FPDEM_POINTCLOUD_BASENAME, self.tile_name,
                                                            self.first_date_name, self.last_date_name)

                write_raster_ungridded(res_pointcloud_ds, output_fpdem_pointcloud_name)

            except:
                traceback.print_exc()
                logging.error(f"Error writing output file")
            else:
                logging.info("Write output file: OK")
        else:
            logging.warning("Output is empty for this dataset")

# Main program
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=
                                     '''Compute fpdem intermediate product from multiple tiles of PIXC products 
                                     and their associated PIXCVecRiver products.''')
    parser.add_argument("parameter_file", help="parameter_file (*.rdf)")
    args = parser.parse_args()

    level = getattr(logging, "INFO")
    logging.basicConfig(filename=None, format='%(asctime)s [%(levelname)s] %(message)s', level=level)

    parameters = my_rdf.myRdfReader(args.parameter_file)
    fpdem = Floodplain(parameters)
    fpdem.compute_fpdem_pointcloud_boundaries()
    fpdem.write_fpdem_pointcloud_output()
    logging.info('Calculation of FPDEM was performed')
