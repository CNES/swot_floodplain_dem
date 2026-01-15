# -*- coding: utf8 -*-
'''
Create raster from floodplain dem pixel cloud

Copyright (c) 2018, CNES
'''

import os
import sys
import logging
import numpy as np
import geopandas as gpd
import xarray as xr
import shapely
from shapely.geometry import MultiPoint

import shp as shp

class Extract_Area(object):
    """
    Class Floodplain
    Main class to extract area of interest from set of tiles
    """

    def __init__(self, param, input_file=None, output_file=None, ch_ratio=0.005):
        """
        Constructor: initialize variables

        :param param: input parameters to run the processor
        :type param: dictionary
        :param input_file: Input filename with the FPDEM results (ungridded file)
        :param output_file: Output filename
        :param ch_ratio: concave hull ratio to extract point cloud contours
        """

        if input_file == None:
            self.input_file = param.getValue("input file")
        else:
            self.input_file = input_file

        if input_file == None:
            self.output_file = param.getValue("output file")
        else:
            self.output_file = output_file

        if param:
            # Extraction of polygons
            self.ch_ratio = float(param.getValue("concave hull ratio").split(" ")[0])
        else:
            self.ch_ratio = ch_ratio

    def load_input_extract_area(self):
        """
        Open the FPDEM point cloud file using xarray and convert it into a dataframe
        """
        cloud_xr = xr.open_dataset(self.input_file)
        self.input_cloud_df = cloud_xr.to_dataframe()

    def extract_area_polygons(self):
        """
        Get the labels of the different cluster in the FPDEM point cloud product
        and apply concave hull to get the contours
        """
        # Read dataframe with clusters label numbers
        output_dir = os.path.dirname(self.output_file)
        gdf_lab = gpd.read_file(os.path.join(output_dir, 'results_fpdem_labels_dbscan.shp'))
        res_data = self.input_cloud_df.copy()
        if len(res_data) != len(gdf_lab):
            logging.warning('This is weird!')
            logging.warning('The number of points in the dataframe from results_fpdem_labels_dbscan.shp')
            logging.warning(f'is not the same as the number of points in the ungridded FPDEM results file ({self.input_file}).')
            sys.exit()
        res_data['lab_dbscan'] = gdf_lab.lab_dbscan
        #
        N_cluster = len(np.unique(gdf_lab.lab_dbscan.values))
        logging.info(f'There are {N_cluster} clusters (on which to extract polygons for rasterization)')
        # Extract Polygons
        polygons = []
        for label in range(N_cluster):
            res_extract2 = res_data[res_data.lab_dbscan == label].copy()
            if len(res_extract2) > 0:
                # Get utm coordinates
                res_extract2.loc[:, 'geometry'] = gpd.points_from_xy(res_extract2.longitude, res_extract2.latitude)
                #res_extract2.loc[:, 'geometry_utm'] = gpd.points_from_xy(res_extract2.x, res_extract2.y)
                concave_hull = gpd.GeoSeries([MultiPoint(res_extract2['geometry'].values)]).concave_hull(ratio=self.ch_ratio,
                                                                                                         allow_holes=True)
            polygons.append(concave_hull.geometry.values[0])
        polygons = shapely.unary_union(gpd.GeoDataFrame(geometry=polygons, crs=4326))
        polygons = list(polygons.geoms)
        shp.polygons_to_file(self.output_file, polygons)
