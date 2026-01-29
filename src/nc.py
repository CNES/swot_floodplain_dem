# -*- coding: utf8 -*-
"""
Reader for netcdf file
Convert netcdf files to geopandas dataframe
Pixel cloud reader using geopandas

Copyright (c) 2018 CNES. All rights reserved.
"""

import os
import logging
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
import math
import textwrap
import numpy as np
from netCDF4 import Dataset
from pyproj import CRS
try:
    from osgeo import osr
except ImportError:
    import osr

from Pekel_Sword_PLD_intersect import intersect_pixc_pixcvec_mask

EARTH_RADIUS = 6371000.

class PixcReader():

    def __init__(self, polygon_mask, pixc: str, vec: str = None):
        """
        Read pixel cloud file and vec file.
        
        :param pixc: filename of the pixel cloud file
        :param vec: filename of the vec file
        """

        logging.info(f'PIXC: {pixc}')
        
        # Read pixel cloud and convert to dataframe
        dnc = xr.open_dataset(pixc, group="pixel_cloud", decode_times=False)
        pixc_line_qual = dnc["pixc_line_qual"].values
        dnc = dnc.drop_dims('complex_depth')
        
        try:
            dnc = dnc.drop_dims('num_pixc_lines')
        except:
            dnc = dnc
                
        df = dnc.to_dataframe()

        df = df[['classification', 'pixel_area',
                 'longitude', 'latitude', 'height',
                 'range_index', 'azimuth_index',
                 'inc', 'cross_track', 'pole_tide', 'load_tide_got', 'load_tide_fes',
                 'solid_earth_tide', 'geoid', 'illumination_time',
                 'sig0', 'sig0_qual', 'geolocation_qual', 'classification_qual']]

        # Keep only water pixel point
        df = df.loc[(df.classification > 1.0)]

        # Mask on PIXC
        if polygon_mask is not None:
            df = intersect_pixc_pixcvec_mask(df, polygon_mask)

        # Read vec file and convert to dataframe
        """
        if vec:
            vec_dnc = xr.open_dataset(vec,decode_times=False)  
            vec_df = vec_dnc.to_dataframe()
            vec_df[['index_i']] = vec_df[['range_index']].astype(int)
            vec_df[['index_j']] = vec_df[['azimuth_index']].astype(int)
            vec_df = vec_df[['index_i', 'index_j', 'longitude_vectorproc',
                             'latitude_vectorproc', 'height_vectorproc']]
            vec_df.rename(columns={'longitude_vectorproc': 'longitude', 
                    'latitude_vectorproc': 'latitude',
                    'height_vectorproc': 'height'}, inplace=True)
            vec_df = vec_df.set_index(['index_i', 'index_j'])
            # Mask on Pixcvec
            if polygon_mask is not None:
                vec_df.dropna(inplace=True)
                vec_df = intersect_pixc_pixcvec_mask(vec_df, polygon_mask)

            df = df.drop(['longitude','latitude', 'height'], axis=1)
            df[['index_i']] = df[['range_index']].astype(int)
            df[['index_j']] = df[['azimuth_index']].astype(int)
            df = df.set_index(['index_i', 'index_j'])
            
            df = pd.concat([df, vec_df], axis=1, join='inner')
            df = df.reset_index()
            df = df.drop(['index_i', 'index_j'], axis=1)
        """

        # Create geodataframe
        geom = [Point(x,y) for x, y in zip(df['longitude'], df['latitude'])]
        self.data = gpd.GeoDataFrame(df, geometry=geom)
        # Drop Nan values
        self.data.dropna(inplace=True)
        # Longitude must be set between -180 and 180
        self.data['longitude'] = self.data.apply(lambda row: row.longitude if row.longitude < 180.0 \
                                                                            else row.longitude - 360.0,
                                                 axis=1)
        self.data[['range_index']] = self.data[['range_index']].astype(int)
        self.data[['azimuth_index']] = self.data[['azimuth_index']].astype(int)
         
        # Add information (range spacing)
        attrs_globals = xr.open_dataset(pixc).attrs
        nominal_range_spacing = attrs_globals["nominal_slant_range_spacing"]
        self.data['range_spacing'] = self.data.apply(lambda row: nominal_range_spacing/math.sin(row['inc']*np.pi/180.),
                                                     axis=1)

        # Compute wse using heights and corrections
        self.data['elevation'] = self.data.apply(lambda row: row['height'] - row['pole_tide'] -
                                            row['load_tide_got'] - row['load_tide_fes'] -
                                            row['solid_earth_tide'] - row['geoid'], axis=1)

        # Get time attribute
        self.data['time'] = self.data.apply(lambda row: round(float(row['illumination_time'])/3600.), axis=1)
                        
        # Get attributes
        self.azimuth_min = np.min(np.where(pixc_line_qual == 0))
        self.azimuth_max = np.max(np.where(pixc_line_qual == 0))
        self.range_min = 0
        self.range_max = dnc.attrs['interferogram_size_range']
        self.azimuth_size = dnc.attrs['interferogram_size_azimuth']
        self.range_size = dnc.attrs['interferogram_size_range']
        
        # Read orbit and convert to dataframe
        dnc_trj = xr.open_dataset(pixc, group="tvp")
        df_trj = dnc_trj.to_dataframe()
        d_az_trj = np.sqrt((df_trj[['x']].values[0] - df_trj[['x']].values[1])**2
                           + (df_trj[['y']].values[0] - df_trj[['y']].values[1])**2
                           + (df_trj[['z']].values[0] - df_trj[['z']].values[1])**2)
        alt = np.sqrt(df_trj[['x']].values[0]**2 + df_trj[['y']].values[0]**2 + df_trj[['z']].values[0]**2)
        d_az_ground = d_az_trj * EARTH_RADIUS / alt
        self.along_track_sampling = d_az_ground

    def get_data(self) -> gpd.GeoDataFrame:
        return self.data.copy()

def textjoin(text):
    """ Dedent join and strip text """
    text = textwrap.dedent(text)
    text = text.replace('\n', ' ')
    text = text.strip()
    return text

def write_raster_ungridded(data, filename):
    """
    Write the file with FPDEm point clouds results

    :param data: dataframe with points information
    :param filename: name of the output file
    :return:
    """

    logging.info('Writing FPDEM pointcloud file')

    epsg = data.attrs['epsg']

    ds = xr.Dataset(
        data_vars={
            "latitude": (("index",), data.variables['latitude'].values),
            "longitude": (("index",), data.variables['longitude'].values),
            "elevation": (("index",), data.variables['elevation'].values),
            "time": (("index",), data.variables['time'].values),
            "x": (("index",), data.variables['x'].values),
            "y": (("index",), data.variables['y'].values),
            "fpdem_ungridded_qual": (("index",), data.variables['fpdem_ungridded_qual'].values),
        },
        coords={"index": ("index", np.arange(len(data.variables['latitude'].values))), }
    )

    # Assign astype
    ds = ds.astype({
        "latitude": "float64",
        "longitude": "float64",
        "elevation": "float32",
        "time": "float64",
        "x": "float64",
        "y": "float64",
        "fpdem_ungridded_qual": "int32",
    })

    # Define CRS
    ds["crs"] = xr.DataArray(
        0,
        attrs={
            "grid_mapping_name": "latitude_longitude",
            "longitude_of_prime_meridian": 0.0,
            "semi_major_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
            "spatial_ref": (
                'GEOGCS["WGS 84",'
                'DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
                'PRIMEM["Greenwich",0],'
                'UNIT["degree",0.0174532925199433]]'
            ),
            "crs_wkt": CRS.from_epsg(epsg).to_wkt(),
            "epsg_code": f"EPSG:{epsg}",
        },
    )

    file = os.path.basename(filename)
    # Add global attributes
    ds.attrs.update({
        "Conventions": "CF-1.7",
        "title": "Level 2 KaRIn High Rate FPDEM Ungridded Data Product",
        "institution": "CNES",
        "source": "Large scale Simulator",
        "platform": "SWOT",
        "history": "None",
        "references": "None",
        "reference_document": "None",
        "contact": "damien.desroches@cnes.fr",
        "coordinate_reference_system": f"{epsg}",
        "short_name": "L2_HR_FPDEM_Ungridded",
        "descriptor_string": file.split('_')[5],
        "crid": "Dx0000",
        "product_version": "1",
        "pge_name": "Pge_v0",
        "pge_version": "1",

        "time_coverage_start": file.split('_')[6],
        "time_coverage_end": file.split('_')[7],

        "geospatial_lat_min": np.min(data.variables["latitude"].values),
        "geospatial_lat_max": np.max(data.variables["latitude"].values),
        "geospatial_lon_min": np.min(data.variables["longitude"].values),
        "geospatial_lon_max": np.max(data.variables["longitude"].values),
    })

    # Add necessary attributes
    ds["elevation"] = ds["elevation"].assign_attrs(
        grid_mapping="crs",
        long_name="surface elevation above geoid",
        units="m",
        valid_min=-1500,
        valid_max=15000,
    )

    ds["time"] = ds["time"].assign_attrs(
        grid_mapping="crs",
        long_name="Measurement time in hours (UTC) -- Time of measurement in hours in the UTC time scale since 1 Jan 2000 00:00:00 UTC, obtained by dividing the illumination_time (in seconds) of the pixel in the L2_HR_PIXC product by 3600, and rounding it to entire hours",
        units="h",
        valid_min=175320,
        valid_max=438300,
    )

    ds["x"] = ds["x"].assign_attrs(
        grid_mapping="crs",
        long_name="x coordinates (UTM)",
        units="m",
        valid_min=166021,
        valid_max=441868,
    )

    ds["y"] = ds["y"].assign_attrs(
        grid_mapping="crs",
        long_name="y coordinates (UTM)",
        units="m",
        valid_min=1116915,
        valid_max=8883085,
    )

    ds["fpdem_ungridded_qual"] = ds["fpdem_ungridded_qual"].assign_attrs(
        grid_mapping="crs",
        long_name="ungridded floodplain DEM quality flag",
        units="",
        flag_meanings='good suspect bad',
        flag_values='0 1 2',
        valid_min=0,
        valid_max=2,
    )

    # Write the netcdf file
    ds.to_netcdf(filename)

def write_raster_gridded(filename: str, mode: str, x: np.array, y: np.array,
                         out_image: np.array, out_dist_min_2d: np.array,
                         out_dist_mean_2d: np.array, qual_flag: np.array,
                         resolution: float, epsg: str,
                         zone_number=1, zone_letter='N'):
    """
    Write output raster netcdf file

    :param zone_letter:
    :param filename: Name of the raster file
    :param mode: latlon or utm
    :param x: Array of x (longitude)
    :param y: Array of y (latitude)
    :param out_image:
    :param out_dist_min_2d: Minimum of the distance between neighbors
    :param out_dist_mean_2d: Mean distance between neighbors
    :param qual_flag: Array of raster pixels' quality flag
    :param epsg: String with the EPSG number
    """

    logging.info('Writing FPDEM raster file')

    # Define the dataset
    if mode == 'latlon':

        ds = xr.Dataset(
            {
                "elevation": (("latitude", "longitude"), out_image),
                "distance_to_closest": (("latitude", "longitude"), out_dist_min_2d),
                "mean_distance": (("latitude", "longitude"), out_dist_mean_2d),
                "fpdem_gridded_qual": (("latitude", "longitude"), qual_flag),
            },
            coords={
                "latitude": (("latitude"), y),
                "longitude": (("longitude"), x),
            },
        )

        # Coordinates
        ds = ds.assign_coords(
            latitude=ds.latitude.assign_attrs(
                standard_name="latitude",
                long_name="latitude coordinate (positive N, negative S)",
                units="degrees_north",
                valid_min=-80,
                valid_max=80,
            ),
            longitude=ds.longitude.assign_attrs(
                standard_name="longitude",
                long_name="longitude coordinate (degrees East)",
                units="degrees_east",
                valid_min=-180,
                valid_max=180,
            ),
        )

        # Define CRS
        ds["crs"] = xr.DataArray(
            0,
            attrs={
                "grid_mapping_name": "latitude_longitude",
                "longitude_of_prime_meridian": 0.0,
                "semi_major_axis": 6378137.0,
                "inverse_flattening": 298.257223563,
                "spatial_ref": (
                    'GEOGCS["WGS 84",'
                    'DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
                    'PRIMEM["Greenwich",0],'
                    'UNIT["degree",0.0174532925199433]]'
                ),
                "crs_wkt": CRS.from_epsg(epsg).to_wkt(),
                "epsg_code": f"EPSG:{epsg}",
            },
        )

    elif mode == 'utm':

        hemisphere = "N" if zone_letter.upper() >= "N" else "S" # N to X -> North ; C to M -> South
        false_northing = 0.0 if hemisphere == "N" else 10_000_000.
        lon0 = (zone_number - 1) * 6 - 180 + 3
        epsg = (32600 if hemisphere.upper() == "N" else 32700) + zone_number

        ds = xr.Dataset(
            {
                "elevation": (("y_utm", "x_utm"), out_image),
                "distance_to_closest": (("y_utm", "x_utm"), out_dist_min_2d),
                "mean_distance": (("y_utm", "x_utm"), out_dist_mean_2d),
                "fpdem_gridded_qual": (("y_utm", "x_utm"), qual_flag),
            },
            coords={
                "x_utm": (("x_utm"), x),
                "y_utm": (("y_utm"), y),
            },
        )

        # Coordinates
        ds = ds.assign_coords(
            x_utm=ds.x_utm.assign_attrs(
                standard_name="x_utm",
                long_name="x coordinate",
                units="m",
            ),
            y_utm=ds.y_utm.assign_attrs(
                standard_name="y_utm",
                long_name="y coordinate",
                units="m",
            ),
        )

        # Define CRS
        ds["crs"] = xr.DataArray(
            0,
            attrs={
                "grid_mapping_name": "transverse_mercator",
                "scale_factor_at_central_meridian": 0.9996,
                "longitude_of_central_meridian": lon0,
                "latitude_of_projection_origin": 0.0,
                "false_easting": 500_000.0,
                "false_northing": false_northing,
                "semi_major_axis": 6378137.0,
                "inverse_flattening": 298.257223563,
                "longitude_of_prime_meridian": 0.0,
                "spatial_ref": f"EPSG:{epsg}",
                "crs_wkt": CRS.from_epsg(epsg).to_wkt(),
                "epsg_code": f"EPSG:{epsg}",
            },
        )

    file = os.path.basename(filename)
    # Add global attributes
    ds.attrs.update({
        "Conventions": "CF-1.7",
        "title": "Level 2 KaRIn High Rate FPDEM Gridded Data Product",
        "institution": "CNES",
        "source": "Large scale Simulator",
        "platform": "SWOT",
        "history": "None",
        "references": "None",
        "reference_document": "None",
        "contact": "damien.desroches@cnes.fr",
        "coordinate_reference_system": f"{epsg}",
        "sampling": resolution,
        "short_name": "L2_HR_FPDEM_Gridded",
        "descriptor_string": file.split('_')[5],
        "crid": "Dx0000",
        "product_version": "1",
        "pge_name": "Pge_v0",
        "pge_version": "1",

        "time_coverage_start": file.split('_')[6],
        "time_coverage_end": file.split('_')[7],

        "geospatial_lat_min": np.min(x),
        "geospatial_lat_max": np.max(x),
        "geospatial_lon_min": np.min(y),
        "geospatial_lon_max": np.max(y),
    })

    # Add necessary attributes
    ds["mean_distance"] = ds["mean_distance"].assign_attrs(
        grid_mapping="crs",
        long_name="mean distance to selected boundaries pixels",
        units="m",
        valid_min=0,
        valid_max=999999,
    )

    ds["distance_to_closest"] = ds["distance_to_closest"].assign_attrs(
        grid_mapping="crs",
        long_name="distance to closest boundary pixel",
        units="m",
        valid_min=0,
        valid_max=999999,
    )

    ds["elevation"] = ds["elevation"].assign_attrs(
        grid_mapping="crs",
        long_name="surface elevation above geoid",
        units="m",
        valid_min=-9999,
        valid_max=9999,
    )

    ds["fpdem_gridded_qual"] = ds["fpdem_gridded_qual"].assign_attrs(
        grid_mapping="crs",
        long_name="DEM quality flag",
        units='None',
        flag_values='0 1 2 4',
        valid_min=0,
        valid_max=4,
    )

    # Write the netcdf file
    ds.to_netcdf(filename)
