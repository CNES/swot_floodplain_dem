# -*- coding: utf8 -*-
'''
Shapefile writer

Copyright (c) 2018 CNES. All rights reserved.
'''

import fiona
import fiona.crs
import shapely
from shapely.geometry import Polygon, mapping
from collections import OrderedDict
from typing import List
import geopandas as gpd

def gdf_to_file(filename: str, gdf: gpd.GeoDataFrame, index: bool = False):
    '''
    Write to shapefile (lat/lon WGS84)

    :param filename: File path
    :param gdf: DataFrame
    ''' 
    # Driver
    driver = "ESRI Shapefile"
    # Projection
    prj = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]'
    # Schema
    if index:
        schema = {'properties': OrderedDict([('longitude', 'float:24.15'),
                                             ('latitude', 'float:24.15'),
                                             ('elevation', 'float:24.15'),
                                             ('range_index', 'int'),
                                             ('azimuth_index', 'int'),
                                             ('time', 'int'),]),
                  'geometry': 'Point'}
        gdf[['geometry', 'longitude', 'latitude', 'elevation',
             'range_index', 'azimuth_index', 'time']].to_file(filename, driver=driver, schema=schema, crs=prj,
                                                              engine='fiona')
    else:
        schema = {'properties': OrderedDict([('longitude', 'float:24.15'),
                                             ('latitude', 'float:24.15'),
                                             ('elevation', 'float:24.15'),]),
                  'geometry': 'Point'}
        gdf[['geometry', 'longitude', 'latitude', 'elevation']].to_file(filename, driver=driver, schema=schema, crs=prj,
                                                                        engine='fiona')


def polygons_to_file(filename: str, polygons: List[Polygon], wse: List[float] = None):
    '''
    Write to shapefile (lat/lon WGS84)

    :param filename: File path
    :param polygon: List of polygons
    ''' 
    # Driver
    driver = "ESRI Shapefile"
    crs = fiona.crs.from_epsg(4326) # WGS84
    # Schema
    if wse != None:
        schema = {'geometry': 'Polygon', 'properties': {'id': 'int', 'elevation': 'float'}}
    else:
        schema = {'geometry': 'Polygon', 'properties': {'id': 'int'}}

    with fiona.open(filename, 'w', driver=driver, crs=crs, schema=schema) as c:
        for i, polygon in enumerate(polygons):
            if wse != None:
                c.write({'geometry': mapping(polygon), 'properties': {'id': i, 'elevation': wse[i]}})
            else:
                c.write({'geometry': mapping(polygon), 'properties': {'id': i}})

            

