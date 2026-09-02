# -*- coding: utf8 -*-
'''
Intersect PIXC with Pekel, PLD or SWORD

Copyright (c) 2018, CNES
'''

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import sys, os
import shapely
from shapely.geometry import Polygon, Point, LineString, shape
import shapely.wkt
import rasterio
from rasterio.features import shapes
from rasterio.mask import mask
import utm
try:
    from osgeo import ogr
except:
    import ogr

########################################################################################################################
# Extract the polygon used for the mask from PLD data
def get_PLD_mask(lake_id, pld_path, pixc):
    """
    :param lake_id:
    :param pld_path:
    :param pixc:
    :return:
    """
    dist_buffer = 0.01 # in degrees because the PLD polygon is in EPSG:4326 (WGS84)

    ind_name = pixc.rfind('/')+1
    pass_id = pixc[ind_name:][20:23]
    logging.info(f'pass_id: {pass_id}')
    pld_filename = [i for i in os.listdir(pld_path) if 'SWOT_LakeDatabase_Nom_%s' %(str(pass_id).zfill(3)) in i]
    if len(pld_filename) >= 1:
        pld_filename = pld_filename[0]
        logging.info(f'pld_filename: {pld_filename}')
    else:
        logging.warning(f'No PLD file found for pass {pass_id}')
        sys.exit()

    pld_file = pld_path + pld_filename
    sqlite_driver = ogr.GetDriverByName(str('SQLITE'))
    pld_ds = sqlite_driver.Open(pld_file, 0) # 0 for read-only mode
    lake_lyr = pld_ds.GetLayer('lake')
    lake_lyr.SetAttributeFilter('"lake_id" = ' + "'" + str(lake_id) + "'")  # string value

    if lake_lyr.GetFeatureCount() > 0:
        pld_feat = lake_lyr.GetNextFeature()
        pld_geom = pld_feat.GetGeometryRef()
        # Convert geom to wkt
        tile_geom_wkt = shapely.wkt.loads(pld_geom.ExportToIsoWkt())
        df_pld_geom = gpd.GeoDataFrame(pd.DataFrame({'geometry':[tile_geom_wkt]}), geometry='geometry', crs="EPSG:4326") # => geometry is a polygon
        poly_mask = df_pld_geom.loc[0, 'geometry'].buffer(dist_buffer)

    return poly_mask

# Extract the polygon used for the mask from Pekel data
def get_Pekel_mask(pekel_path, pixc_tiles_geom):
    """
    :param pekel_path:
    :param pixc_tiles_geom:
    :return:
    """
    # Extract (approximately) boundaries of pixc tile
    tile_geom_wkt = [shapely.wkt.loads(pixc_tiles_geom.wkt)]

    # Extract the polygon of the mask on the pixc tile region
    # Read Pekel
    raster = rasterio.open(pekel_path)
    # Get array of water mask on the area defined by tile_geom_wkt (=pixc area)
    out_image, out_transform = mask(raster, tile_geom_wkt, all_touched=True, nodata=int(101), crop=True)
    out_image = out_image[0]
    # Get crs
    pekel_crs = raster.meta['crs']

    return [pekel_crs, out_image, out_transform]

# Extract the Pekel polygon of the selected occurrences occ_min, occ_max; and add a buffer if wanted
def extract_Pekel_polygon(data_Pekel, occ_min=0, occ_max=100, bufsize=0.01):
    """
    :param data_Pekel:
    :param occ_min:
    :param occ_max:
    :param bufsize:
    :return:
    """
    # Mask Pekel
    shape_gen = ((shape(s), v) for s, v in shapes(data_Pekel[1],
                                                  mask=((data_Pekel[1] > occ_min) & (data_Pekel[1] <= occ_max)),
                                                  transform=data_Pekel[2]))
    datas = dict(zip(["geometry", "class"], zip(*shape_gen)))

    gdf_mask = gpd.GeoDataFrame({'class': list(datas['class'])}, geometry=list(datas['geometry']), crs=data_Pekel[0])

    # Apply buffer on Pekel geometry
    if bufsize > 0.:
        gdf_mask.geometry = gdf_mask.geometry.buffer(bufsize)
    # Union of geometries when possible to get a multipolygon
    poly_pekel = gdf_mask.geometry.unary_union

    return poly_pekel

# From the science tiles file retrieve the geometry of the selected tiles
def get_tiles_geom(gtg_tile_to_open_forMask, gtg_path_fname_to_orbits):
    # Get tiles attributes to be read in a specific CNES orbits file
    gtg_wanted_tiles = [f"PIXC tile {i}" for i in gtg_tile_to_open_forMask]
    gtg_tile_file = gpd.read_file(gtg_path_fname_to_orbits)

    # Keep wanted tiles from orbits file
    if "TILE_NAME" in gtg_tile_file.columns:
        gtg_ind = np.where(gtg_tile_file["TILE_NAME"].isin(gtg_wanted_tiles))[0]

    elif "TILE_REF" in gtg_tile_file.columns:
        gtg_ind = np.where(gtg_tile_file["TILE_REF"].isin(gtg_tile_to_open_forMask))[0]

    else:
        raise KeyError(
            f"Invalid SWOT tiles shapefile. Expected a column named "
            f"'TILE_NAME' or 'TILE_REF', but found: "
            f"{list(gtg_tile_file.columns)}"
        )

    gtg_tile_file2 = gtg_tile_file.loc[list(gtg_ind)].reset_index()

    # Get mask from geometry of tiles
    if len(gtg_tile_file2) > 1:
        gtg_mask = gtg_tile_file2.unary_union
        gtg_mask = gpd.GeoDataFrame(geometry=[gtg_mask], crs=4326)
    else:
        gtg_mask = gtg_tile_file2.geometry
        gtg_mask = gpd.GeoDataFrame(geometry=gtg_mask, crs=4326)

    return gtg_mask


# Select utm coordinates or zone number or zone letter depending on the choice made
def utm_to_latlon(coords, cflag):
    """

    :param coords:
    :param cflag:
    :return:
    """
    if cflag == 1:
        utm_x = utm.from_latlon(coords[1], coords[0])[0] # lon
        utm_y = utm.from_latlon(coords[1], coords[0])[1] # lat
        return (utm_x, utm_y)
    elif cflag == 2:
        return utm.from_latlon(coords[1], coords[0])[2]
    else:
        return utm.from_latlon(coords[1], coords[0])[3]

#  Retrieve the information about the reaches and nodes present on the studied region provided as tiles' info
def get_reach_node_info(pixc_tiles_geom, sword_path, hydrobasin_lv1_file, hydrobasin_lv2_file):
    """

    :param pixc_tiles_geom:
    :param sword_path:
    :param hydrobasin_lv1_file:
    :param hydrobasin_lv2_file:
    :return:
    """
    # Get hydrobasin number corresponding to PIXC tile
    gdf_hydrobasin1 = gpd.read_file(hydrobasin_lv1_file)
    gdf_hydrobasin2 = gpd.read_file(hydrobasin_lv2_file)

    continent_code = {1: 'AF', 2: 'EU', 3: 'SI', 4: 'AS', 5: 'AU', 6: 'SA', 7: 'NA', 8: 'AR', 9: 'GR'}
    ct_number = gdf_hydrobasin1.clip(pixc_tiles_geom).PFAF_ID.values
    ct_id = [continent_code[i] for i in ct_number]
    hb_number = gdf_hydrobasin2.clip(pixc_tiles_geom).PFAF_ID.values
    logging.info(f'    Continent number: {ct_number}, {ct_id} -- Hydrobasin number: {hb_number}')

    sword_nodes_path = [os.path.join(sword_path, ct_id[i], f'{ct_id[i].lower()}_sword_nodes_hb{str(j)}_v17.shp')
                      for i in range(len(ct_id)) for j in hb_number if continent_code[int(str(j)[0])] == ct_id[i]]
    sword_reaches_path = [os.path.join(sword_path, ct_id[i], f'{ct_id[i].lower()}_sword_reaches_hb{str(j)}_v17.shp')
                      for i in range(len(ct_id)) for j in hb_number if continent_code[int(str(j)[0])] == ct_id[i]]

    gdf_sword_nodes_clipped = gpd.GeoDataFrame()
    gdf_sword_reaches_clipped = gpd.GeoDataFrame()
    for i in range(len(sword_nodes_path)):
        gdf_sword_nodes = gpd.read_file(sword_nodes_path[i])
        gdf_sword_reaches = gpd.read_file(sword_reaches_path[i])
        gdf_sword_nodes_clipped = pd.concat([gdf_sword_nodes_clipped, gdf_sword_nodes.clip(pixc_tiles_geom)])
        gdf_sword_reaches_clipped = pd.concat([gdf_sword_reaches_clipped, gdf_sword_reaches.clip(pixc_tiles_geom)])

    return gdf_sword_reaches_clipped, gdf_sword_nodes_clipped


# Retrieve the SWORD polygon using reaches info and apply a buffer if wanted
def get_poly_sword_from_reach(gdf_sword_reaches, buffer_size=0):
    """

    :param gdf_sword_reaches:
    :param buffer_size:
    :return:
    """
    gdf_sword_reaches = gdf_sword_reaches.explode(index_parts=True)

    gdf_reach_poly = gpd.GeoDataFrame()
    for item in gdf_sword_reaches.itertuples():
        reach_width = item.width
        # Get reach vertices coordinates in utm
        linestr_utm = LineString([utm_to_latlon(xy, 1) for xy in tuple(item.geometry.coords)])
        zone_number = utm_to_latlon(item.geometry.coords[0], 2)
        zone_letter = utm_to_latlon(item.geometry.coords[0], 3)

        # Apply buffer equal to reach width
        linestr_buf = linestr_utm.buffer(reach_width)
        # Get polygon coordinates back to latlon
        linestr_buf_latlon = Polygon([tuple(reversed(utm.to_latlon(linestr_buf.exterior.coords.xy[0][k],
                                                                   linestr_buf.exterior.coords.xy[1][k],
                                                                   zone_number, zone_letter)))
                                      for k in range(len(linestr_buf.exterior.coords.xy[0]))])
        if buffer_size > 0.:
            linestr_buf_latlon = linestr_buf_latlon.buffer(buffer_size)

        gdf_reach = gpd.GeoDataFrame(geometry=[linestr_buf_latlon], crs=gdf_sword_reaches.crs)
        gdf_reach_poly = pd.concat([gdf_reach_poly, gdf_reach])

    reach_poly = gdf_reach_poly.geometry.unary_union

    return reach_poly

# Retrieve all the points of a geodataframe present in the selected mask
def intersect_pixc_pixcvec_mask(df, poly_mask):

    data_pixc = gpd.GeoDataFrame(df, crs='EPSG:4326',
                                 geometry=[Point(x, y) for x, y in zip(df['longitude'], df['latitude'])])

    # Intersect
    try:
        new_pixc = data_pixc.clip(poly_mask)
    except:
        new_pixc = data_pixc.loc[data_pixc.intersects(poly_mask)]

    new_pixc = pd.DataFrame(new_pixc).drop(['geometry'], axis=1)

    return new_pixc
