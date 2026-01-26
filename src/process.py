# -*- coding: utf8 -*-
"""
Create ply file from a list of pixel cloud files

Copyright (c) 2018, CNES
"""

import os
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import mahotas as mh
from sklearn.cluster import DBSCAN

from constants import (WATER_LABEL, WATER_NEAR_LAND_LABEL, DARK_WATER_LABEL,
                       WATER_NEAR_LAND_LOW_COH_LABEL, WATER_LOW_COH_LABEL)

from spatial import compute_binary_mask

def extract_water_points(pixelcloud: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Extract pixel cloud with water labels

    :param pixelcloud: Pixel cloud Dataframe

    :return points: Extracted water points
    """

    # Extract pixel cloud with water labels
    return pixelcloud.loc[(pixelcloud.classification == WATER_LABEL) |
                          (pixelcloud.classification == WATER_NEAR_LAND_LABEL) |
                          (pixelcloud.classification == DARK_WATER_LABEL) |
                          (pixelcloud.classification == WATER_NEAR_LAND_LOW_COH_LABEL) |
                          (pixelcloud.classification == WATER_LOW_COH_LABEL)].copy()


def remove_near_range_pixels(water, azimuth_max, cross_track_min=5000):
    """
    Filter points whose cross-track distance in below a threshold

    :param water: GeoPandas Dataframe
    :param azimuth_max: Maximum azimuth value for a tile
    :param cross_track_min: minimum cross-track value to keep in data

    :return: GeoPandas Dataframe filtered
    :rtype: GeoPandas Dataframe
    :return: Numpy 1D Array with range indices corresponding to the border of the tile, for each azimuth position
    :rtype: numpy array
    """

    water_fil = water.loc[(np.abs(water['cross_track']) > cross_track_min)]
    water_removed = water.loc[(np.abs(water['cross_track']) <= cross_track_min)]
    min_range_indices_to_remove = np.zeros([azimuth_max])

    for i in range(azimuth_max):
        try:
            min_after_filtering = np.nanmin(water_fil.loc[water_fil['azimuth_index'] == i]['range_index'])
            max_filtered_area = np.nanmax(water_removed.loc[water_removed['azimuth_index'] == i]['range_index'])
            if max_filtered_area == min_after_filtering - 1:
                min_range_indices_to_remove[i] = min_after_filtering
        except:
            pass
    return water_fil, min_range_indices_to_remove

def extract_contiguous_water_points(water: gpd.GeoDataFrame,
                                    range_max: int,
                                    azimuth_max: int,
                                    threshold: float = 100000.0) -> gpd.GeoDataFrame:
    """
    Extract water points
    Steps :
      1) Create water mask
      2) Labelize regions
      3) Compute regions area and remove small regions

    :param water: Water Dataframe
    :param range_max: Range size
    :param azimuth_max: Azimuth size
    :param threshold: Threshold for the keeping region area

    :return points: Extracted water points
    """

    # Create water mask
    water_mask = compute_binary_mask(azimuth_max,
                                     range_max,
                                     water['azimuth_index'].values,
                                     water['range_index'].values)

    # Boundary conditions
    bc = np.ones((3, 3))
    # Labelize regions
    labeled, nr_objects = mh.label(water_mask, Bc=bc)

    # Region size
    sizes = mh.labeled.labeled_size(labeled)
    df_sizes = pd.DataFrame(data=sizes[1:], columns=["size"], index=list(range(1, len(sizes))))
    # Compute region area
    water['region'] = water.apply(lambda row: labeled[row['azimuth_index'], row['range_index']], axis=1)
    df_sizes['area'] = water.groupby(['region'])['pixel_area'].sum()
    df_sizes.sort_values(['size'], ascending=False)
    # Keep regions with area superior to threshold
    keep_regions = df_sizes.loc[df_sizes.area > threshold].index.tolist()

    # Extract points corresponding to remaining regions
    return water.loc[water['region'].isin(keep_regions)].copy()

#
def pre_processing_pixc(pixc_reader, cross_track_min, threshold, sig0_qual):
    """
    Pre-processing/ filtering of the entire PIXC

    :param pixc_reader: DataFrame with PIXC info
    :param cross_track_min: minimum crosstrack value to remove
    :param threshold: minimum surface of points clusters to be kept
    :param sig0_qual: value of sig0_qual (PIXC attribute) : 0 good, 1: bad
    :param filtering_pekel_start: yes or no
    :param pekel_0_100_poly: Pekel polygon of occurrences >0%
    :return: water, min_range_indices_to_remove
    """
    # Extract points
    logging.info("Extract water points")
    water = extract_water_points(pixc_reader.get_data())

    # Remove pixels too close to near_range
    logging.info("Remove pixels too close to near_range")
    water, min_range_indices_to_remove = remove_near_range_pixels(water, pixc_reader.azimuth_max,
                                                                  cross_track_min=cross_track_min)

    # Extract contiguous water points by keeping only water bodies whose area is greater than threshold
    logging.info("Extract contiguous water points by keeping only water bodies whose area is greater than threshold")
    water = extract_contiguous_water_points(water, pixc_reader.range_size, pixc_reader.azimuth_size,
                                            threshold=threshold)

    # Select only points with a good sig0_qual
    if sig0_qual == 'yes':
        water = water.loc[(water.sig0_qual == 0)]

    return water, min_range_indices_to_remove

#
def get_range_azimuth_extrema(grae_variable, grae_range_size, grae_azimuth_size):
    """
    Obtain the maximum and minimum of range and azimuth around a selected water body/label

    :param grae_variable: DataFrame of the PIXC points info for the selected water body/label
    :param grae_range_size: Size of range of the entire PIXC
    :param grae_azimuth_size: Size of azimuth of the entire PIXC
    :return:
    """
    azimuth_index = grae_variable["azimuth_index"].values.astype('int')
    range_index = grae_variable["range_index"].values.astype('int')
    height = grae_variable["height"].values
    #
    height_tab = np.zeros([grae_azimuth_size, grae_range_size], dtype=float)
    for i in range(len(height)):
        height_tab[azimuth_index[i], range_index[i]] = height[i]

    # Get the azimuth and range extrema where data are available
    a = list(np.all(height_tab == 0, axis=1))
    b = list(np.all(np.transpose(height_tab) == 0, axis=1))
    if False in a:
        grae_l0 = max(0, a.index(False) - 10)
        grae_l1 = min(len(a) - a[::-1].index(False) + 10, len(a))
    else:
        grae_l0 = 0
        grae_l1 = len(a) - 1
    if False in b:
        grae_c0 = max(0, b.index(False) - 10)
        grae_c1 = min(len(b) - b[::-1].index(False) + 10, len(b))
    else:
        grae_c0 = 0
        grae_c1 = len(b) - 1
    print('    Azimuth/Range extrema:  lat ', grae_l0, grae_l1, '-- lon ', grae_c0, grae_c1)

    return grae_l0, grae_l1, grae_c0, grae_c1

def prepare_clustering_image(pci_height_tab, pci_sig0_tab, pci_latitude_tab, pci_longitude_tab,
                             pci_label_tab, pci_label, pci_l0, pci_l1, pci_c0, pci_c1):
    """
    Preparation of the image/array needed for the clustering step (lat, lon, h, sig0 are required in this order)

    :param pci_height_tab: Array of height
    :param pci_sig0_tab: Array of sig0
    :param pci_latitude_tab: Array of latitude
    :param pci_longitude_tab: Array of longitude
    :param pci_label_tab: Array of water bodies labels
    :param pci_label: Integer of the selected label
    :param pci_l0: Minimum azimuth
    :param pci_l1: Maximum azimuth
    :param pci_c0: Minimum range
    :param pci_c1: Maximum range
    :return: Array with all the information needed for the clustering step
    """
    height = np.where(pci_label_tab == pci_label, pci_height_tab, -1e6)
    sig0 = np.where(pci_label_tab == pci_label, pci_sig0_tab, -1e6)
    lat = np.where(pci_label_tab == pci_label, pci_latitude_tab, -1e6)
    lon = np.where(pci_label_tab == pci_label, pci_longitude_tab, -1e6)

    image = np.zeros([sig0.shape[0], height.shape[1], 4])

    image[:, :, 0] = height
    image[:, :, 1] = sig0
    image[:, :, 2] = lon
    image[:, :, 3] = lat
    pci_sub_image = image[pci_l0:pci_l1, pci_c0:pci_c1, :]

    return pci_sub_image

def find_attributes(row, water, flag):
    """
    Retrieve from the entire PIXC DataFrame the columns of selected attributes

    :param row: row of the PIXC DataFrame
    :param water: DataFrame of the restricted DataFrame lacking attributes columns
    :param flag: integer to select the list of columns wanted in the new DataFrame
    :return: Series with the wanted attributes
    """
    row_water = water[(water.latitude == row.latitude) & (water.longitude == row.longitude)]
    if flag == 1:
        return pd.Series([row_water['classification_qual'].values[0],
                          row_water['geolocation_qual'].values[0],
                          row_water['time'].values[0],
                          row_water['elevation'].values[0]])
    elif flag == 2:
        return pd.Series([row_water['height'].values[0],
                          row_water['sig0'].values[0],
                          row_water['azimuth_index'].values[0],
                          row_water['range_index'].values[0],
                          row_water['classification'].values[0]])

def remove_borders(data: gpd.GeoDataFrame, range_min: int, range_max: int,
                        azimuth_min: int, azimuth_max: int, min_range_indices_to_remove: np.array):
    """
    Remove borders points in range and azimuth

    :param data: Boundary points list
    :param range_max: Maximum range value for a tile
    :param azimuth_max: Maximum azimuth value for a tile
    :param min_range_indices_to_remove: minimum range_indice to remove for each azimuth line

    :return: Boundary points list filtered
    :rtype: GeoPandas Dataframe
    """
    data = data.loc[data.range_index > range_min]
    data = data.loc[data.azimuth_index > azimuth_min]
    data = data.loc[data.range_index < range_max]
    data = data.loc[data.azimuth_index < azimuth_max]

    for i in range(azimuth_max):
        data = data.loc[np.logical_or((data['azimuth_index'] != i),
                                      (data['range_index'] != min_range_indices_to_remove[i]))]

    return data

def filter_data_based_on_quality_flag(water, classif_qual, geoloc_qual):
    """
    :param water: DataFrame with PIXC points info
    :param classif_qual: integer value (bitwise) of the maximum classification quality wanted for data
    :param geoloc_qual: integer value (bitwise) of the maximum geolocation quality wanted for data
    :return: water filtered with the quality flags
    """
    logging.info("Filtering using flags")

    water = water[water.classification_qual < classif_qual]
    water = water[water.geolocation_qual < geoloc_qual]

    return water

def remove_isolated_points_dbscan(rip_gdf: gpd.GeoDataFrame, dist_neighbors: float = 0.001, nb_neighbors: int = 10):
    """
    Using DBSCAN remove isolated points with fewer neighbors than a defined parameter
    and when distance is longer than defined distance

    :param rip_gdf: Dataframe of points with lat/lon information
    :param dist_neighbors: Maximum distance for two points to be considered neighbors
    :param nb_neighbors: Number of required neighbors not to be considered an isolated point
    :return: dataframe filtered out with a labeling column added
            (needed for the polygon extraction necessary for the rasterization step)
    """
    rip_gdf = rip_gdf.reset_index()

    x = np.stack((rip_gdf.longitude, rip_gdf.latitude), axis=-1)

    db = DBSCAN(eps=dist_neighbors, min_samples=nb_neighbors).fit(x)
    db_labels = db.labels_

    try:
        rip_gdf.loc[:, 'lab_dbscan'] = db_labels
        rip_gdf2 = rip_gdf.loc[np.where(db_labels != -1)]
    except:
        rip_gdf2 = rip_gdf
    logging.info(f'        {len(rip_gdf) - len(rip_gdf2)} points were removed')

    return rip_gdf2

def final_filtering_FPDEM_results(data, filtering_pekel_end, pekel_X2_100_poly, d_ngbr, n_ngbr, output_path):
    """
    Filtering with Pekel if wanted, where all points within Pekel polygon of high occurrences are removed
    and filtering of isolated points

    :param data: All FPDEM results dataframe
    :param filtering_pekel_end: yes or no
    :param pekel_X2_100_poly: Pekel polygon of high water occurrences
    :param d_ngbr: distance between neighbors (for DBSCAN)
    :param n_ngbr: number of neighbors (for DBSCAN)
    :param output_path: path to the output directory to write file containing the labels
                        of the différent clusters within FPDEM results
    :return: filtered dataframe
    """

    # Filter using Pekel occurrences
    if filtering_pekel_end == 'yes' and pekel_X2_100_poly is not None:
        logging.info('Filtering FPDEM points within Pekel high occurrences')
        pts_out = data.copy().clip(pekel_X2_100_poly)
        data = data[~data.geometry.isin(pts_out.geometry)]

    # Filtering isolated points once all water bodies, cycles and tiles concatenated
    logging.info("Remove isolated points")
    # TODO : should we change the distance to neighbors in meters instead of degrees?
    #  This would imply to convert latlon in utm.
    data = remove_isolated_points_dbscan(data, dist_neighbors=d_ngbr, nb_neighbors=n_ngbr)
    data[['geometry', 'lab_dbscan']].to_file(os.path.join(output_path, 'results_fpdem_labels_dbscan.shp'))
    logging.info('The results of DBSCAN labeling were written in file "results_fpdem_labels_dbscan.shp" ')

    return data

def valid_date(dataFrame):
    """
    Create valid_date attribute to create raster file
    """
    dataFrame['fpdem_ungridded_qual'] = pd.Series([1 for i in range(dataFrame.size)])
    return dataFrame

def compute_mean_3sigma(cm3s_gdf):
    """
    Filter out points whose elevation is outside the 3 sigma limit

    :param cm3s_gdf: dataframe of FPDEM ungridded points
    :return: filtered out dataframe
    """
    cm3s_gdf = cm3s_gdf.reset_index()
    in_v_val = cm3s_gdf['elevation'].values

    # Retrieve all values != numpy.nan
    not_nan_idx = np.where(np.isfinite(in_v_val))[0]

    if len(not_nan_idx) != 0:
        # Compute statistical values over the input vector
        med = np.nanmedian(in_v_val)
        std = np.nanstd(in_v_val)
        # Remove indices with value out of 3 sigma
        idx_lower = np.where(in_v_val[not_nan_idx.copy()] < med - 3 * std)[0]
        v_indices = np.delete(not_nan_idx.copy(), idx_lower)
        idx_higher = np.where(in_v_val[v_indices] > med + 3 * std)[0]
        v_indices = np.delete(v_indices, idx_higher)

    cm3s_gdf = cm3s_gdf[cm3s_gdf.index.isin(list(v_indices))]
    return cm3s_gdf

def filter_elevation(gf_gdf, method):
    """
    Filter out the entire dataframe depending on elevation
    Three different methods are available (because some are working better for smaller samples)

    :param gf_gdf: dataframe
    :param method: method dependent on the sampling size
    :return: filtered dataframe
    """
    gf_gdf = gf_gdf.reset_index()
    in_v_val = gf_gdf['elevation'].values

    # Retrieve all values != numpy.nan
    not_nan_idx = np.where(np.isfinite(in_v_val))[0]

    if len(not_nan_idx) != 0:
        if method == '3sigma':
            # Compute statistical values over the input vector
            med = np.nanmedian(in_v_val)
            std = np.nanstd(in_v_val)
            # Remove indices with value out of 3 sigma
            idx_lower = np.where(in_v_val[not_nan_idx] < med - 3 * std)[0]
            idx_higher = np.where(in_v_val[not_nan_idx] > med + 3 * std)[0]
            inds = np.append(idx_lower, idx_higher)

        elif method == 'mad':
            median = np.nanmedian(in_v_val)
            mad = np.median(np.abs(in_v_val[not_nan_idx] - median))
            scores = np.abs(in_v_val[not_nan_idx] - median) / mad
            # Depending on the sample size the score threshold is different
            if len(gf_gdf) < 10:
                inds = np.where(scores > 5)[0]
            else:
                inds = np.where(scores > 3)[0]

        elif method == 'iqr':
            q1 = np.quantile(in_v_val, 0.25)
            q3 = np.quantile(in_v_val, 0.75)
            iqr = q3 - q1
            inds = np.where((in_v_val[not_nan_idx] >= q1 - 1.5 * iqr) & (in_v_val[not_nan_idx] <= q3 + 1.5 * iqr))

    else:
        inds = not_nan_idx

    v_indices = np.delete(not_nan_idx, inds)

    return gf_gdf[gf_gdf.index.isin(list(v_indices))]
