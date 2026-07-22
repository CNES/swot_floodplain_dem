# -*- coding: utf8 -*-
"""
Categorize the clusters into WATER, WATER_LAND or LAND depending on normality parameters

Copyright (c) 2018, CNES
"""

import os
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import scipy
from skimage import morphology
import random
from scipy.stats import skew, probplot, kurtosis, zscore

def compute_label_and_remove_small_object(water, range_size, azimuth_size, pekel_occ_X_100,
                                          min_size=1000, connectivity=1, plot='no', outpath='', cycle='007'):
    """
    Filter PIXC DataFrame depending on classification
    and extract arrays for lat, lon, range, azimuth, h, sig0 and classification
    which are then filtered to remove isolated points and clustered nto different labels
    (one for each water body present in the PIXC)

    :param water: DataFrame with all PIXC points info
    :param range_size: integer for the size of range of the PIXC
    :param azimuth_size: integer for the size of azimuth of the PIXC
    :param pekel_occ_X_100: Pekel polygon for occurrences > X% to filter Dark Water points
    :param min_size: integer of the Minimum size of a cluster/water body
    :param connectivity: integer of Morphology connectivity parameter value
    :param plot: yes or no to plot figures
    :param outpath: path of output directory to plot figures
    :param cycle: cycle number of the studied PIXC
    :return:
    """

    # Select all points with classification 3 and 4
    subset_all = water.loc[(water["classification"] == 3) | (water["classification"] == 4)]
    # Select all points in Pekel water occurrences from 95% to 100%
    if pekel_occ_X_100 is not None:
        subset_pekel = water.clip(pekel_occ_X_100)
        subset_pekel = subset_pekel.loc[subset_pekel["classification"] == 5]
    else:
        subset_pekel = gpd.GeoDataFrame()
    # Create new water dataframe with only the selected points
    new_water = pd.concat([subset_pekel, subset_all])
    # All negative values of sig0 are set to -1e6
    new_water.sig0 = new_water.sig0.mask(new_water.sig0.le(0), -1e6)

    # Read wanted columns and set them up into arrays
    classification = new_water["classification"].values
    height = new_water["height"].values
    sig0 = new_water["sig0"].values
    azimuth_index = new_water["azimuth_index"].values.astype('int')
    range_index = new_water["range_index"].values.astype('int')
    latitude = new_water["latitude"].values
    longitude = new_water["longitude"].values

    # Redefine those arrays using range/azimuth as coordinates
    sig0_tab = np.zeros([azimuth_size, range_size], dtype=float)
    height_tab = np.zeros([azimuth_size, range_size], dtype=float)
    classification_tab = np.zeros([azimuth_size, range_size], dtype=float)
    azimuth_index_tab = np.zeros([azimuth_size, range_size], dtype=int)
    range_index_tab = np.zeros([azimuth_size, range_size], dtype=int)
    latitude_tab = np.zeros([azimuth_size, range_size], dtype=float)
    longitude_tab = np.zeros([azimuth_size, range_size], dtype=float)
    for i in range(len(sig0)):
        sig0_tab[azimuth_index[i], range_index[i]] = sig0[i]
        height_tab[azimuth_index[i], range_index[i]] = height[i]
        classification_tab[azimuth_index[i], range_index[i]] = classification[i]
        azimuth_index_tab[azimuth_index[i], range_index[i]] = azimuth_index[i]
        range_index_tab[azimuth_index[i], range_index[i]] = range_index[i]
        latitude_tab[azimuth_index[i], range_index[i]] = latitude[i]
        longitude_tab[azimuth_index[i], range_index[i]] = longitude[i]

    # Filter out all points classified as 1 and 2
    water_tab = np.where(classification_tab > 2, 1, 0).astype(bool)

    # Plot all water bodies
    if plot[0:3] == 'yes':
        extent = [np.min(longitude_tab[np.nonzero(longitude_tab)]), np.max(longitude_tab[np.nonzero(longitude_tab)]),
                  np.min(latitude_tab[np.nonzero(latitude_tab)]), np.max(latitude_tab[np.nonzero(latitude_tab)])]
        lon_tab = np.where(water_tab == 1, longitude_tab, 0)
        lat_tab = np.where(water_tab == 1, latitude_tab, 0)
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot()
        ax.scatter(lon_tab, lat_tab, s=1)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(f"All PIXC points after pre-processing")
        if plot == 'yes2':
            plt.show()
        else:
            fig.savefig(f'{outpath}/cycle{cycle}/plot_all_pts_after_classif_filtering.png')
            plt.close(fig)

    # Remove all points which are isolated
    water_tab = morphology.remove_small_objects(water_tab, min_size=min_size, connectivity=connectivity)

    # Plot all water bodies
    if plot[0:3] == 'yes':
        lon_tab = np.where(water_tab == 1, longitude_tab, 0)
        lat_tab = np.where(water_tab == 1, latitude_tab, 0)
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot()
        ax.scatter(lon_tab, lat_tab, s=1)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(f"All points after filtering of isolated points")
        if plot == 'yes2':
            plt.show()
        else:
            fig.savefig(f'{outpath}/cycle{cycle}/plot_all_pts_after_classif_filtering_and_morphology.png')
            plt.close(fig)

    # Label all clusters of points to obtain the water bodies
    label_tab, count = scipy.ndimage.label(water_tab)

    # Plot all water bodies
    if plot[0:3] == 'yes':
        lon_tab = np.where(water_tab == 1, longitude_tab, 0)
        lat_tab = np.where(water_tab == 1, latitude_tab, 0)
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, count)]
        random.shuffle(colors)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot()
        for k, color in enumerate(colors, start=1):
            # Get indices of the label points
            ind = np.where(label_tab == k)
            # Figure/plot of the different bodies
            ax.scatter(lon_tab[ind], lat_tab[ind], c=np.array([color]), s=1)
            lon_avg = (min(lon_tab[ind]) + max(lon_tab[ind])) / 2
            lat_avg = (min(lat_tab[ind]) + max(lat_tab[ind])) / 2
            ax.annotate(k, xy=(lon_avg, lat_avg), xycoords='data', color='black', fontsize=12)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(f"All labels after filtering of isolated points")
        if plot == 'yes2':
            plt.show()
        else:
            fig.savefig(f'{outpath}/cycle{cycle}/plot_all_labels_position.png')
            plt.close(fig)

    return (label_tab, count, classification_tab, height_tab, sig0_tab,
            azimuth_index_tab, range_index_tab, latitude_tab, longitude_tab)

def compute_subwater_extract_from_label(water, label_tab, azimuth_index_tab, range_index_tab, label):
    """
    Filter the PIXC DataFrame to get only the points for the selected water body/label

    :param water: DataFrame of PIXC points info
    :param label_tab: array of labels
    :param azimuth_index_tab: array of azimuth
    :param range_index_tab: array of range
    :param label: integer with label value
    :return: water_filtered -> DataFrame
    """
    ind = np.where(label_tab == label)
    azimuth_index_filt = azimuth_index_tab[ind]
    range_index_filt = range_index_tab[ind]

    # Produce Geopanda dataframe by filtering initial gdf. Probably more elegant way to do it...
    water_filtered = gpd.GeoDataFrame()
    water_filtered["azimuth_index"] = azimuth_index_filt
    water_filtered["range_index"] = range_index_filt
    water_filtered["azimuth_range_index"] = water_filtered["azimuth_index"]*100000 + water_filtered["range_index"]

    water["azimuth_range_index"] = water["azimuth_index"]*100000 + water["range_index"]
    water_filtered = water.loc[(water['azimuth_range_index'].isin(water_filtered["azimuth_range_index"]))]

    return water_filtered

def remove_outliers_in_height(h, sig0):
    """
    Filter outliers from h and sig0 arrays using zscore on h

    :param h: list of height
    :param sig0: list of sig0
    :return: h, sig0
    """
    zvalue = 2.7
    zh = zscore(h)
    h = h[np.where(np.abs(zh) < zvalue)]
    sig0 = sig0[np.where(np.abs(zh) < zvalue)]
    return h, sig0

def get_h_sig0_and_filter(water):
    """
    Extract h and sig0 from DataFrame and filter outliers

    :param water: DataFrame of PIXC points info
    :return: h, sig0
    """
    h = water['height'].values
    sig0 = water['sig0'].values

    sig0 = np.where(sig0 > 0, 10 * np.log10(sig0), -1e6)

    # Remove height outliers using zscore (threshold zvalue)
    h, sig0 = remove_outliers_in_height(h, sig0)

    sig0 = np.array([x for x in sig0 if x >= 0])
    return h, sig0

def compute_histogram(water):
    """
    Obtain the density histograms for h and sig0

    :param water: DataFrame of PIXC points info
    :param plot: yes or no
    :return: h histo info, sig0 histo info
    """
    h, sig0 = get_h_sig0_and_filter(water)

    # Get histograms for height and sig0
    counts_h, bins_h = np.histogram(h, bins=200)
    counts_sig, bins_sig = np.histogram(sig0, bins=200)

    return (counts_h, bins_h), (counts_sig, bins_sig)

def determine_normality_parameters(water):
    """
    Obtain the normality parameters for the h and sig0 array of a particular water body/label
    :param water: DataFrame of PIXc points info
    :return: all normality parameters
    """
    h, sig0 = get_h_sig0_and_filter(water)

    meds = np.nanmedian(sig0)

    # Determine skewness
    sh = skew(h, bias=True, nan_policy='omit')
    ss = skew(sig0, bias=True, nan_policy='omit')

    # Determine kurtosis
    kh = kurtosis(h)
    ks = kurtosis(sig0)

    # Determine R**2 of Q-Q plots
    resh = probplot(h, rvalue=True, plot=None)
    r2_h = resh[1][2]**2
    ress = probplot(sig0, rvalue=True, plot=None)
    r2_s = ress[1][2]**2

    return meds, sh, ss, kh, ks, r2_h, r2_s

def determine_water_body_type(choice, meds, sk_h, sk_s, ku_h, ku_s, ratio_pekel, ratio_sword):
    """
    Determine the category (land, water or water_land) of a particular water body/label depending on normality parameters

    :param choice: list of possible choices ['WATER', 'WATER_LAND', 'LAND']
    :param meds: median of sig0
    :param sk_h: skew of h
    :param sk_s: skew of sig0
    :param ku_h: kurtosis of h
    :param ku_s: kurtosis of sig0
    :param ratio_pekel: percentage of points in label present in Pekel polygon
    :param ratio_sword: percentage of points in label present in SWORD polygon
    :return: water_body_type
    """
    if 'WATER_LAND' in choice:
        # TODO: tests are currently performed to evaluate the changes in the categorization
        if ratio_sword > 0.3 or meds > 17.:
            if 'LAND' in choice:
                choice.remove('LAND')
            if 'WATER_LAND' in choice:
                choice.remove('WATER_LAND')
        else:
            if ratio_pekel > 0.98:
                if 'LAND' in choice:
                    choice.remove('LAND')
                if 'WATER_LAND' in choice:
                    choice.remove('WATER_LAND')
            elif 0.2 <= ratio_pekel <= 0.98:
                if sk_s > 0.6:
                    if 'LAND' in choice:
                        choice.remove('LAND')
                    if 'WATER' in choice:
                        choice.remove('WATER')
            elif ratio_pekel < 0.2:
                if sk_s > 0.8:
                    if 'WATER' in choice:
                        choice.remove('WATER')
                    if 'LAND' in choice:
                        choice.remove('LAND')
                else:
                    if 'WATER' in choice:
                        choice.remove('WATER')
                    if 'WATER_LAND' in choice:
                        choice.remove('WATER_LAND')

    else:
        #
        if ratio_sword > 0.3 or meds > 17.:
            choice.remove('LAND')
        else:
            if ratio_pekel < 0.1:
                if 'WATER' in choice:
                    choice.remove('WATER')
            else:
                if meds < 12. and sk_s < 0.5:
                    if 'WATER' in choice:
                        choice.remove('WATER')
                if meds < 10. and sk_h < 0.2:
                    if 'WATER' in choice:
                        choice.remove('WATER')
                if meds < 12. and ku_h < -0.5:
                    if 'WATER' in choice:
                        choice.remove('WATER')
                if meds < 10. and ku_s < 0.:
                    if 'WATER' in choice:
                        choice.remove('WATER')

    if len(choice) == 0:
        choice = ['WATER']
    water_body_type = choice[0]

    return water_body_type

# Find category of the water body : WATER, WATER_LAND, LAND
def find_body_category(water_l, pekel_0_100_poly, poly_sword,
                       plot='no', choices=['WATER', 'WATER_LAND', 'LAND'],
                       cycle='007', label=1, output_path=''):
    """
    Get the density histograms for h and sig0
    and determine if the water body/label is water, land or a mix of both

    :param water_l: DataFrame of PIXC points info for the studied label
    :param pekel_0_100_poly: pekel polygon of occurrences >0%
    :param poly_sword: SWORD polygon
    :param plot: yes or no
    :param choices: list of choices for categorization ['WATER', 'WATER_LAND', 'LAND']
    :param cycle: cycle of the studied PIXC
    :param label: number of the studied label
    :param output_path: directory of the output for saving plots
    :return: water_type, categ_params
    """
    # Categorizing the detected water body into land, water or water_land
    h_hist, sig_hist = compute_histogram(water_l)

    med_s, skh, sks, kh, ks, r2h, r2s = determine_normality_parameters(water_l)

    ratio_pekel = len(water_l.clip(pekel_0_100_poly)) / len(water_l)
    if poly_sword is not None:
        ratio_sword = len(water_l.clip(poly_sword)) / len(water_l)
    else:
        ratio_sword = 0.
    logging.info(f'Ratio pekel, ratio sword: {ratio_pekel}, {ratio_sword}')
    logging.info(f'Normality parameters (median_sig0, skew_h, skew_sig0, kurtosis_h, kurtosis_sig0, R²_h, R²_sig0):'
                 f'{med_s}, {skh}, {sks}, {kh}, {ks}, {r2h}, {r2s}')
    categ_params = [ratio_pekel, ratio_sword, med_s, skh, sks, kh, ks, r2h, r2s]

    if plot[0:3] == "yes":
        fig1, (ax11, ax12) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        fig1.suptitle(f'ratio_Pekel=%.2f; ratio_sword=%.2f' %(ratio_pekel, ratio_sword))
        ax11.set_title(f"Height (m) hist", fontsize=10)
        ax11.set_ylabel('Count')
        ax11.set_xlabel('Height')
        ax11.stairs(h_hist[0], h_hist[1],
                    label=f'skew=%.2f\nkurtosis=%.2f\nR^2=%.3f' %(skh, kh, r2h))
        ax11.legend(handlelength=0., handleheight=0, handletextpad=0)
        ax12.set_title(f"Sig0 (dB) hist", fontsize=10)
        ax12.stairs(sig_hist[0], sig_hist[1],
                    label=f'median=%.2f\nskew=%.2f\nkurtosis=%.2f\nR^2=%.3f' %(med_s, sks, ks, r2s))
        ax12.legend(handlelength=0., handleheight=0, handletextpad=0)
        ax12.set_xlabel('Sig0')
        ax12.set_ylabel('Count')
        if plot == 'yes2':
            plt.show()
        else:
            fig1.savefig(os.path.join(output_path,
                                      f'cycle{cycle}/plot_histograms_of_h_sig0_cycle{cycle}_label{label}.png'))
            plt.close(fig1)

    water_type = determine_water_body_type(choices.copy(), med_s, skh, sks, kh, ks, ratio_pekel, ratio_sword)
    return water_type, categ_params
