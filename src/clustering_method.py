# -*- coding: utf8 -*-
"""
Calculate direct method using clustering algorithm (KMeans)
When the identified water body is classified as 'WATER_LAND'

Copyright (c) 2018, CNES
"""

import os
import logging
import random
import pandas as pd
import geopandas as gpd
import numpy as np
from skimage import morphology
import scipy
from scipy import stats
from scipy.spatial import distance, Delaunay

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, Birch, HDBSCAN
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.ensemble import IsolationForest

import umap
from umap import UMAP

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.io.img_tiles import GoogleTiles

    # Plot basemap params
    wgs84 = ccrs.PlateCarree(globe=ccrs.Globe(ellipse='WGS84'))
    geodetic = ccrs.Geodetic(globe=ccrs.Globe(datum='WGS84'))
    tiler = GoogleTiles(style="satellite")
except:
    logging.warning("Module cartopy needed for some plots")

from water_or_land import find_body_category

#
class Clustering_Method(object):

    def __init__(self, data, body_label, output_path, cycle, plot='no'):
        """ Initialization of clustering method """
        self.data = data
        self.H, self.W, self.D = self.data.shape
        self.plot = plot
        self.sub_img_obs = None

        self.body_label = body_label
        self.outPath = output_path
        self.cycle = cycle

    def flat_img(self):
        """ Flatten the 4D image and filter out NaN and negative values """
        self.sub_img_flat = self.data.reshape(-1, self.D)
        self.mask_data = ~((self.sub_img_flat == -1e6) | np.isnan(self.sub_img_flat)).any(axis=1)
        self.sub_img_obs = self.sub_img_flat[self.mask_data]

    def normalize_data(self, filter_method='zscore', contamination='auto', random_state=42):
        """
        Normalization of the 4 image variables so none has a weight higher than the other for the clustering

        :param filter_method: String of selected method
        :param contamination: Selected contamination for the Isolation Forest method
        :param random_state: Integer of random state for the Isolation Forest method
        :return:
        """
        logging.info('    Normalizing the 4 parameters (lon, lat, h, sig0) for clustering')
        if self.sub_img_obs is None:
            self.flat_img()

        if filter_method == 'zscore':
            mask_z_height = np.abs(stats.zscore(self.sub_img_obs[:, 0])) < 2
            mask_z_sig0 = np.abs(stats.zscore(self.sub_img_obs[:, 1])) < 2
            self.img_filtered = self.sub_img_obs[mask_z_height & mask_z_sig0]
            self.indices_filtered = np.where(self.mask_data)[0][mask_z_height & mask_z_sig0]

        if filter_method == 'isolation_forest':
            iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
            iso_forest.fit(self.sub_img_obs[:, 0:2])
            outliers = iso_forest.predict(self.sub_img_obs[:, 0:2])
            self.img_filtered = self.sub_img_obs[outliers == 1]
            self.indices_filtered = np.where(self.mask_data)[0][outliers == 1]

        scaler = RobustScaler()
        # Normalize height and sig0
        param_scaled = scaler.fit_transform(self.img_filtered[:, 0:2])
        # Gather normalized height sig0 + lon/lat
        self.sub_img_rshp_scaled = np.hstack((param_scaled, self.img_filtered[:, max(0, len(param_scaled[0])):]))

        if self.plot[0:3] == 'yes':
            fig, axs = plt.subplots(1, 3, layout="constrained", figsize=(10, 6))

            axs[0].scatter(self.sub_img_obs[:, 0], self.sub_img_obs[:, 1], marker='+')
            axs[0].set_xlabel('height')
            axs[0].set_ylabel('sig0')
            axs[0].set_title("Cycle: " + self.cycle + ' and label:  ' + str( self.body_label) + 'raw data')

            axs[1].scatter(self.img_filtered[:, 0], self.img_filtered[:, 1], marker='+')
            axs[1].set_xlabel('height')
            axs[1].set_ylabel('sig0')
            axs[1].set_title('Filtered data')

            axs[2].scatter(self.sub_img_rshp_scaled[:, 0], self.sub_img_rshp_scaled[:, 1], marker='+')
            axs[2].set_xlabel('height')
            axs[2].set_ylabel('sig0')
            axs[2].set_title('Filtered and scaled data')

            if self.plot == 'yes2':
                plt.show()
            else:
                fig.savefig(os.path.join(self.outPath,
                                         f'cycle{self.cycle}/plot_cluster_filt_norm_cycle{self.cycle}_label{self.body_label}.png'))
                plt.close(fig)

    #
    def pca(self):
        """ Principal Component Analysis """
        pca = PCA()
        self.var = np.var(self.sub_img_rshp_scaled, axis=0)
        self.data_pca = pca.fit_transform(self.sub_img_rshp_scaled)
        self.expl_vr = pca.explained_variance_ratio_

    def plotting_clusters(self, method='kmeans'):

        """
        Show, on a basemap, the different clusters determined by the clustering algorithm

        :param method: Name of the clustering method
        """

        gdf_data = gpd.GeoDataFrame(geometry=gpd.points_from_xy(self.sub_img_labeled[:, -3],
                                                                self.sub_img_labeled[:, -2]),
                                    crs="EPSG:4326")

        gdf_data['longitude'] = gdf_data.geometry.get_coordinates().x.values
        gdf_data['latitude'] = gdf_data.geometry.get_coordinates().y.values
        gdf_data['h'] = self.sub_img_labeled[:, 0]
        gdf_data['sig0'] = self.sub_img_labeled[:, 1]

        # Extent
        bnds = gdf_data.total_bounds
        extent = [bnds[0], bnds[2], bnds[1], bnds[3]]

        # Define the colors of labels
        n_clusters = len(list(set(self.labels)))
        if n_clusters == 3:
            colors = np.array(['red', 'teal', 'yellow'])[self.labels]
        else:
            colors = np.array(["#"+''.join([random.choice('0123456789ABCDEF')
                                            for j in range(6)])
                               for i in range(n_clusters)])[self.labels]
        # Plot of clustering method
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=tiler.crs)
        ax.set_extent(extent, geodetic)
        ax.add_image(tiler, 16)
        gdf_data.plot(ax=ax, color=colors, markersize=1.5, alpha=0.7, zorder=5, transform=wgs84)
        gl1 = ax.gridlines(draw_labels=True)
        gl1.top_labels = False
        gl1.right_labels = False
        ax.set_title(f'Clustering results with {n_clusters} clusters to categorize: Label {self.body_label}')
        if self.plot == 'yes2':
            plt.show()
        else:
            fig.savefig(os.path.join(self.outPath,
                                     f'cycle{self.cycle}/plot_cluster_{method}_cycle{self.cycle}_label{self.body_label}.png'))
            plt.close(fig)

        return gdf_data, extent

    #
    def clustering_method(self, method='kmeans'):
        """
        Perform the clustering and separate label into three clusters with Kmeans or Birch

        :param method: Choose between Kmeans or Birch
        """

        logging.info(f'    Performing clustering with algorithm {method}')

        nb_cluster = 3
        random_state = 42

        if method == 'kmeans':
            model = KMeans(n_clusters=nb_cluster, random_state=random_state, n_init="auto").fit(self.sub_img_rshp_scaled)
            self.centroids = model.cluster_centers_

        if method == 'birch':
            model = Birch(n_clusters=nb_cluster).fit(self.sub_img_rshp_scaled)
            self.centroids = model.subcluster_centers_

        self.labels = model.labels_
        self.sub_img_labeled = np.column_stack((self.img_filtered, self.labels))

        if self.plot[0:3] == 'yes':

            gdf_data, extent = self.plotting_clusters(method=method)

            median_h = np.median(gdf_data.h.values)
            median_s = np.median(gdf_data.sig0.values)
            var_h = np.std(gdf_data.h.values)
            var_s = np.std(gdf_data.sig0.values)
            hmin = median_h-var_h
            hmax = median_h+var_h
            smin = median_s-var_s
            smax = median_s+var_s

            # Plots of height and sig0 after clustering filtering
            fig = plt.figure(figsize=(15, 7))
            ax1 = fig.add_subplot(121, projection=tiler.crs)
            ax1.set_extent(extent, geodetic)
            ax1.add_image(tiler, 16)
            cs = gdf_data.plot(ax=ax1, kind='scatter', x='longitude', y='latitude', c='h', vmin=hmin, vmax=hmax,
                               s=1.5, zorder=5, transform=wgs84, colorbar=False)
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=hmin, vmax=hmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax1, shrink=0.35, aspect=30, label='Height')
            gl2 = ax1.gridlines(draw_labels=True)
            gl2.top_labels = False
            gl2.right_labels = False
            ax1.set_title("Height after filtering")

            ax2 = fig.add_subplot(122, projection=tiler.crs)
            ax2.set_extent(extent, geodetic)
            ax2.add_image(tiler, 16)
            cs2 = gdf_data.plot(ax=ax2, kind='scatter', x='longitude', y='latitude', c='sig0', vmin=smin, vmax=smax,
                                s=1.5, zorder=5, transform=wgs84, colorbar=False)
            sm2 = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=smin, vmax=smax))
            sm2.set_array([])
            cbar2 = plt.colorbar(sm2, ax=ax2, shrink=0.35, aspect=30, label='Sig0')
            gl3 = ax2.gridlines(draw_labels=True)
            gl3.top_labels = False
            gl3.right_labels = False
            ax2.set_title("Sig0 after filtering")
            if self.plot == 'yes2':
                plt.show()
            else:
                fig.savefig(os.path.join(self.outPath,
                                         f'cycle{self.cycle}/plot_cluster_cycle{self.cycle}_h_sig0_label{self.body_label}.png'))
                plt.close(fig)

    #
    def tsne(self):
        """ Dimension reduction using TSNE algorithm """
        np.random.seed(42)
        if self.sub_img_rshp_scaled.shape[0] < 100_000:
            size_ech = self.sub_img_rshp_scaled.shape[0]
        else:
            logging.warning("The sample size is reduced because it is bigger than 100_000")
            size_ech = 100_000
        self.indices_ssech = np.random.choice(self.sub_img_rshp_scaled.shape[0], size=size_ech, replace=False)
        self.sub_img_tsne = self.sub_img_rshp_scaled[self.indices_ssech]
        tsne = TSNE(n_components=2, perplexity=40, random_state=42, init='pca')
        self.data_tsne = tsne.fit_transform(self.sub_img_tsne)

    #
    def clustering_tsne_hdbscan(self):
        """ Clustering on TSNE new dimension using HDBSCAN """
        model_tsne = HDBSCAN(min_cluster_size=int(self.data_tsne.shape[0]*0.01)).fit(self.data_tsne)
        self.labels = model_tsne.labels_
        self.sub_img_labeled = np.column_stack((self.sub_img_tsne, self.labels))

        # Plot
        gdf_data, extent = self.plotting_clusters(method='hdbscan_tsne')

    #
    def umap(self):
        """ Dimension reduction using UMAP algorithm """
        if len(self.sub_img_rshp_scaled) < 300_000:
            size_ech = len(self.sub_img_rshp_scaled)
        else:
            logging.warning("The sample size is reduced because it is bigger than 300_000")
            size_ech = 300_000
        self.indices_ssech_umap = np.random.choice(self.sub_img_rshp_scaled.shape[0], size=size_ech, replace=False)
        self.sub_img_umap = self.sub_img_rshp_scaled[self.indices_ssech_umap]
        self.data_umap = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1).fit_transform(self.sub_img_umap)

    #
    def clustering_umap_hdbscan(self):
        """ Clustering on UMAP new dimension using HDBSCAN """
        model_umap = HDBSCAN(min_cluster_size=int(self.data_umap.shape[0]*0.01)).fit(self.data_umap)
        self.labels = model_umap.labels_
        self.sub_img_labeled = np.column_stack((self.sub_img_umap, self.labels))

        # Plot
        gdf_data, extent = self.plotting_clusters(method='hdbscan_umap')

    #
    def get_water_soil_labels(self, water_extract, pekel_0_100_poly, poly_sword, method='kmeans'):
        """
        :param water_extract:
        :param pekel_0_100_poly:
        :param poly_sword:
        """
        logging.info('    Get the labels for soil and for water')

        labels_present = list(set(self.labels))

        # Initialize
        self.sub_img_labeled_land = None
        self.sub_img_labeled_water = None
        water_body_type = {'LAND': [], 'WATER': []}

        # Loop over the clusters labels
        for k, lab in enumerate(labels_present):
            if lab != -1:
                # Retrieve array only for the cluster label
                sub_array = self.sub_img_labeled[self.sub_img_labeled[:, -1] == lab]
                # Transform into a GeoDataFrame
                geom = gpd.points_from_xy(sub_array[:, 2], sub_array[:, 3])
                water_filtered = water_extract[water_extract.geometry.isin(geom)]
                # Categorize the cluster label
                cat, cat_params = find_body_category(water_filtered, pekel_0_100_poly, poly_sword,
                                                     plot='no', choices=['WATER', 'LAND'])
                water_body_type[cat].append(int(lab))
        # print(f'    water_body_type:  {water_body_type}')

        if water_body_type['WATER'] != []:
            self.sub_img_labeled_water = self.sub_img_labeled[np.isin(self.sub_img_labeled[:, 4],
                                                                      water_body_type['WATER'])]
        if water_body_type['LAND'] != []:
            self.sub_img_labeled_land = self.sub_img_labeled[np.isin(self.sub_img_labeled[:, 4],
                                                                     water_body_type['LAND'])]

        if self.plot[0:3] == 'yes':
            if self.sub_img_labeled_water is not None:
                gdf_dataw = gpd.GeoDataFrame(geometry=gpd.points_from_xy(self.sub_img_labeled_water[:, -3],
                                                                         self.sub_img_labeled_water[:, -2]),
                                             crs="EPSG:4326")
                gdf_dataw['longitude'] = gdf_dataw.geometry.get_coordinates().x.values
                gdf_dataw['latitude'] = gdf_dataw.geometry.get_coordinates().y.values
            else:
                gdf_dataw = gpd.GeoDataFrame()
            if self.sub_img_labeled_land is not None:
                gdf_datal = gpd.GeoDataFrame(geometry=gpd.points_from_xy(self.sub_img_labeled_land[:, -3],
                                                                         self.sub_img_labeled_land[:, -2]),
                                             crs="EPSG:4326")
                gdf_datal['longitude'] = gdf_datal.geometry.get_coordinates().x.values
                gdf_datal['latitude'] = gdf_datal.geometry.get_coordinates().y.values
            else:
                gdf_datal = gpd.GeoDataFrame()
            gdf_data = pd.concat([gdf_dataw, gdf_datal])

            # Extent
            bnds = gdf_data.total_bounds
            extent = [bnds[0], bnds[2], bnds[1], bnds[3]]

            # Plot of clustering method after categorization of clusters
            fig = plt.figure()
            ax = fig.add_subplot(111, projection=tiler.crs)
            ax.set_extent(extent, geodetic)
            ax.add_image(tiler, 16)
            if self.sub_img_labeled_land is not None:
                gdf_datal.plot(ax=ax, color='r', markersize=1.5, alpha=0.7, zorder=5, transform=wgs84, label='land')
            if self.sub_img_labeled_water is not None:
                gdf_dataw.plot(ax=ax, color='cyan', markersize=1.5, alpha=0.7, zorder=5, transform=wgs84, label='water')
            gl1 = ax.gridlines(draw_labels=True)
            gl1.top_labels = False
            gl1.right_labels = False
            ax.set_title(f'Clustering results after categorization of clusters: Label {self.body_label}')
            ax.legend()
            if self.plot == 'yes2':
                plt.show()
            else:
                fig.savefig(os.path.join(self.outPath,
                                         f'cycle{self.cycle}/plot_cluster_lw_{method}_cycle{self.cycle}_label{self.body_label}.png'))
                plt.close(fig)
