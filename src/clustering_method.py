# -*- coding: utf8 -*-
"""
Calculate direct method using clustering algorithm (KMeans)
When the identified water body is classified as 'WATER_LAND'

Copyright (c) 2018, CNES
"""

import os
import logging
import pandas as pd
import geopandas as gpd
import numpy as np
from skimage import morphology

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, Birch, HDBSCAN
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.ensemble import IsolationForest

import umap
from umap import UMAP

import scipy
from scipy import stats
from scipy.spatial import distance, Delaunay

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
        self.data = data
        self.H, self.W, self.D = self.data.shape
        self.plot = plot
        self.sub_img_obs = None

        self.body_label = body_label
        self.outPath = output_path
        self.cycle = cycle

    #
    def flat_img(self):
        self.sub_img_flat = self.data.reshape(-1, self.D)
        self.mask_data = ~((self.sub_img_flat == -1e6) | np.isnan(self.sub_img_flat)).any(axis=1)
        self.sub_img_obs = self.sub_img_flat[self.mask_data]
    #
    def normalize_data(self, filter_method='zscore', contamination='auto', random_state=42):
        """
        :param filter_method:
        :param contamination:
        :param random_state:
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

            fig.savefig(os.path.join(self.outPath,
                                     f'cycle{self.cycle}/plot_clust_cycle{self.cycle}_bodyLabel{self.body_label}.png'))
            plt.close(fig)

    #
    def pca(self):
        pca = PCA()
        self.var = np.var(self.sub_img_rshp_scaled, axis=0)
        self.data_pca = pca.fit_transform(self.sub_img_rshp_scaled)
        self.expl_vr = pca.explained_variance_ratio_

    #
    def tsne(self):
        np.random.seed(42)
        if self.sub_img_rshp_scaled.shape[0] < 100_000:
            size_ech = self.sub_img_rshp_scaled.shape[0]

        else:
            size_ech = 100_000
        self.indices_ssech = np.random.choice(self.sub_img_rshp_scaled.shape[0], size=size_ech, replace=False)
        self.sub_img_tsne = self.sub_img_rshp_scaled[self.indices_ssech]
        tsne = TSNE(n_components=2, perplexity=40, random_state=42, init='pca')
        self.data_tsne = tsne.fit_transform(self.sub_img_tsne)
        if self.plot[0:3] == 'yes':
            plt.figure(figsize=(6, 4))
            plt.scatter(self.data_tsne[:, 0], self.data_tsne[:, 1], marker='+')
            plt.title('TSNE')
            if self.plot == 'yes2':
                plt.show()

    #
    def umap(self):
        if len(self.sub_img_rshp_scaled) < 300_000:
            size_ech = len(self.sub_img_rshp_scaled)
        else:
            size_ech = 300_000
        self.indices_ssech_umap = np.random.choice(self.sub_img_rshp_scaled.shape[0], size=size_ech, replace=False)
        self.sub_img_umap = self.sub_img_rshp_scaled[self.indices_ssech_umap]
        self.data_umap = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1).fit_transform(self.sub_img_umap)
        if self.plot[0:3] == 'yes':
            plt.figure(figsize=(6, 4))
            plt.scatter(self.data_umap[:, 0], self.data_umap[:, 1], s=5, alpha=0.6)
            plt.title('UMAP')
            if self.plot == 'yes2':
                plt.show()

    #
    def clustering_method(self, method='kmeans'):
        """

        :param method:
        :return:
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

        gdf_data = gpd.GeoDataFrame(geometry=gpd.points_from_xy(self.sub_img_labeled[:, -3],
                                                                self.sub_img_labeled[:, -2]),
                                    crs="EPSG:4326")
        gdf_data['clustLabel'] = self.labels

        if self.plot[0:3] == 'yes':

        #     # TODO: Delete the below once the debug and parametrization phases are done
        #     gdf_data.to_file(os.path.join(self.outPath,
        #                                   f'cycle{self.cycle}/clust_results_cycle{self.cycle}_bodyLabel{self.body_label}.shp'))

            gdf_data['longitude'] = gdf_data.geometry.get_coordinates().x.values
            gdf_data['latitude'] = gdf_data.geometry.get_coordinates().y.values
            gdf_data['h'] = self.sub_img_labeled[:, 0]
            gdf_data['sig0'] = self.sub_img_labeled[:, 1]

            median_h = np.median(gdf_data.h.values)
            median_s = np.median(gdf_data.sig0.values)
            var_h = np.std(gdf_data.h.values)
            var_s = np.std(gdf_data.sig0.values)
            hmin = median_h-var_h
            hmax = median_h+var_h
            smin = median_s-var_s
            smax = median_s+var_s

            # Plots extent
            bnds = gdf_data.total_bounds
            extent = [bnds[0], bnds[2], bnds[1], bnds[3]]

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
            ax1.set_title("Height after clustering filtering")

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
            ax2.set_title("Sig0 after clustering filtering")
            if self.plot == 'yes2':
                plt.show()
            fig.savefig(os.path.join(self.outPath,
                                     f'cycle{self.cycle}/plot_clust_cycle{self.cycle}_h_sig0_bodyLabel{self.body_label}.png'))
            plt.close(fig)

            # Plot of clustering method
            colors = np.array(['red', 'teal', 'yellow'])[self.labels]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection=tiler.crs)
            ax.set_extent(extent, geodetic)
            ax.add_image(tiler, 16)
            gdf_data.plot(ax=ax, color=colors, markersize=1.5, alpha=0.7, zorder=5, transform=wgs84)
            gl1 = ax.gridlines(draw_labels=True)
            gl1.top_labels = False
            gl1.right_labels = False
            ax.set_title(f'Clustering results: Label {self.body_label}')
            if self.plot == 'yes2':
                plt.show()
            fig.savefig(os.path.join(self.outPath,
                                     f'cycle{self.cycle}/plot_clust_cycle{self.cycle}_results_bodyLabel{self.body_label}.png'))
            plt.close(fig)

    #
    def clustering_tsne_hdbscan(self):
        model_tsne = HDBSCAN(min_cluster_size=int(self.data_tsne.shape[0]*0.01)).fit(self.data_tsne)
        self.labels_tsne = model_tsne.labels_

        if self.plot[0:3] == 'yes':
            unique_labels = np.unique(self.labels_tsne)
            cmap = plt.cm.get_cmap('viridis', len(unique_labels))
            colors = cmap(np.searchsorted(unique_labels, self.labels_tsne))
            gdf_data_tsne = gpd.GeoDataFrame(geometry=gpd.points_from_xy(self.sub_img_labeled[self.indices_ssech, -3],
                                                                         self.sub_img_labeled[self.indices_ssech, -2]),
                                             crs="EPSG:4326")
            gdf_data_tsne = gdf_data_tsne.to_crs(epsg=3857)
            fig, ax = plt.subplots(figsize=(12, 6))
            gdf_data_tsne.plot(ax=ax, color=colors, markersize=1, alpha=0.7)

            norm = Normalize(vmin=0, vmax=len(unique_labels) - 1)
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Labels')
            if self.plot == 'yes2':
                plt.show()

            cmap = plt.cm.viridis
            plt.scatter(self.data_tsne[:, 0], self.data_tsne[:, 1], c=self.labels_tsne, marker='+', cmap=cmap)
            if self.plot == 'yes2':
                plt.show()

    #
    def clustering_umap_hdbscan(self):
        """

        """
        model_umap = HDBSCAN(min_cluster_size=int(self.data_umap.shape[0]*0.01)).fit(self.data_umap)
        self.labels_umap = model_umap.labels_

        mask_umap_valid = self.labels_umap != -1
        mask_indice = self.indices_ssech_umap[mask_umap_valid]

        if self.plot[0:3] == 'yes':
            unique_labels = np.unique(self.labels_umap)
            cmap = plt.cm.get_cmap('viridis', len(unique_labels))
            colors = cmap(np.searchsorted(unique_labels, self.labels_umap))
            gdf_data_umap = gpd.GeoDataFrame(geometry=gpd.points_from_xy(self.sub_img_labeled[self.indices_ssech_umap, -3],
                                                                         self.sub_img_labeled[self.indices_ssech_umap, -2]),
                                             crs="EPSG:4326")
            gdf_data_umap = gdf_data_umap.to_crs(epsg=3857)
            fig, ax = plt.subplots(figsize=(12, 6))
            gdf_data_umap.plot(ax=ax, color=colors, markersize=1, alpha=0.7)

            norm = Normalize(vmin=0, vmax=len(unique_labels) - 1)
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Labels')
            if self.plot == 'yes2':
                plt.show()

            cmap = plt.cm.viridis
            plt.scatter(self.data_umap[:, 0], self.data_umap[:, 1], c=self.labels_umap, marker='+', cmap=cmap)
            if self.plot == 'yes2':
                plt.show()

    #
    def get_water_soil_labels(self, water_extract, pekel_0_100_poly, poly_sword):
        """
        :param water_extract:
        :param pekel_0_100_poly:
        :param poly_sword:
        :return:
        """
        logging.info('    Get the labels for soil and for water')

        labels_present = list(set(self.labels))
        nb_labels = len(labels_present)
        # print(f'nb_labels: {nb_labels}, {[[list(self.labels).count(i), int(i)] for i in set(self.labels)]}')

        self.sub_img_labeled_land = None
        self.sub_img_labeled_water = None
        water_body_type = {'LAND': [], 'WATER': []}
        for k, lab in enumerate(labels_present):
            sub_array = self.sub_img_labeled[self.sub_img_labeled[:, -1] == lab]
            geom = gpd.points_from_xy(sub_array[:, 2], sub_array[:, 3])
            water_filtered = water_extract[water_extract.geometry.isin(geom)]
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
