# -*- coding: utf8 -*-
'''
Create raster

Copyright (c) 2018, CNES
'''

import os
import logging
import argparse
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import shapefile
import shapely.geometry as shpgeo
import pyproj
import rasterio
from rasterio.io import MemoryFile
from scipy import interpolate
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import utm

import my_rdf_file as my_rdf

from names import FPDEM_BASENAME, FPDEM_RASTER_BASENAME, FPDEM_POINTCLOUD_BASENAME, MASK_SUFFIX, compute_name

from process import filter_elevation, compute_mean_3sigma

from nc import write_raster_gridded

class FPDEM_Raster(object):
    """
    Class FPDEM_Raster
    Main class to compute HR_FPDEM product
    """
    def __init__(self, param, input_file=None, output_file=None, mask=None):
        """
        Constructor: initialize variables

        :param param: input parameters to run the processor
        :type param: dictionary
        :param input_file: Input filename with the FPDEM results (ungridded file)
        :param output_file: Output filename
        :param mask: Filename of the extract area process result
        """

        if param:
            self.output_directory = param.getValue("output directory").split(" ")[0]

        if input_file is None:
            self.input_file = compute_name(self.output_directory, FPDEM_POINTCLOUD_BASENAME,
                                           param.getValue("tile name"),
                                           param.getValue("first date name"),
                                           param.getValue("last date name"))
        else:
            self.input_file = input_file

        if output_file is None:
            self.output_file = compute_name(self.output_directory, FPDEM_RASTER_BASENAME,
                                            param.getValue("tile name"),
                                            param.getValue("first date name"),
                                            param.getValue("last date name"))
        else:
            self.output_file = output_file

        if mask is None:
            self.mask = os.path.join(self.output_directory, FPDEM_BASENAME + MASK_SUFFIX)
        else:
            self.mask = mask

        if param:
            # AOI
            self.lat_max = float(param.getValue("latitude max").split(" ")[0])
            self.lon_max = float(param.getValue("longitude max").split(" ")[0])
            self.lat_min = float(param.getValue("latitude min").split(" ")[0])
            self.lon_min = float(param.getValue("longitude min").split(" ")[0])
            # Raster parameters
            self.resolution = float(param.getValue("resolution").split(" ")[0])
            self.mode = param.getValue("mode").split(" ")[0]
            # Plot
            self.plot = param.getValue("plot").split(" ")[0]

        self.epsg = 4326

    def load_input_fpdem_raster(self):
        """
        Loading the FPDEM ungridded netcdf using xarray
        """
        cloud_xr = xr.open_dataset(self.input_file)
        self.cloud_df_raster = cloud_xr.to_dataframe()

    def compute_fpdem_raster(self):
        self.compute_raster_exotic()

    def compute_raster_exotic(self):
        """
        Compute all variables for the raster file and plot them in a png figure.
        """
        # Extract xyz
        if self.mode == 'utm':
            nb = len(self.cloud_df_raster['x'])
            xyz_data = np.ndarray((nb, 3))
            xyz_data[:, 0] = self.cloud_df_raster['x']
            xyz_data[:, 1] = self.cloud_df_raster['y']
        # Extract latlon
        elif self.mode == 'latlon':
            nb = len(self.cloud_df_raster['longitude'])
            lonlat_data = np.ndarray((nb, 3))
            lonlat_data[:, 0] = self.cloud_df_raster['longitude']
            lonlat_data[:, 1] = self.cloud_df_raster['latitude']

        mean_lat = np.mean(self.cloud_df_raster['latitude'])
        mean_lon = np.mean(self.cloud_df_raster['longitude'])
        utm_coords = utm.from_latlon(mean_lat, mean_lon)
        self.zone_number = utm_coords[2]
        self.zone_letter = utm_coords[3]
        print('zone nb and letter:', self.zone_number, self.zone_letter)

        elevation = self.cloud_df_raster['elevation']
        if self.mode == 'latlon':
            latitude = self.cloud_df_raster['latitude']
            longitude = self.cloud_df_raster['longitude']
            geom = gpd.points_from_xy(longitude, latitude)
        elif self.mode == 'utm':
            xutm = self.cloud_df_raster['x']
            yutm = self.cloud_df_raster['y']
            geom = gpd.points_from_xy(xutm, yutm)
        gdf_xy = gpd.GeoDataFrame(geom, columns=['geometry'], crs=4326)
        gdf_xy['elevation'] = elevation
        if self.mode == 'latlon':
            gdf_xy['lon'] = longitude
            gdf_xy['lat'] = latitude
        elif self.mode == 'utm':
            gdf_xy['x'] = xutm
            gdf_xy['y'] = yutm

        # Compute grid
        if self.mode == 'utm':
            # Interpolation
            x0 = min(xyz_data[:, 0])
            y0 = min(xyz_data[:, 1])
            nx = abs(int((x0 - max(xyz_data[:, 0])) / self.resolution)) + 1
            ny = abs(int((y0 - max(xyz_data[:, 1])) / self.resolution)) + 1
        elif self.mode == 'latlon':
            # x0 = min(lonlat_data[:, 0])
            # y0 = min(lonlat_data[:, 1])
            # nx = abs(int((x0 - max(lonlat_data[:, 0])) / self.resolution)) + 1
            # ny = abs(int((y0 - max(lonlat_data[:, 1])) / self.resolution)) + 1
            x0 = self.lon_min
            y0 = self.lat_min
            nx = abs(int((x0 - self.lon_max) / self.resolution)) + 1
            ny = abs(int((y0 - self.lat_max) / self.resolution)) + 1

        x1 = x0 + self.resolution * (nx - 1)
        y1 = y0 + self.resolution * (ny - 1)
        x = np.linspace(x0, x1, nx)
        y = np.linspace(y0, y1, ny)

        # Grid preparation
        GRID = np.meshgrid(x, y)
        grid_shape = GRID[0].shape
        GRID = np.reshape(GRID, (2, -1)).T

        # Filter elevation points with 3sigma filtering for each cell of GRID, or another method,
        #
        # Associate all points to a cell
        if self.mode == 'latlon':
            gdf_xy['ix'] = ((gdf_xy['lon'] - x[0]) // (x[1] - x[0])).astype(int)
            gdf_xy['iy'] = ((gdf_xy['lat'] - y[0]) // (y[1] - y[0])).astype(int)
        elif self.mode == 'utm':
            gdf_xy['ix'] = ((gdf_xy['x'] - x[0]) // (x[1] - x[0])).astype(int)
            gdf_xy['iy'] = ((gdf_xy['y'] - y[0]) // (y[1] - y[0])).astype(int)
        gdf_xy['cell_id'] = gdf_xy['ix'] * len(y) + gdf_xy['iy']
        # Group the dataframe rows by cell_id
        grouped = gdf_xy.groupby('cell_id')
        # Apply operation to each group
        results = []
        for cell_id, group in grouped:
            if len(group) >= 5:
                if len(group) >= 5 and len(group) < 30:
                    results.append(filter_elevation(group, 'mad'))
                else:
                    results.append(compute_mean_3sigma(group))
            else:
                results.append(group)
        #
        gdf_elev = pd.concat(results, ignore_index=True).drop_duplicates()
        nb = len(gdf_elev)
        logging.info(f'The filtering (MAD, 3sigma) removed {len(gdf_xy) - nb} points out of {len(gdf_xy)}.')

        # Retrieve variables a second time after filtering
        # Extract xyz
        if self.mode == 'utm':
            xyz_data2 = np.ndarray((nb, 3))
            xyz_data2[:, 0] = gdf_elev.x
            xyz_data2[:, 1] = gdf_elev.y
        # Extract latlon
        elif self.mode == 'latlon':
            lonlat_data2 = np.ndarray((nb, 3))
            lonlat_data2[:, 0] = gdf_elev.lon
            lonlat_data2[:, 1] = gdf_elev.lat
        # Train
        if self.mode == 'utm':
            idw_tree = cKDTree(xyz_data2[:, 0:2])
        elif self.mode == 'latlon':
            idw_tree = cKDTree(lonlat_data2[:, 0:2])

        nb_points = 10
        distances, results = idw_tree.query(GRID, k=nb_points)

        # With filtering
        elev_array = gdf_elev['elevation'].to_numpy()
        if results.max() >= len(elev_array) or results.min() < 0:
            raise ValueError("Results contains indices out of limits compared to gdf_elev")
        elevation_tab = elev_array[results]

        # Initializing variables
        z = np.zeros(len(results), dtype=float)
        z_rel = np.zeros(len(results), dtype=float)
        distances_filtered = np.where(distances < self.resolution, distances, np.nan)
        dist_mean = np.nanmean(distances_filtered, axis=1)
        dist_min = np.nanmin(distances_filtered, axis=1)
        z_mean = np.nanmean(elevation_tab, axis=1)

        # Filter outliers pixels (whose height is too different from their averaging height around)
        threshold_outlier = 1.
        ind_to_remove = np.where(
            np.abs(elevation_tab - np.tile(z_mean, (elevation_tab.shape[1], 1)).T) > threshold_outlier)
        elevation_tab[ind_to_remove] = np.nan
        distances_filtered[ind_to_remove] = np.nan

        # Compute Z and Z_rel (height weighted with the distance to center)
        z = np.nansum(elevation_tab / distances_filtered, axis=1) / np.nansum(1. / distances_filtered, axis=1)
        z = np.nan_to_num(z, copy=True, nan=0.0, posinf=None, neginf=None)
        z_rel = np.abs(np.nanstd(elevation_tab, axis=1) / z_mean)
        z_rel = np.nan_to_num(z_rel, copy=True, nan=0.0, posinf=None, neginf=None)
        dist_mean = np.nan_to_num(dist_mean, copy=True, nan=0.0, posinf=None, neginf=None)
        dist_min = np.nan_to_num(dist_min, copy=True, nan=0.0, posinf=None, neginf=None)

        # Reproject into 2d grid
        z2d = z.reshape(grid_shape)
        dist_mean_2d = dist_mean.reshape(grid_shape)
        dist_min_2d = dist_min.reshape(grid_shape)
        z_rel_2d = z_rel.reshape(grid_shape)

        # Determine good (no empty) points to use
        good_points = np.where(z2d != 0)
        points = np.dstack((good_points[0], good_points[1]))
        grid_x, grid_y = np.mgrid[range(z2d.shape[0]), range(z2d.shape[1])]

        # Compute cubic interpolation
        z2d = interpolate.griddata(points[0], z2d[np.where(z2d != 0)], (grid_x, grid_y), method='cubic')

        # Compute quality flag
        qual_flag_2d = compute_qual_flag(dist_mean_2d, z_rel_2d)

        # Read the area polygon defined in the extract_area step
        mask = shapefile.Reader(self.mask)

        # Define the projection according to provided EPSG
        if self.mode == 'latlon':
            proj = pyproj.Proj(init='EPSG:' + str(self.epsg))
        elif self.mode == 'utm':
            hemisphere = "N" if self.zone_letter.upper() >= "N" else "S"  # N to X -> North ; C to M -> South
            epsg = (32600 if hemisphere.upper() == "N" else 32700) + self.zone_number
            proj = pyproj.Proj(init='EPSG:' + str(epsg))

        # Compute mask in latlon and utm coordinates
        extracted_area_utm = []
        extracted_area_latlon = []
        for extracted_area in mask.shapeRecords():
            extracted_area_utm.append(toFromUTM(extracted_area.shape, proj))
            extracted_area_latlon.append(extracted_area.shape)

        transform = rasterio.transform.from_origin(min(x), min(y), self.resolution, -self.resolution)

        # Filter data out of mask
        if self.mode == 'utm':
            with MemoryFile() as memfile:
                with memfile.open(driver='GTiff', height=z2d.shape[0], width=z2d.shape[1], count=1, dtype=z2d.dtype,
                                  crs='EPSG:' + str(self.epsg), transform=transform) as src:
                    src.write(z2d, 1)
                    out_image, out_transform = rasterio.mask.mask(src, extracted_area_utm, crop=False,
                                                                  indexes=1, nodata=np.nan)

            with MemoryFile() as memfile:
                with memfile.open(driver='GTiff', height=dist_min_2d.shape[0], width=dist_min_2d.shape[1], count=1,
                                  dtype=dist_min_2d.dtype, crs='EPSG:' + str(self.epsg), transform=transform) as src:
                    src.write(dist_min_2d, 1)
                    out_dist_min_2d, out_transform = rasterio.mask.mask(src, extracted_area_utm, crop=False,
                                                                        indexes=1, nodata=np.nan)

            with MemoryFile() as memfile:
                with memfile.open(driver='GTiff', height=dist_mean_2d.shape[0], width=dist_mean_2d.shape[1], count=1,
                                  dtype=dist_mean_2d.dtype, crs='EPSG:' + str(self.epsg), transform=transform) as src:
                    src.write(dist_mean_2d, 1)
                    out_dist_mean_2d, out_transform = rasterio.mask.mask(src, extracted_area_utm, crop=False,
                                                                         indexes=1, nodata=np.nan)

            with MemoryFile() as memfile:
                with memfile.open(driver='GTiff', height=z_rel_2d.shape[0], width=z_rel_2d.shape[1], count=1,
                                  dtype=z_rel_2d.dtype, crs='EPSG:' + str(self.epsg), transform=transform) as src:
                    src.write(z_rel_2d, 1)
                    out_z_rel_2d, out_transform = rasterio.mask.mask(src, extracted_area_utm, crop=False,
                                                                     indexes=1, nodata=np.nan)

            with MemoryFile() as memfile:
                with memfile.open(driver='GTiff', height=qual_flag_2d.shape[0], width=qual_flag_2d.shape[1], count=1,
                                  dtype=qual_flag_2d.dtype, crs='EPSG:' + str(self.epsg), transform=transform) as src:
                    src.write(qual_flag_2d, 1)
                    out_qual_flag_2d, out_transform = rasterio.mask.mask(src, extracted_area_utm, crop=False, indexes=1,
                                                                         nodata=np.nan)

        elif self.mode == 'latlon':
            with MemoryFile() as memfile:
                with memfile.open(driver='GTiff', height=z2d.shape[0], width=z2d.shape[1], count=1, dtype=z2d.dtype,
                                  crs='EPSG:' + str(self.epsg), transform=transform) as src:
                    src.write(z2d, 1)
                    out_image, out_transform = rasterio.mask.mask(src, extracted_area_latlon, crop=False,
                                                                  indexes=1, nodata=np.nan)

            with MemoryFile() as memfile:
                with memfile.open(driver='GTiff', height=dist_min_2d.shape[0], width=dist_min_2d.shape[1], count=1,
                                  dtype=dist_min_2d.dtype, crs='EPSG:' + str(self.epsg), transform=transform) as src:
                    src.write(dist_min_2d, 1)
                    out_dist_min_2d, out_transform = rasterio.mask.mask(src, extracted_area_latlon, crop=False,
                                                                        indexes=1, nodata=np.nan)

            with MemoryFile() as memfile:
                with memfile.open(driver='GTiff', height=dist_mean_2d.shape[0], width=dist_mean_2d.shape[1], count=1,
                                  dtype=dist_mean_2d.dtype, crs='EPSG:' + str(self.epsg), transform=transform) as src:
                    src.write(dist_mean_2d, 1)
                    out_dist_mean_2d, out_transform = rasterio.mask.mask(src, extracted_area_latlon, crop=False,
                                                                         indexes=1, nodata=np.nan)

            with MemoryFile() as memfile:
                with memfile.open(driver='GTiff', height=z_rel_2d.shape[0], width=z_rel_2d.shape[1], count=1,
                                  dtype=z_rel_2d.dtype, crs='EPSG:' + str(self.epsg), transform=transform) as src:
                    src.write(z_rel_2d, 1)
                    out_z_rel_2d, out_transform = rasterio.mask.mask(src, extracted_area_latlon, crop=False,
                                                                     indexes=1, nodata=np.nan)

            with MemoryFile() as memfile:
                with memfile.open(driver='GTiff', height=qual_flag_2d.shape[0], width=qual_flag_2d.shape[1], count=1,
                                  dtype=qual_flag_2d.dtype, crs='EPSG:' + str(self.epsg), transform=transform) as src:
                    src.write(qual_flag_2d, 1)
                    out_qual_flag_2d, out_transform = rasterio.mask.mask(src, extracted_area_latlon, crop=False,
                                                                         indexes=1, nodata=np.nan)

        self.x = x
        self.y = y
        self.out_image = out_image
        self.out_dist_min_2d = out_dist_min_2d
        self.out_dist_mean_2d = out_dist_mean_2d
        self.out_qual_flag_2d = out_qual_flag_2d

        # Plotting
        if self.plot[0:3] == 'yes':
            fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(18, 18))
            if self.mode == 'utm':
                mat1 = ax1.scatter(xyz_data[:, 0], xyz_data[:, 1], s=2, c=self.cloud_df_raster['elevation'],
                                   linewidths=0,
                                   cmap="RdBu", vmin=np.min(z), vmax=np.max(z))
            elif self.mode == 'latlon':
                mat1 = ax1.scatter(lonlat_data[:, 0], lonlat_data[:, 1], s=2, c=self.cloud_df_raster['elevation'],
                                   linewidths=0,
                                   cmap="RdBu", vmin=np.min(z), vmax=np.max(z))
            mat2 = ax2.contourf(x, y, out_image, 100, cmap="RdBu", vmin=np.nanmin(out_image), vmax=np.nanmax(out_image))
            mat3 = ax3.contourf(x, y, out_z_rel_2d, 100, cmap="RdBu", vmin=np.nanmin(out_z_rel_2d),
                                vmax=np.nanmax(out_z_rel_2d))
            mat4 = ax4.contourf(x, y, out_qual_flag_2d, 100, cmap="RdBu", vmin=np.nanmin(out_qual_flag_2d),
                                vmax=np.nanmax(out_qual_flag_2d))
            ax1.set_title('Point cloud')
            ax2.set_title('Raster (iwd interpolation)')
            ax3.set_title('Relative elevation variance')
            ax4.set_title('Flag')
            fig.colorbar(mat1, label="Elevation (m)", orientation="vertical", ax=ax1)
            fig.colorbar(mat2, label="Elevation (m)", orientation="vertical", ax=ax2)
            fig.colorbar(mat3, label="Relative elevation variance", orientation="vertical", ax=ax3)
            fig.colorbar(mat4, label="Flag", orientation="vertical", ax=ax4)
            if self.plot == 'yes2':
                plt.show()
            else:
                fig.savefig(f'{os.path.join(self.output_directory, "plot_raster_variables.png")}')
                plt.close(fig)

    def write_fpdem_raster(self):
        """
        Write output file
        """
        write_raster_gridded(self.output_file, self.mode, self.x, self.y, self.out_image,
                             self.out_dist_min_2d, self.out_dist_mean_2d, self.out_qual_flag_2d,
                             self.resolution, str(self.epsg),
                             zone_number=int(self.zone_number), zone_letter=str(self.zone_letter))


def compute_qual_flag(mean_dist, mean_z_rel):
    """
    Compute the quality of a raster pixel depending on distance to neighbors points and mean elevation

    :param mean_dist: Mean of distances between neighbors
    :param mean_z_rel: Mean of relative elevation
    :return: Array of flags
    """
    flag = np.zeros_like(mean_z_rel)
    flag[:] = 1
    flag[np.where(mean_z_rel > 0.1)] = 2
    flag[np.where(mean_dist > 500.)] = 3
    flag[np.logical_and(mean_z_rel > 0.30, mean_dist > 500.)] = 4

    return flag

def toFromUTM(shp, proj, inv=False):
    """
    Convert to utm coordinates

    :param shp: Shapefile
    :param proj: Projection
    :param inv:
    :return: Shapefile with utm coordinates
    """
    geointerface = shp.__geo_interface__
    shptype = geointerface['type']
    coords = geointerface['coordinates']
    if shptype == 'Polygon':
        newcoord = [[proj(*point, inverse=inv) for point in linring] for linring in coords]
    elif shptype == 'MultiPolygon':
        newcoord = [[[proj(*point, inverse=inv) for point in linring] for linring in poly] for poly in coords]

    return shpgeo.shape({'type': shptype, 'coordinates': tuple(newcoord)})

# Main program
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute fpdem raster product")
    parser.add_argument("parameter_file", help="parameter_file (*.rdf)")
    args = parser.parse_args()
    parameters = my_rdf.myRdfReader(args.parameter_file)

    level = getattr(logging, "INFO")
    logging.basicConfig(filename=None, format='%(asctime)s [%(levelname)s] %(message)s', level=level)

    fpdem_raster = FPDEM_Raster(parameters)
    fpdem_raster.load_input_fpdem_raster()
    fpdem_raster.compute_fpdem_raster()
    fpdem_raster.write_fpdem_raster()
    logging.info('Rasterization of FPDEM product was performed')
