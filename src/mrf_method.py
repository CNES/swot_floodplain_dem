# -*- coding: utf8 -*-
"""
Apply the MRF (Markov Random Field) method to extract bathymetry
"""

import os
from pyproj import CRS
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from scipy.spatial import cKDTree
from skimage.morphology import erosion, dilation
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mrf_waterland_toolbox as toolbox


def meters_to_deg(step_m, latitude_deg):
    """
    Convert meters to degrees for lat/lon coordinates

    :param step_m: Step inm meters
    :param latitude_deg: Latitude position
    :return: lat and lon in degrees
    """
    lat_rad = latitude_deg * np.pi / 180.

    meters_per_deg_lat = 111_320
    meters_per_deg_lon = 111_320 * np.cos(lat_rad)

    dlat = step_m / meters_per_deg_lat
    dlon = step_m / meters_per_deg_lon

    return dlat, dlon


def get_grid_lat_lon_ref(refdem_grid, path_ref_dem='', pixc='', step=50., margin=[0., 0.]):

    if refdem_grid:
        ref_dem = xr.open_dataset(path_ref_dem)
        latitude_ref_dem = ref_dem.latitude.values
        longitude_ref_dem = ref_dem.longitude.values

    else:
        ds = xr.open_dataset(pixc)
        geospatial_lon_min = ds.geospatial_lon_min
        geospatial_lon_max = ds.geospatial_lon_max
        geospatial_lat_min = ds.geospatial_lat_min
        geospatial_lat_max = ds.geospatial_lat_max

        dlat, dlon = meters_to_deg(step, (geospatial_lat_min + geospatial_lat_max) / 2.)

        longitude_ref_dem = np.arange(geospatial_lon_min - margin[0],
                                      geospatial_lon_max + margin[0],
                                      dlon)
        latitude_ref_dem = np.arange(geospatial_lat_min - margin[1],
                                     geospatial_lat_max + margin[1],
                                     dlat)
        mean_dem = np.zeros([len(latitude_ref_dem), len(longitude_ref_dem)])

        ref_dem = xr.Dataset(
            {
                "elevation": (("latitude", "longitude"), mean_dem)
            },
            coords={
                "latitude": (("latitude"), latitude_ref_dem),
                "longitude": (("longitude"), longitude_ref_dem),
            }, )

    return ref_dem, latitude_ref_dem, longitude_ref_dem


def extract_params_from_pixc(file, root, ref_dem):
    date = file.split("_")[7][0:8]

    grid_h = xr.DataArray(
        np.zeros_like(ref_dem.elevation),
        dims=ref_dem.dims,
        coords=ref_dem.coords
    )

    grid_inc = xr.DataArray(
        np.zeros_like(ref_dem.elevation),
        dims=ref_dem.dims,
        coords=ref_dem.coords
    )

    grid_sig0 = xr.DataArray(
        np.zeros_like(ref_dem.elevation),
        dims=ref_dem.dims,
        coords=ref_dem.coords
    )

    grid_coh = xr.DataArray(
        np.zeros_like(ref_dem.elevation),
        dims=ref_dem.dims,
        coords=ref_dem.coords
    )

    grid_coh_th = xr.DataArray(
        np.zeros_like(ref_dem.elevation),
        dims=ref_dem.dims,
        coords=ref_dem.coords
    )

    grid_count = xr.DataArray(
        np.zeros_like(ref_dem.elevation),
        dims=ref_dem.dims,
        coords=ref_dem.coords
    )

    points = xr.open_dataset(os.path.join(root, file), group="pixel_cloud")
    noise = xr.open_dataset(os.path.join(root, file), group="noise")

    points_classif = points.classification.values

    ind_classif = np.where(points_classif > 2)

    points_lon = points.longitude.values[ind_classif]
    points_lat = points.latitude.values[ind_classif]
    points_inc = points.inc.values[ind_classif]
    points_height = points.height.values[ind_classif]
    points_sig0 = points.sig0.values[ind_classif]

    # points_interferogram = (points.interferogram.values[ind_classif][0, :]
    #                         + 1j * points.interferogram.values[ind_classif][1, :])
    points_coh = (np.abs((points.interferogram.values[ind_classif][:, 0]
                          + 1j * points.interferogram.values[ind_classif][:, 1]) ** 2)
                  / np.sqrt(
                (points.power_minus_y.values[ind_classif][:] ** 2 * points.power_plus_y.values[ind_classif][:] ** 2)))
    points_snr = ((points.power_plus_y.values[ind_classif] - np.nanmean(noise.noise_plus_y.values))
                  / np.nanmean(noise.noise_plus_y.values))
    points_coh_th_noise = points_snr / (points_snr + 1)

    lon_2d, lat_2d = np.meshgrid(ref_dem.longitude, ref_dem.latitude)
    coords_2d = np.column_stack((lon_2d.ravel(), lat_2d.ravel()))

    tree = cKDTree(coords_2d)

    distances, indices = tree.query(np.column_stack((points_lon, points_lat)))

    rows = indices // (ref_dem.sizes['longitude'])
    cols = indices % (ref_dem.sizes['longitude'])

    mask = (rows < ref_dem.sizes['latitude']) & (cols < ref_dem.sizes['longitude']) & (10 * np.log10(points_sig0) > -20)

    grid_inc.data[rows[mask], cols[mask]] += points_inc[mask]
    grid_h.data[rows[mask], cols[mask]] += points_height[mask]
    grid_sig0.data[rows[mask], cols[mask]] += points_sig0[mask]
    grid_coh.data[rows[mask], cols[mask]] += points_coh[mask]
    grid_coh_th.data[rows[mask], cols[mask]] += points_coh_th_noise[mask]
    grid_count.data[rows[mask], cols[mask]] += 1
    grid_count.data[np.where(grid_count.data == 0)] = 1.

    grid_h.data = grid_h.data / grid_count.data
    grid_inc.data = grid_inc.data / grid_count.data
    grid_sig0.data = grid_sig0.data / grid_count.data
    grid_coh.data = grid_coh.data / grid_count.data
    grid_coh_th.data = grid_coh_th.data / grid_count.data

    grid_h_nan = grid_h.copy()
    grid_h_nan = grid_h_nan.sortby('latitude')
    grid_h_nan.data = np.where(grid_h == 0, np.nan, grid_h)

    grid_sig_nan = grid_sig0.copy()
    grid_sig_nan = grid_sig_nan.sortby('latitude')
    grid_sig_nan.data = np.where(grid_sig0 == 0, np.nan, grid_sig0)

    grid_coh_nan = grid_coh.copy()
    grid_coh_nan = grid_coh_nan.sortby('latitude')
    grid_coh_nan.data = np.where(grid_coh == 0, np.nan, grid_coh)

    grid_coh_th_nan = grid_coh_th.copy()
    grid_coh_th_nan = grid_coh_th_nan.sortby('latitude')
    grid_coh_th_nan.data = np.where(grid_coh_th == 0, np.nan, grid_coh_th)

    grid_h_interp = grid_h_nan.interpolate_na(dim="latitude", method="nearest")
    grid_sig0_interp = grid_sig_nan.interpolate_na(dim="latitude", method="nearest")
    grid_coh_interp = grid_coh_nan.interpolate_na(dim="latitude", method="nearest")
    grid_coh_th_interp = grid_coh_th_nan.interpolate_na(dim="latitude", method="nearest")

    return (grid_h, grid_h_interp, grid_inc, grid_sig0, grid_sig0_interp,
            grid_coh_interp, mask, date, grid_coh_th_interp)


def plot_var_and_zoom(var, var_name, cmap=None, vmin=None, vmax=None, zoom=None):
    """
    Plot a variable and, if wanted, a zoom of it right next to it
    :param var: variable to plot
    :param var_name: name of the variable for the title
    :param cmap: colormap
    :param vmin: variable minimum
    :param vmax: variable maximum
    :param zoom: range of the wanted zoom
    """
    if zoom is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
        cax0 = make_axes_locatable(axes[0]).append_axes("right", size="5%", pad=0.05)
        axes[0].set_title(f"{var_name}")
        im0 = axes[0].imshow(var, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im0, ax=axes[0], cax=cax0)

        cax1 = make_axes_locatable(axes[1]).append_axes("right", size="5%", pad=0.05)
        axes[1].set_title(f"{var_name} - zoom")
        im1 = axes[1].imshow(var[zoom], cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im1, ax=axes[1], cax=cax1)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(4, 5))  # 1 row, 1 column
        cax0 = make_axes_locatable(axes).append_axes("right", size="5%", pad=0.05)
        axes.set_title(f"{var_name}")
        im0 = axes.imshow(var, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im0, ax=axes, cax=cax0)


# def plot_variables(date, vars, zoom=zoom):
#
#     fig, axes = plt.subplots(1, len(vars), figsize=(30, 10))  # 1 row, 2 columns
#
#     axes[0].set_title("swot dem " + date)
#     cax0 = make_axes_locatable(axes[0]).append_axes("right", size="5%", pad=0.05)
#     dem_plot = axes[0].imshow((vars[0].data)[zoom], vmin=25, vmax=32)
#     fig.colorbar(dem_plot, ax=axes[0], cax=cax0)
#
#     axes[1].set_title("swot sig0 (dB) " + date)
#     cax1 = make_axes_locatable(axes[1]).append_axes("right", size="5%", pad=0.05)
#     sig0_plot = axes[1].imshow(vars[1][zoom], vmin=0, vmax=15, cmap="Greys_r")
#     fig.colorbar(sig0_plot, ax=axes[1], cax=cax1)
#
#     if len(vars) >= 2:
#         axes[2].set_title("proba map without height " + date)
#         cax2 = make_axes_locatable(axes[2]).append_axes("right", size="5%", pad=0.05)
#         labels_plot = axes[2].imshow(vars[2][0][zoom])
#         fig.colorbar(labels_plot, ax=axes[2], cax=cax2)
#
#     if len(vars) >= 3:
#         axes[3].set_title("proba map with height " + date)
#         cax3 = make_axes_locatable(axes[3]).append_axes("right", size="5%", pad=0.05)
#         labels_plot_with_height = axes[3].imshow(vars[2][2][zoom])
#         fig.colorbar(labels_plot_with_height, ax=axes[3], cax=cax3)


def apply_mrf_method(fpdem,
                     p_sig0_init, p_ssh_init, land_law, max_iters, convergence, weight_ssh,
                     pixel_res_m, tile_size_km, stride_km, border_exclude_km):

    height_cycle_0 = fpdem[1].data[:, :]
    sig0_cycle_0 = 10 * np.log10(fpdem[4].data[:, :])

    sig0_cycle_0 = np.where(np.isinf(sig0_cycle_0), np.nan, sig0_cycle_0)
    height_cycle_0 = np.where(np.isinf(sig0_cycle_0), np.nan, height_cycle_0)

    tab_tmp = np.where(fpdem[0] == 0., 0, 1)
    kernel_dilate = np.ones([3, 3])
    kernel_erosion = np.ones([3, 3])
    tab_tmp = erosion(dilation(tab_tmp, kernel_dilate), kernel_erosion)

    height_cycle_0 = np.where(tab_tmp == 1, height_cycle_0, np.nan)
    sig0_cycle_0 = np.where(tab_tmp == 1, sig0_cycle_0, np.nan)

    proba_map = toolbox.run_tiled_mrf_pipeline(
        sig0_cycle_0, height_cycle_0,
        p_sig0_init, p_ssh_init,
        land_law=land_law,
        max_iters=max_iters,
        convergence_threshold=convergence,
        weight_ssh=0.,  # Nul weight on ssh
        pixel_res_m=pixel_res_m,
        tile_size_km=tile_size_km,
        stride_km=stride_km,
        border_exclude_km=border_exclude_km
    )

    proba_smooth, labels_smooth = toolbox.fix_isolated_pixels(proba_map, threshold=0.5, min_neighbors=3, connectivity=8)

    proba_smooth_0 = np.where(fpdem[0].data[:, :] == 0, np.nan, proba_smooth)
    labels_smooth_0 = np.where(fpdem[0].data[:, :] == 0, np.nan, labels_smooth)

    height_cycle_0 = height_cycle_0 - np.nanmean(fpdem[0].data[:, :][np.where(labels_smooth_0 == 1)])

    p_sig0_init = {
        'mu_ice': np.nanmean(sig0_cycle_0[np.where(labels_smooth_0 == 0)]),
        'std_ice': np.nanstd(sig0_cycle_0[np.where(labels_smooth_0 == 0)]),
        'mu_water': np.nanmean(sig0_cycle_0[np.where(labels_smooth_0 == 1)]),
        'std_water': np.nanstd(sig0_cycle_0[np.where(labels_smooth_0 == 1)])
    }

    p_ssh_init = {
        'mu_ice': np.nanmean(height_cycle_0[np.where(labels_smooth_0 == 0)]),
        'std_ice': np.nanstd(height_cycle_0[np.where(labels_smooth_0 == 0)]),
        'mu_water': np.nanmean(height_cycle_0[np.where(labels_smooth_0 == 1)]),
        'std_water': np.nanstd(height_cycle_0[np.where(labels_smooth_0 == 1)]),
        'expon_k': 1.0, 'expon_loc': 0.1, 'expon_scale': 0.05
    }

    print(p_sig0_init)
    print(p_ssh_init)

    proba_map = toolbox.run_tiled_mrf_pipeline(
        sig0_cycle_0, height_cycle_0,
        p_sig0_init, p_ssh_init,
        ice_law=ice_law,
        max_iters=max_iters,
        convergence_threshold=convergence,
        weight_ssh=weight_ssh,
        pixel_res_m=pixel_res_m,
        tile_size_km=tile_size_km,
        stride_km=stride_km,
        border_exclude_km=border_exclude_km
    )

    proba_smooth, labels_smooth = toolbox.fix_isolated_pixels(proba_map, threshold=0.5, min_neighbors=3, connectivity=8)

    proba_smooth_h = np.where(fpdem[0].data[:, :] == 0, np.nan, proba_smooth)
    labels_smooth_h = np.where(fpdem[0].data[:, :] == 0, np.nan, labels_smooth)

    return height_cycle_0, sig0_cycle_0, proba_smooth_0, labels_smooth_0, proba_smooth_h, labels_smooth_h

def get_labels(fpdem, idate, proba_map_list, threshold_with_height, threshold_without_height, zoom):

    proba_smooth = proba_map_list[idate][0]
    labels_smooth = proba_map_list[idate][1]
    proba_smooth_with_height = proba_map_list[idate][2]
    # labels_smooth_with_height = proba_map_list[idate][3]

    # Get combined labels by filtering using proba maps
    labels_combined = np.zeros_like(labels_smooth)
    labels_combined = np.where((proba_smooth_with_height > threshold_with_height) &
                               (proba_smooth > threshold_without_height), 1, labels_combined)
    labels_combined = np.where(fpdem[idate][0].data[:, :] == 0, np.nan, labels_combined)

    # Plot, zoomed on a small part of the region
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))  # 1 row, 2 columns

    axes[0].set_title("swot dem " + fpdem[idate][7])
    axes[1].set_title("swot sig0 (dB) " + fpdem[idate][7])
    axes[2].set_title("label map combined " + fpdem[idate][7])

    cax0 = make_axes_locatable(axes[0]).append_axes("right", size="5%", pad=0.05)
    cax1 = make_axes_locatable(axes[1]).append_axes("right", size="5%", pad=0.05)
    cax2 = make_axes_locatable(axes[2]).append_axes("right", size="5%", pad=0.05)

    dem_plot = axes[0].imshow(fpdem[idate][0].data[zoom], vmin=25, vmax=32)
    sig0_plot = axes[1].imshow(10 * np.log10(fpdem[idate][3])[zoom], vmin=0, vmax=15, cmap="Greys_r")
    labels_plot_combined = axes[2].imshow(labels_combined[zoom])

    fig.colorbar(dem_plot, ax=axes[0], cax=cax0)
    fig.colorbar(sig0_plot, ax=axes[1], cax=cax1)
    fig.colorbar(labels_plot_combined, ax=axes[2], cax=cax2)

    return labels_combined

def get_mean_dem(fpdem_filter, labels_combined_list_filter, bad_date):

    mean_dem = np.zeros_like(fpdem_filter[0][0], dtype=float)
    mean_sig0 = np.zeros_like(fpdem_filter[0][0], dtype=float)
    mean_coh = np.zeros_like(fpdem_filter[0][0], dtype=float)
    mean_coh_th = np.zeros_like(fpdem_filter[0][0], dtype=float)
    count = np.zeros_like(fpdem_filter[0][0], dtype=float)

    label_land = 0.

    for i in range(len(fpdem_filter)):
        if int(fpdem_filter[i][7]) not in bad_date:
            # ind = np.where(labels_combined_list_filter[i] == 0)

            mean_dem += np.where(labels_combined_list_filter[i] == label_land, fpdem_filter[i][0], 0.)
            mean_sig0 += np.where(labels_combined_list_filter[i] == label_land, fpdem_filter[i][3], 0.)
            mean_coh += np.where(labels_combined_list_filter[i] == label_land, fpdem_filter[i][5], 0.)
            mean_coh_th += np.where(labels_combined_list_filter[i] == label_land, fpdem_filter[i][8], 0.)

            # inc = np.where(labels_combined_list_filter[i] == label_land, fpdem_filter[i][2], 0.)
            # coh = np.where(labels_combined_list_filter[i] == label_land, fpdem_filter[i][5], 0.)
            # var_h = wavelength * np.sqrt(1/coh**2-1) / 2 / np.pi / np.sin(inc*np.pi/180.)
            # weight = 1/var_h**2

            # Sum of labels
            count += np.where(labels_combined_list_filter[i] == label_land, 1., 0.)
            print('Maximum of count: ', np.max(count))

            # Plot labels
            plt.figure()
            plt.title(int(fpdem_filter[i][7]))
            plt.imshow(np.where(labels_combined_list_filter[i] == label_land, 1., 0.))
            plt.colorbar()
            plt.show()

    mean_dem = mean_dem / count
    mean_sig0 = mean_sig0 / count
    mean_coh = mean_coh / count
    mean_coh_th = mean_coh_th / count

    mean_dem = np.where(count == 0., np.nan, mean_dem)
    mean_sig0 = np.where(count == 0., np.nan, mean_sig0)
    mean_coh = np.where(count == 0., np.nan, mean_coh)
    mean_coh_th = np.where(count == 0., np.nan, mean_coh_th)

    return mean_dem, mean_sig0, mean_coh, mean_coh_th, count


def write_mrf_fpdem_file(output_file, x, y, out_image, epsg):

    ds = xr.Dataset(
        {
            "mean_dem": (("latitude", "longitude"), out_image),
        },
        coords={
            "latitude": ("latitude", y),
            "longitude": ("longitude", x),
        },
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

    # Add necessary attributes
    ds["mean_dem"].attrs["units"] = "m"
    ds["mean_dem"].attrs["long_name"] = "Mean Digital Elevation Model"
    ds["mean_dem"].attrs["standard_name"] = "height_above_mean_sea_level"
    ds["mean_dem"].attrs["grid_mapping"] = 'crs'

    ds["latitude"].attrs["units"] = "degrees_north"
    ds["latitude"].attrs["long_name"] = "latitude coordinate"
    ds["latitude"].attrs["standard_name"] = "latitude"
    ds["latitude"].attrs["grid_mapping"] = 'crs'

    ds["longitude"].attrs["units"] = "degrees_east"
    ds["longitude"].attrs["long_name"] = "longitude coordinate"
    ds["longitude"].attrs["standard_name"] = "longitude"
    ds["longitude"].attrs["grid_mapping"] = 'crs'

    # Add global attributes
    ds.attrs["Conventions"] = "CF-1.6"

    # Write the netcdf file
    ds.to_netcdf(output_file)
