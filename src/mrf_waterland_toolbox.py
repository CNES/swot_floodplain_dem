"""
SWOT Water Land Classification Toolbox.

This module provides tools for:
1. Data Loading: Fetching SWOT L2/L3 products from S3 zcollections.
2. Preprocessing: Data cleaning, normalization, and large-scale surface estimation.
3. Classification: Unsupervised Iterative Markov Random Field (MRF) pipeline.
"""

import numpy as np
import pandas as pd
# import swot_calval.io
from scipy.stats import norm, exponnorm
from scipy.ndimage import convolve, gaussian_filter
from tqdm import tqdm
from typing import Tuple, Dict, List, Optional, Union
from scipy.ndimage import median_filter
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label
import pyarrow.parquet as pq
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# ==========================================
# 1. DATA I/O & PREPROCESSING
# ==========================================

def load_data(path_l3: str, cycle: int, pass_number: int, side: str, 
              fs, bounds_lat: Tuple[float, float] = (-70, -68)):
    """
    Load and filter L2 and L3 data for a specific cycle, pass, and swath side.

    Parameters
    ----------
    path_l3 : str
        S3 paths to the zcollections.
    cycle, pass_number : int
        SWOT orbit parameters.
    side : str
        'Right' or 'Left' swath selection.
    fs : s3fs.S3FileSystem
        FileSystem object for S3 access.
    vars_l3: list, optional
        List of variables to load. Defaults are used if None.
    bounds_lat : tuple, optional
        Latitude bounds (min, max) for subsetting.

    Returns
    -------
    ds_l3: xarray.Dataset
        Subsets of L3 and L2 data.
    """
    # Default variables configuration
    vars_l3 = [
        'cycle_number', 'pass_number', 'time', 'longitude', 'latitude', 
        'latitude_nadir', 'cvl_cross_track_distance', 
        'sig0_karin_2', 'duacs_ssha_karin_2_calibrated', 
        'duacs_water_land_classification', 'ancillary_surface_classification_flag', 'cvl_distance_to_coast'
    ]

    # Open collections
    col_l3 = swot_calval.io.open_collection(path_l3, mode="r", filesystem=fs)

    # Query L3 Data
    zds_l3 = col_l3.query(
        cycle_numbers=[cycle], 
        pass_numbers=[pass_number], 
        selected_variables=vars_l3
    )
    
    if zds_l3 is None:
        return None
    ds_l3 = zds_l3.to_xarray()

    if ds_l3 is None or ds_l3.dims.get("num_lines", 0) == 0:
        return None

    # Latitude subsetting (Along-track)
    if bounds_lat:
        lines = (ds_l3.latitude_nadir.values <= bounds_lat[1]) & \
                (ds_l3.latitude_nadir.values >= bounds_lat[0])
        ds_l3 = ds_l3.isel(num_lines=lines).compute()

    # Après sous-ensemble latitude, vérifier à nouveau
    if ds_l3.dims.get("num_lines", 0) == 0:
        return None
        
    # Swath side selection (Cross-track)
    # Using 'cvl_cross_track_distance' to determine side geometry.
    # The range 10km-60km excludes the nadir gap and the far swath edge.
    dct = ds_l3.cvl_cross_track_distance[0].values
    if side == 'Right':
        indx = np.where((dct >= 10000) & (dct <= 60000))[0]
    elif side == 'Left':
        indx = np.where((dct <= -10000) & (dct >= -60000))[0]
    else:
        raise ValueError("Side must be 'Right' or 'Left'")

    ds_l3 = ds_l3.isel(num_pixels=indx)
    
    return ds_l3


def clean_data(sig0_db: np.ndarray, ssh: np.ndarray, surftype:np.array) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply basic physical thresholds to filter gross outliers.
    
    Parameters
    ----------
    sig0_db : np.ndarray
        Sigma0 backscatter in dB.
    ssh : np.ndarray
        Water Surface Height anomaly.

    Returns
    -------
    sig0_db, ssh_clean : np.ndarray
        Cleaned arrays with NaNs where values were out of bounds.
    """
    # Sig0 valid range: -40 to +40 dB
    #sig0_clean = np.where((sig0_db > -40) & (sig0_db < 40) & (surftype==0), sig0_db, np.nan)
    sig0_clean =  np.where((surftype==0), sig0_db, np.nan)

    # SSH valid range: +/- 2 meters
    ssh_clean = np.where((surftype==0), ssh, np.nan)

    return sig0_clean, ssh_clean


def clean_abnormal_lines(image, n_std=3.0, window_size=10, max_iters=10):
    """
    Detects and removes abnormal lines (artifacts) based on High-Frequency (HF) analysis.
    
    The algorithm computes the HF component of the image, calculates its standard deviation
    (ignoring NaNs), and flags lines where the mean HF exceeds N * std. Flagged lines are
    replaced by NaN. This process repeats iteratively until no new lines are detected.

    Parameters
    ----------
    image : 2D array
        Input image (e.g., SSH or Sigma0) to be cleaned.
    n_std : float, optional
        Threshold multiplier for standard deviation (default: 3.0).
    sigma_hf : float, optional
        Sigma for Gaussian filter to extract HF component (Image - LowPass).
    max_iters : int, optional
        Maximum number of iterations to prevent infinite loops.

    Returns
    -------
    cleaned_image : 2D array
        The input image with abnormal lines set to NaN.
    """
    #print(f"[Info] Cleaning abnormal lines (Threshold: {n_std} std)")
    cleaned_image = image.copy()
    rows, cols = cleaned_image.shape
    
    # 1. Compute High Frequency (HF) component
    profile_raw = np.nanmean(np.abs(cleaned_image), axis=0)
    
    # 2. Estimation de la basse fréquence (fond) via filtre médian
    background = median_filter(profile_raw, size=window_size)
    
    # 3. Soustraction pour isoler la haute fréquence
    high_freq = profile_raw - background
    bad_lines=[]
    for i in range(max_iters):
        # Calculate statistics on the HF component
        current_std = np.nanstd(high_freq)
        threshold = n_std * current_std
              
        # Identify bad lines
        # We look for lines where the average HF energy is abnormally high
        bad_lines_idx = np.where(np.abs(high_freq) > threshold)[0]
        
        if len(bad_lines_idx) == 0:
            #print(f"   Iter {i+1}: No more abnormal lines found. Convergence reached.")
            break
            
        #print(f"   Iter {i+1}: Detected {len(bad_lines_idx)} abnormal lines.")
        
        # Set bad lines to NaN in both the image and the HF reference
        high_freq[bad_lines_idx]=np.nan
        bad_lines.append(bad_lines_idx)
        
    return bad_lines


def estimate_large_scale_ocean(ssh:  np.ndarray, sig0: np.ndarray, xtrack: np.ndarray, window_size: np.int32) -> np.ndarray:
    """
    Estimates the large-scale water surface
    """
    # 1. Calcul des médianes (déjà dans ton code)
    medians_all = np.nanmedian(ssh, axis=1)

    #Filtre water
    mask_water = (
        (sig0 >= 12) 
        & (sig0 <=25)
        & (~np.isnan(ssh))
        & (ssh > -0.5)
        & (ssh < 0.5)
        & (np.abs(xtrack) > 35000)
        & (ssh <= medians_all[: , None])
    )

    rows, cols = ssh.shape
    medians_water = np.full(rows, np.nan)
    for i in range(rows):
        selected = ssh[i, mask_water[i, :]]
        if len(selected) > 0:
            medians_water[i] = np. nanmedian(selected)
    medians_water = median_filter(medians_water, size=window_size, mode='nearest')

    # 2. Interpolation forward
    medians_interp_fwd = np.copy(medians_water)
    N = len(medians_water)
    for i in range(1, N):
        if np.isnan(medians_interp_fwd[i]) and not np.isnan(medians_interp_fwd[i-1]):
            delta = medians_all[i] - medians_all[i-1]
            medians_interp_fwd[i] = medians_interp_fwd[i-1] + delta

    # 3. Interpolation backward (appliqué sur l'inversé)
    medians_interp_bwd = np.copy(medians_water[: :-1])
    medians_all_inv = medians_all[::-1]
    for i in range(1, N):
        if np.isnan(medians_interp_bwd[i]) and not np.isnan(medians_interp_bwd[i-1]):
            delta = medians_all_inv[i] - medians_all_inv[i-1]
            medians_interp_bwd[i] = medians_interp_bwd[i-1] + delta
    medians_interp_bwd = medians_interp_bwd[::-1]

    # 4. Fusion forward/backward
    fwd = medians_interp_fwd
    bwd = medians_interp_bwd
    final = np.copy(fwd)
    final[: ] = np.nan
    final[0] = fwd[0] if not np.isnan(fwd[0]) else bwd[0]
    final[1] = fwd[1] if not np.isnan(fwd[1]) else bwd[1]
    for i in range(2, len(final)):
        f = fwd[i]
        b = bwd[i]
        if np.isnan(f) and np.isnan(b):
            final[i] = np.nan
        elif not np. isnan(f) and np.isnan(b):
            final[i] = f
        elif np. isnan(f) and not np.isnan(b):
            final[i] = b
        else:
            prev_val = final[i-2]
            if np.isnan(prev_val):
                final[i] = f
            else:
                diff_fwd = abs(f - prev_val)
                diff_bwd = abs(b - prev_val)
                final[i] = f if diff_fwd <= diff_bwd else b

    medians_interp_final = final
    
    # 5. Remplissage et Filtre médian final
    filled = np. where(np.isnan(medians_interp_final),
                      medians_all - 0.2,
                      medians_interp_final)

    medians_interp_final_smooth = median_filter(filled, size=window_size, mode='nearest')

    return medians_interp_final_smooth


def estimate_large_scale_land(ssh: np.ndarray, sig0: np.ndarray, xtrack: np.ndarray, window_size: np.int32) -> np.ndarray:
    """
    Estimates the large-scale water surface
    """
    # 1. Calcul des médianes (déjà dans ton code)
    medians_all = np.nanmedian(ssh, axis=1)

    # Filtre land
    mask_land = (
        (sig0 <= 5)
        & (sig0 >=0)
        & (~np.isnan(ssh))
        & (ssh > -2)
        & (ssh < 2)
        & (ssh >= medians_all[: , None])
    )

    rows, cols = ssh.shape
    medians_land = np.full(rows, np. nan)
    for i in range(rows):
        selected = ssh[i, mask_land[i, :]]
        if len(selected) > 0:
            medians_land[i] = np. nanmedian(selected)
    medians_land = median_filter(medians_land, size=window_size, mode='nearest')

    # 2. Interpolation forward
    medians_interp_fwd = np.copy(medians_land)
    N = len(medians_land)
    for i in range(1, N):
        if np.isnan(medians_interp_fwd[i]) and not np.isnan(medians_interp_fwd[i-1]):
            delta = medians_all[i] - medians_all[i-1]
            medians_interp_fwd[i] = medians_interp_fwd[i-1] + delta

    # 3. Interpolation backward (appliqué sur l'inversé)
    medians_interp_bwd = np. copy(medians_land[::-1])
    medians_all_inv = medians_all[::-1]
    for i in range(1, N):
        if np.isnan(medians_interp_bwd[i]) and not np.isnan(medians_interp_bwd[i-1]):
            delta = medians_all_inv[i] - medians_all_inv[i-1]
            medians_interp_bwd[i] = medians_interp_bwd[i-1] + delta
    medians_interp_bwd = medians_interp_bwd[::-1]

    # 4. Fusion forward/backward
    fwd = medians_interp_fwd
    bwd = medians_interp_bwd
    final = np.copy(fwd)
    final[:] = np. nan
    final[0] = fwd[0] if not np.isnan(fwd[0]) else bwd[0]
    final[1] = fwd[1] if not np.isnan(fwd[1]) else bwd[1]
    for i in range(2, len(final)):
        f = fwd[i]
        b = bwd[i]
        if np.isnan(f) and np.isnan(b):
            final[i] = np.nan
        elif not np.isnan(f) and np.isnan(b):
            final[i] = f
        elif np.isnan(f) and not np.isnan(b):
            final[i] = b
        else:
            prev_val = final[i-2]
            if np. isnan(prev_val):
                final[i] = f
            else: 
                diff_fwd = abs(f - prev_val)
                diff_bwd = abs(b - prev_val)
                final[i] = f if diff_fwd <= diff_bwd else b

    medians_interp_final = final
    
    # 5. Remplissage et Filtre médian final
    filled = np.where(np. isnan(medians_interp_final),
                      medians_all + 0.2,
                      medians_interp_final)

    medians_interp_final_smooth = median_filter(filled, size=window_size, mode='nearest')

    return medians_interp_final_smooth


# ==========================================
# 2. STATISTICAL TOOLS & MRF CLASSIFICATION
# ==========================================

def get_pdf_proba(data: np.ndarray, params: Dict, law: str = 'gaussian') -> np.ndarray:
    """
    Calculate Probability Density Function (PDF) value for a given distribution.
    """
    if law == 'gaussian':
        return norm.pdf(data, loc=params.get('mu', 0), scale=params.get('std', 1))
    elif law == 'exponnorm':
        # Scipy exponnorm parameters: K (shape), loc, scale
        return exponnorm.pdf(data, params.get('expon_k', 1), 
                             loc=params.get('expon_loc', 0), 
                             scale=params.get('expon_scale', 1))
    return np.zeros_like(data)


def fit_distribution(data: np.ndarray, law: str = 'gaussian') -> Optional[Dict]:
    """
    Fit data to a specific distribution (Gaussian or ExponNorm).
    Returns dictionary of parameters or None if fitting fails.
    """
    # Basic outlier cleaning to ensure fit stability
    if len(data) < 100:
        return None
    
    p5, p95 = np.percentile(data, [2, 98])
    clean_data = data[(data >= p5) & (data <= p95)]
    
    if len(clean_data) == 0:
        return None

    if law == 'gaussian':
        mu, std = norm.fit(clean_data)
        return {'mu': mu, 'std': std}
        
    elif law == 'exponnorm':
        try:
            k, loc, scale = exponnorm.fit(clean_data)
            return {'expon_k': k, 'expon_loc': loc, 'expon_scale': scale, 
                    'mu_land': loc, 'std_land': scale}
        except Exception:
            # Fallback to Gaussian if ExponNorm fit fails
            mu, std = norm.fit(clean_data)
            return {'expon_k': 0.1, 'expon_loc': mu, 'expon_scale': std, 
                    'mu_land': mu, 'std_land': std}
    return None


def mrf_icm_fusion_final(sig0, ssh, p_sig0, p_ssh, beta=2.0, weight_ssh=2.0, 
                         iterations=5, land_law='gaussian'):
    """
    Perform Markov Random Field classification using Iterated Conditional Modes (ICM).
    Fuses information from both Sigma0 and SSH.
    
    Parameters
    ----------
    sig0, ssh : 2D arrays
        Input observables.
    p_sig0, p_ssh : dict
        Distribution parameters for Land/Water for each observable.
    beta : float
        Smoothness parameter (Potts model). Higher = smoother regions.
    weight_ssh : float
        Relative weight of SSH term in the energy function.
    iterations : int
        Number of ICM iterations.
    land_law : str
        'gaussian' or 'exponnorm' for SSH Land distribution.
    
    Returns
    -------
    proba_water : 2D array
        Probability map (0.0=Land, 1.0=Water). NaNs where invalid.
    """
    # --- 1. Initialization (Log-Likelihoods) ---
    
    # Calculate P(Sig0 | Class) - Always Gaussian
    p_s0_land_args = {'mu': p_sig0['mu_land'], 'std': p_sig0['std_land']}
    p_s0_wat_args = {'mu': p_sig0['mu_water'], 'std': p_sig0['std_water']}
    
    prob_s0_land = get_pdf_proba(sig0, p_s0_land_args, 'gaussian')
    prob_s0_wat = get_pdf_proba(sig0, p_s0_wat_args, 'gaussian')

    # Calculate P(SSH | Class)
    p_ssh_wat_args = {'mu': p_ssh['mu_water'], 'std': p_ssh['std_water']}
    prob_ssh_wat = get_pdf_proba(ssh, p_ssh_wat_args, 'gaussian')
    
    # Land depends on configuration
    if land_law == 'gaussian':
        p_ssh_land_args = {'mu': p_ssh['mu_land'], 'std': p_ssh['std_land']}
        prob_ssh_land = get_pdf_proba(ssh, p_ssh_land_args, 'gaussian')
    else:
        # Use provided ExponNorm params directly
        prob_ssh_land = get_pdf_proba(ssh, p_ssh, 'exponnorm')

    # Avoid log(0) errors
    epsilon = 1e-10
    prob_s0_land = np.clip(prob_s0_land, epsilon, None)
    prob_s0_wat = np.clip(prob_s0_wat, epsilon, None)
    prob_ssh_land = np.clip(prob_ssh_land, epsilon, None)
    prob_ssh_wat = np.clip(prob_ssh_wat, epsilon, None)

    # --- Data Energy Term (Negative Log Likelihood) ---
    # U_data(label) = - log(P(obs|label))
    # Combined Energy: U = U_s0 + (weight * U_ssh)
    u_data_land = -np.log(prob_s0_land) - weight_ssh * np.log(prob_ssh_land)
    u_data_wat = -np.log(prob_s0_wat) - weight_ssh * np.log(prob_ssh_wat)

    # Initial Labeling (MAP estimate without smoothness) for ICM Initialization
    labels = np.zeros_like(sig0, dtype=np.int8) # Default to 0 (Land)
    labels[u_data_wat < u_data_land] = 1 # Set to 1 (Water) where energy is lower
    
    # Handle NaNs (invalid data)
    mask_valid = (~np.isnan(sig0)) & (~np.isnan(ssh))
    labels[~mask_valid] = -1 # Background/Invalid marker

    # --- 2. ICM Iterations (Smoothness Prior) ---
    # 4-connectivity kernel for neighbors
    kernel = np.array([[0, 1, 0], 
                       [1, 0, 1], 
                       [0, 1, 0]]) 

#     kernel = np.array([[1, 1, 1], 
#                        [1, 0, 1], 
#                        [1, 1, 1]]) 
        
    for _ in range(iterations):
        # Count neighbors belonging to class 1 (Water)
        # Treat invalid pixels (-1) as 0 for neighbor counting to avoid boundary artifacts
        valid_labels = np.where(labels == -1, 0, labels)
        neighbors_1 = convolve(valid_labels.astype(float), kernel, mode='constant', cval=0.0)
        
        # Smoothness Energy (Potts Model)
        # Cost is 'beta' if center pixel differs from neighbor.
        # U_smooth(0) = beta * count(neighbors==1)
        # U_smooth(1) = beta * count(neighbors==0) = beta * (4 - neighbors_1)
        u_smooth_land = beta * neighbors_1
        u_smooth_wat = beta * (4.0 - neighbors_1)
        
        # Total Energy = Data Term + Smoothness Term
        total_e_land = u_data_land + u_smooth_land
        total_e_wat = u_data_wat + u_smooth_wat
        
        # Update Labels
        new_labels = np.zeros_like(labels)
        new_labels[total_e_wat < total_e_land] = 1
        new_labels[~mask_valid] = -1
        
        # Convergence Check
        diff = np.sum(new_labels != labels)
        labels = new_labels
        if diff == 0:
            break
            
    # --- 3. Compute Final Probabilities ---
    # Calculate the probability of being Water based on the final energy state
    # P(Water) = 1 / (1 + exp(E_wat - E_land))
    
    # Re-calculate neighbor energies for the final converged label state
    valid_labels = np.where(labels == -1, 0, labels)
    neighbors_1 = convolve(valid_labels.astype(float), kernel, mode='constant', cval=0.0)
    
    u_smooth_land = beta * neighbors_1
    u_smooth_wat = beta * (4.0 - neighbors_1)
    
    total_e_land = u_data_land + u_smooth_land
    total_e_wat = u_data_wat + u_smooth_wat
    
    # Calculate Probability
    # Clip energy difference to avoid overflow in exp()
    energy_diff = np.clip(total_e_wat - total_e_land, -100, 100)
    proba_water = 1.0 / (1.0 + np.exp(energy_diff))
    
    # Mask invalid pixels with NaN
    proba_water[~mask_valid] = np.nan
    
    return proba_water


def iterative_mrf_pipeline(sig0, ssh, p_sig0_init, p_ssh_init, 
                           land_law='gaussian',
                           max_iters=10, 
                           convergence_threshold=0.01, 
                           weight_ssh=1,
                           verbose=True):
    """
    Unsupervised Iterative Pipeline (EM-like algorithm).
    
    1. E-Step: MRF Classification using current parameters.
    2. M-Step: Update parameters by fitting distributions to classified subsets.
    3. Repeat until convergence.
    """
    current_p_sig0 = p_sig0_init.copy()
    current_p_ssh = p_ssh_init.copy()
    
    history = []
    prev_land_ratio = -1.0
    
    if verbose:
        print(f"=== START ITERATIVE PIPELINE (Land Law: {land_law}) ===")
    
    proba_map = None

    for k in range(max_iters):
        # 1. E-STEP : MRF Classification (Returns Probability Map)
        proba_map = mrf_icm_fusion_final(sig0, ssh, current_p_sig0, current_p_ssh, 
                                         beta=10.0, weight_ssh=weight_ssh, iterations=5, 
                                         land_law=land_law)
        
        # Calculate Metrics using a 0.5 probability threshold
        n_pixels = np.sum(~np.isnan(proba_map))
        if n_pixels == 0: break
        
        n_water = np.sum(proba_map >= 0.5)
        land_ratio = n_water / n_pixels
        
        if verbose:
            print(f"Iter {k+1}/{max_iters} | Land Ratio: {land_ratio:.2%}")
        
        history.append(land_ratio)
        
        # Convergence Check
        if abs(land_ratio - prev_land_ratio) < convergence_threshold:
            if verbose: print(">>> CONVERGENCE REACHED <<<")
            break
        prev_land_ratio = land_ratio
        
        # 2. M-STEP : Parameter Update
        # Hard thresholding for parameter estimation (0.5 cut-off)
        mask_land = (proba_map < 0.5)
        mask_water = (proba_map >= 0.5)
        
        # A. Update Sig0 Parameters (Always Gaussian)
        s0_land_data = sig0[mask_land & ~np.isnan(sig0)]
        s0_wat_data = sig0[mask_water & ~np.isnan(sig0)]
        
        fit_s0_i = fit_distribution(s0_land_data, 'gaussian')
        fit_s0_w = fit_distribution(s0_wat_data, 'gaussian')
        
        if fit_s0_i and fit_s0_w:
            current_p_sig0.update({
                'mu_land': fit_s0_i['mu'], 'std_land': fit_s0_i['std'],
                'mu_water': fit_s0_w['mu'], 'std_water': fit_s0_w['std']
            })
            
        # B. Update SSH Parameters
        ssh_land_data = ssh[mask_land & ~np.isnan(ssh)]
        ssh_wat_data = ssh[mask_water & ~np.isnan(ssh)]
        
        # Water (Gaussian)
        fit_ssh_w = fit_distribution(ssh_wat_data, 'gaussian')
        if fit_ssh_w:
            current_p_ssh['mu_water'] = fit_ssh_w['mu']
            current_p_ssh['std_water'] = fit_ssh_w['std']
        
        # Land (Configurable: Gaussian or ExponNorm)
        fit_ssh_i = fit_distribution(ssh_land_data, land_law)
        
        if fit_ssh_i:
            if land_law == 'gaussian':
                current_p_ssh.update({'mu_land': fit_ssh_i['mu'],
                                      'std_land': fit_ssh_i['std']})
            elif land_law == 'exponnorm':
                current_p_ssh.update(fit_ssh_i) # Updates K, loc, scale

    return proba_map, current_p_sig0, current_p_ssh, history


def run_tiled_mrf_pipeline(sig0, ssh, p_sig0_init, p_ssh_init, 
                           land_law, max_iters, convergence_threshold, weight_ssh,
                           pixel_res_m=250, 
                           tile_size_km=10, 
                           stride_km=None,
                           border_exclude_km=2):
    """
    Executes the Iterative MRF Pipeline on sliding tiles to handle local statistics variations.
    
    Returns
    -------
    global_proba : 2D array
        Reconstructed global probability map (Water Probability).
    """
    # Defaults: Set stride to half the tile size if not provided
    if stride_km is None:
        stride_km = tile_size_km / 2.0

    # 1. Convert Physical units to Pixels
    tile_px = int((tile_size_km * 1000) / pixel_res_m)
    stride_px = int((stride_km * 1000) / pixel_res_m)
    border_px = int((border_exclude_km * 1000) / pixel_res_m)
    
    rows, cols = sig0.shape
    
    print(f"=== TILED MRF PROCESSING ===")
    print(f"Img Shape: {sig0.shape}")
    print(f"Tile: {tile_px}px ({tile_size_km}km), Stride: {stride_px}px ({stride_km}km)")
    print(f"Border Exclusion: {border_px}px ({border_exclude_km}km)")

    # Accumulators for Weighted Averaging
    acc_proba = np.zeros_like(sig0, dtype=float)
    acc_weight = np.zeros_like(sig0, dtype=float)
    
    # Generate coordinates for sliding windows
    y_starts = range(0, rows, stride_px)
    x_starts = range(0, cols, stride_px)
    
    total_tiles = len(list(range(0, rows, stride_px))) * len(list(range(0, cols, stride_px)))
    
    with tqdm(total=total_tiles, desc="Processing Tiles") as pbar:
        for y in y_starts:
            for x in x_starts:
                # Define Tile Boundaries
                y_end = min(y + tile_px, rows)
                x_end = min(x + tile_px, cols)
                
                # Check if tile is large enough to be meaningful (e.g., > 1/4 size)
                if (y_end - y) < (tile_px // 4) or (x_end - x) < (tile_px // 4):
                    pbar.update(1)
                    continue

                # Extract Views
                sig0_tile = sig0[y:y_end, x:x_end]
                ssh_tile = ssh[y:y_end, x:x_end]
                
                # Check valid data ratio within the tile
                valid_ratio = np.sum(~np.isnan(sig0_tile)) / sig0_tile.size
                if valid_ratio < 0.1:
                    pbar.update(1)
                    continue

                # 3. Remove remaining ocean SSH ==> Useful over open-ocean
                #mask_water = (sig0_tile >= 12) & (~np.isnan(ssh_tile)) & (ssh_tile >-0.5) & (ssh_tile<0.15)
                #if np.sum(mask_water) > 5:
                #    ocean_ssh = np.nanmedian(ssh_tile[mask_water])
                #    ssh_tile = ssh_tile - ocean_ssh 
                
                # --- Run MRF on Tile ---
                # verbose=False suppresses inner loop prints
                prob_tile, _, _, history = iterative_mrf_pipeline(
                    sig0_tile, ssh_tile, p_sig0_init, p_ssh_init, 
                    land_law, max_iters, convergence_threshold, weight_ssh,
                    verbose=False)
                
                # Update progress bar description with tile stats
                final_land = history[-1] if history else 0.0
                pbar.set_postfix_str(f"Land Ratio: {final_land:.1%}")
                
                # --- Weight Mask Generation ---
                # Default weight is 1.0 everywhere
                weight_mask = np.ones_like(prob_tile, dtype=float)
                
                # Determine if this tile touches the global boundaries
                is_top_edge = (y == 0)
                is_bottom_edge = (y_end == rows)
                is_left_edge = (x == 0)
                is_right_edge = (x_end == cols)
                
                # Apply Border Exclusion Logic (set weight to 0)
                # Only mask the internal tile borders, preserve weights at global image edges
                if not is_top_edge:
                    weight_mask[:border_px, :] = 0
                if not is_bottom_edge:
                    weight_mask[-border_px:, :] = 0
                if not is_left_edge:
                    weight_mask[:, :border_px] = 0
                if not is_right_edge:
                    weight_mask[:, -border_px:] = 0
                
                # --- Accumulate ---
                # Handle NaNs in output (replace with 0 for addition, weight 0)
                tile_res = np.nan_to_num(prob_tile, nan=0.0)
                tile_w = np.where(np.isnan(prob_tile), 0.0, weight_mask)
                
                acc_proba[y:y_end, x:x_end] += (tile_res * tile_w)
                acc_weight[y:y_end, x:x_end] += tile_w
                
                pbar.update(1)

    # Normalize accumulated results
    with np.errstate(divide='ignore', invalid='ignore'):
        global_proba = acc_proba / acc_weight
    
    # Cleanup locations with no weights/coverage
    global_proba[acc_weight == 0] = np.nan
    
    return global_proba


def fix_isolated_pixels(proba_map, threshold=0.5, min_neighbors=3, connectivity=8):
    """
    Corrige les pixels isolés dans une carte de probabilité binaire.

    Parameters
    ----------
    proba_map : 2D array
        Probabilité d'être Water (0–1), NaN = invalide.
    threshold : float
        Seuil de décision (>= threshold => Water).
    min_neighbors : int
        Nb minimum de voisins valides pour corriger un pixel.
    connectivity : int
        4 ou 8 pour le voisinage.

    Returns
    -------
    proba_fixed : 2D array
        Probabilité après correction des isolés.
    labels_fixed : 2D array (int8)
        Labels après correction (0=Land, 1=Water, -1=NaN).
    """
    if connectivity == 4:
        kernel = np.array([[0,1,0],
                           [1,0,1],
                           [0,1,0]], dtype=float)
    else:  # 8-connectivité
        kernel = np.ones((3,3), dtype=float)
        kernel[1,1] = 0.0

    valid = ~np.isnan(proba_map)
    labels = np.full(proba_map.shape, -1, dtype=np.int8)
    labels[valid] = (proba_map[valid] >= threshold).astype(np.int8)

    n_water = convolve((labels == 1).astype(float), kernel, mode='constant', cval=0.0)
    n_valid = convolve((labels >= 0).astype(float), kernel, mode='constant', cval=0.0)

    majority_water = n_water > (n_valid / 2)
    majority_land  = (~majority_water) & (n_valid > 0)

    isolated_water = (labels == 1) & majority_land  & (n_valid >= min_neighbors)
    isolated_land  = (labels == 0) & majority_water & (n_valid >= min_neighbors)

    labels_fixed = labels.copy()
    labels_fixed[isolated_water] = 0
    labels_fixed[isolated_land]  = 1

    # Moyenne des probabilités voisines pour les pixels corrigés
    neighbor_sum   = convolve(np.nan_to_num(proba_map, nan=0.0), kernel, mode='constant', cval=0.0)
    neighbor_count = convolve(valid.astype(float), kernel, mode='constant', cval=0.0)
    neighbor_mean  = np.divide(
        neighbor_sum, neighbor_count,
        out=np.full_like(proba_map, np.nan, dtype=float),
        where=neighbor_count > 0
    )

    proba_fixed = proba_map.copy()
    to_update = isolated_water | isolated_land
    proba_fixed[to_update] = neighbor_mean[to_update]
    proba_fixed[~valid] = np.nan

    return proba_fixed, labels_fixed

def fill_nan_with_expanding_neighbor_mean(
    proba_map: np.ndarray,
    surftype: np.ndarray,
    max_radius: int | None = None,   # en pixels ; None = illimité
) -> np.ndarray:
    """
    Remplit les NaN situés sur mer/glace (surftype==0) par la valeur du pixel
    valide le plus proche. Les terres restent NaN.

    max_radius : rayon max (en pixels) ; au-delà, on laisse NaN.
    """
    out = proba_map.astype(float).copy()
    valid_domain = (surftype == 0)

    # Masque des points réellement valides (mer/glace + valeur non-NaN)
    mask_valid = valid_domain & np.isfinite(out)
    mask_nan   = valid_domain & ~mask_valid

    if not np.any(mask_nan):
        out[~valid_domain] = np.nan
        return out

    # Distance transform sur l'inverse du masque valide
    # idx donne, pour chaque pixel, les indices du pixel valide le plus proche
    dist, idx = distance_transform_edt(~mask_valid, return_indices=True)

    # Applique le plus proche voisin sur les NaN mer/glace
    nn0 = idx[0][mask_nan]
    nn1 = idx[1][mask_nan]
    out[mask_nan] = out[nn0, nn1]

    # Option : limite de rayon
    if max_radius is not None:
        out[mask_nan & (dist > max_radius)] = np.nan

    # Terre -> NaN
    out[~valid_domain] = np.nan
    return out


def classwise_smooth(field: np.ndarray, class_mask: np.ndarray, size: np.int32) -> np.ndarray:
    """
    Moyenne locale sizexsize d'un champ 'field' uniquement à l'intérieur de la classe définie par class_mask.
    Normalized convolution: mean = sum(field*mask) / sum(mask), NaN si aucun voisin de la classe.

    Parameters
    ----------
    field : 2D array (float)
        Champ à moyenner (ex: ssh_detrended), peut contenir des NaN.
    class_mask : 2D bool
        True pour les pixels de la classe (ex: water), False sinon.
    size:
        size of kernel

    Returns
    -------
    out : 2D array (float)
        Moyenne locale 3x3 à l'intérieur de la classe. NaN hors classe ou si pas de support.
    """
    kernel = np.ones((size, size), dtype=float)

    # pixels valides = dans la classe ET field non-NaN
    valid = class_mask & ~np.isnan(field)

    # somme des valeurs dans la fenêtre, en ignorant NaN via nan_to_num + masque
    field0 = np.nan_to_num(field, nan=0.0)
    sum_ = convolve(field0 * valid.astype(float), kernel, mode='constant', cval=0.0)
    cnt_ = convolve(valid.astype(float), kernel, mode='constant', cval=0.0)

    out = np.full_like(field, np.nan, dtype=float)
    inside = class_mask & (cnt_ > 0)
    out[inside] = sum_[inside] / cnt_[inside]

    return out

def compute_freeboard_and_land_concentration_sliding_tiles(
    ssh_raw: np.ndarray,
    proba_water: np.ndarray,      # probabilité d'être WATER (0–1), NaN invalide
    surftype: np.ndarray,         # 0 = océan/zone glace, autres = terre/invalide
    xtrack_2d_m: np.ndarray,      # (rows, cols)
    pixel_res_m: float = 250.0,
    tile_size_km: float = 10.0,
    stride_km: float = 1.0,
    min_points_per_class: int = 10,   # pour le freeboard (water & land)
    min_valid_for_conc: int = 10,     # pour la concentration de glace
):
    """
    Calcule freeboard et concentration de glace sur des tuiles glissantes
    en un seul parcours.

    Freeboard :
        freeboard = median(ssh | land) - median(ssh | water)
        (requiert min_points_per_class points dans chaque classe)

    Land concentration :
        land_prob = 1 - proba_water
        land_conc = moyenne(land_prob) sur les pixels valides (surftype==0)

    Returns
    -------
    freeboard_grid : (n_y, n_x) float
    land_conc_grid  : (n_y, n_x) float
    y_centers_lines : (n_y,) int
    x_centers_idx
    x_centers_m     : (n_x,) float
    """
    # Vérifs de forme
    assert ssh_raw.shape == proba_water.shape == surftype.shape == xtrack_2d_m.shape, "Shapes mismatch"
    rows, cols = ssh_raw.shape

    # Masquage terre et proba glace/eau
    proba_water_masked = np.where(surftype == 0, proba_water, np.nan)
    land_prob = 1.0 - proba_water_masked

    # Masques binaires pour le freeboard (seuil 0.5, comme avant)
    water_mask_global = (proba_water_masked > 0.5) & ~np.isnan(proba_water_masked)
    land_mask_global   = (proba_water_masked <=  0.5) & ~np.isnan(proba_water_masked)

    # Axe cross-track 1D
    xtrack_1d_m = np.nanmedian(xtrack_2d_m, axis=0)

    # Tuiles en pixels
    tile_px   = int(round((tile_size_km * 1000.0) / pixel_res_m))
    stride_px = int(round((stride_km   * 1000.0) / pixel_res_m))
    tile_px   = max(tile_px, 1)
    stride_px = max(stride_px, 1)

    y_starts = list(range(0, rows - tile_px + 1, stride_px))
    x_starts = list(range(0, cols - tile_px + 1, stride_px))

    n_y = len(y_starts)
    n_x = len(x_starts)

    freeboard = np.full((n_y, n_x), np.nan, dtype=float)
    land_conc  = np.full((n_y, n_x), np.nan, dtype=float)

    y_centers_lines = np.array([y0 + tile_px // 2 for y0 in y_starts], dtype=int)
    x_centers_idx   = np.array([x0 + tile_px // 2 for x0 in x_starts], dtype=int)
    x_centers_m     = xtrack_1d_m[x_centers_idx]

    for iy, y0 in enumerate(y_starts):
        y1 = y0 + tile_px
        for ix, x0 in enumerate(x_starts):
            x1 = x0 + tile_px

            # Sous-ensembles
            tile_ssh      = ssh_raw[y0:y1, x0:x1]
            tile_water_m  = water_mask_global[y0:y1, x0:x1] & ~np.isnan(tile_ssh)
            tile_land_m    = land_mask_global[y0:y1,   x0:x1] & ~np.isnan(tile_ssh)
            tile_land_prob = land_prob[y0:y1, x0:x1]

            # Freeboard (nécessite assez de points dans chaque classe)
            if tile_water_m.sum() >= min_points_per_class and tile_land_m.sum() >= min_points_per_class:
                med_w = np.nanmedian(tile_ssh[tile_water_m])
                med_i = np.nanmedian(tile_ssh[tile_land_m])
                freeboard[iy, ix] = med_i - med_w

            # Land concentration (moyenne des probas après seuillage à 0.1)
            valid_conc_mask = ~np.isnan(tile_land_prob)
            if valid_conc_mask.sum() >= min_valid_for_conc:
                # On extrait les probas valides
                probas = tile_land_prob[valid_conc_mask]

                # On applique votre nouveau seuil :
                # Si > 0.1 alors 1 (Glace), sinon 0 (Eau)
                land_binarized = np.where(probas > 0.1, 1.0, 0.0)

                # On fait la moyenne de ces valeurs 0 et 1
                land_conc[iy, ix] = np.nanmean(land_binarized)

    return freeboard, land_conc, y_centers_lines, x_centers_idx, x_centers_m

# ##
# PLOT FUNCTIONS
# ##

def bin_variable_ultrafast(
    files,
    varname,
    LAT_MIN,
    LAT_MAX,
    GRID_M,
    to_ps,
    vmin=None,
    vmax=None,
    nbins=64,
    xtrack_min=None,
    xtrack_max=None,
    AGG="median",        # "mean" ou "median"
    MIN_COUNT_MEDIAN=10  # seuil médiane
):
    """
    Binning spatial ultra-rapide d'une variable quelconque
    avec moyenne, std exacts et médiane robuste (fallback mean).
    """

    # ==========================================================
    # PASS 0 — fichiers valides + lat/lon
    # ==========================================================
    lat_col = lon_col = None
    valid_files = []

    for f in files:
        try:
            schema = pq.ParquetFile(f).schema.names
        except Exception:
            continue

        if "lat" in schema and "lon" in schema:
            lat_col, lon_col = "lat", "lon"
        elif "latitude" in schema and "longitude" in schema:
            lat_col, lon_col = "latitude", "longitude"
        else:
            continue

        if varname in schema:
            valid_files.append(f)

    if not valid_files:
        raise RuntimeError(f"Aucun fichier avec {varname}")

    # ==========================================================
    # PASS 1 — bornes spatiales
    # ==========================================================
    xmin = ymin = np.inf
    xmax = ymax = -np.inf

    for f in tqdm(valid_files, desc=f"[{varname}] Scan bornes"):
        df = pd.read_parquet(
            f,
            columns=[lat_col, lon_col],
            engine="pyarrow"
        )

        lat = df[lat_col].values
        lon = df[lon_col].values

        m = (lat >= LAT_MIN) & (lat <= LAT_MAX)
        if not m.any():
            continue

        x, y = to_ps.transform(lon[m], lat[m])
        xmin = min(xmin, x.min())
        xmax = max(xmax, x.max())
        ymin = min(ymin, y.min())
        ymax = max(ymax, y.max())

    xmin = np.floor(xmin / GRID_M) * GRID_M
    ymin = np.floor(ymin / GRID_M) * GRID_M
    xmax = np.ceil(xmax / GRID_M) * GRID_M
    ymax = np.ceil(ymax / GRID_M) * GRID_M

    nx = int((xmax - xmin) / GRID_M)
    ny = int((ymax - ymin) / GRID_M)
    ncell = nx * ny

    # ==========================================================
    # Accumulateurs
    # ==========================================================
    count = np.zeros(ncell, dtype=np.int32)
    sum_v = np.zeros(ncell, dtype=np.float64)
    sum2_v = np.zeros(ncell, dtype=np.float64)

    if AGG == "median":
        if vmin is None or vmax is None:
            raise ValueError("vmin/vmax requis pour la médiane")
        edges = np.linspace(vmin, vmax, nbins + 1)
        hist = np.zeros((ncell, nbins), dtype=np.int32)

    # ==========================================================
    # PASS 2 — binning
    # ==========================================================
    for f in tqdm(valid_files, desc=f"[{varname}] Binning"):
        df = pd.read_parquet(
            f,
            columns=[lat_col, lon_col, varname, "xtrack_m"],
            engine="pyarrow"
        )

        lat = df[lat_col].values
        lon = df[lon_col].values
        v = df[varname].values

        m = (lat >= LAT_MIN) & (lat <= LAT_MAX)

        if xtrack_min is not None:
            m &= np.abs(df["xtrack_m"].values) >= xtrack_min
        if xtrack_max is not None:
            m &= np.abs(df["xtrack_m"].values) <= xtrack_max

        if not m.any():
            continue

        x, y = to_ps.transform(lon[m], lat[m])

        ix = ((x - xmin) / GRID_M).astype(np.int32)
        iy = ((y - ymin) / GRID_M).astype(np.int32)
        idx = iy * nx + ix

        vsel = v[m]

        np.add.at(count, idx, 1)
        np.add.at(sum_v, idx, vsel)
        np.add.at(sum2_v, idx, vsel * vsel)

        if AGG == "median":
            ib = np.searchsorted(edges, vsel, side="right") - 1
            ok = (ib >= 0) & (ib < nbins)
            np.add.at(hist, (idx[ok], ib[ok]), 1)

    # ==========================================================
    # Stats finales
    # ==========================================================
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = sum_v / count
        std = np.sqrt(sum2_v / count - mean**2)

    median = None
    if AGG == "median":
        cdf = np.cumsum(hist, axis=1)
        half = (count + 1) // 2
        ib = (cdf >= half[:, None]).argmax(axis=1)
        median_hist = 0.5 * (edges[ib] + edges[ib + 1])

        # === LOGIQUE FINALE ROBUSTE ===
        median = np.full(ncell, np.nan)

        # médiane fiable
        good = count >= MIN_COUNT_MEDIAN
        median[good] = median_hist[good]

        # fallback mean si peu de points
        fallback = (count > 0) & (count < MIN_COUNT_MEDIAN)
        median[fallback] = mean[fallback]

    # ==========================================================
    # Reshape
    # ==========================================================
    mean_grid = mean.reshape(ny, nx)
    std_grid = std.reshape(ny, nx)
    count_grid = count.reshape(ny, nx)

    median_grid = None
    if AGG == "median":
        median_grid = median.reshape(ny, nx)

    x_centers = xmin + GRID_M * (0.5 + np.arange(nx))
    y_centers = ymin + GRID_M * (0.5 + np.arange(ny))

    return mean_grid, median_grid, std_grid, count_grid, x_centers, y_centers



def fill_grid_cubic_and_mask_lat(
    grid,
    x_centers,
    y_centers,
    to_ps,
    LAT_MIN,
    LAT_MAX,
    method="cubic"
):
    X, Y = np.meshgrid(x_centers, y_centers)
    valid = np.isfinite(grid)

    if np.sum(valid) < 10:
        return grid.copy()

    points = np.column_stack((X[valid], Y[valid]))
    values = grid[valid]

    grid_interp = griddata(points, values, (X, Y), method=method)

    filled = grid.copy()
    filled[~valid] = grid_interp[~valid]

    # Masque polaire correct
    lon_grid, lat_grid = to_ps.transform(X, Y, direction="INVERSE")
    mask = (lat_grid >= LAT_MIN) & (lat_grid <= LAT_MAX)
    filled[~mask] = np.nan

    return filled


def plot_map(param, Xc, Yc, sigma, min_map, max_map):

    if sigma !=0:
        param = gaussian_filter(param, sigma=sigma)

    VMIN_IC, VMAX_IC = min_map, max_map
    crs_data = ccrs.epsg(3413)
    proj_plot = ccrs.NorthPolarStereo(central_longitude=0)

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": proj_plot})

    # Fond terres en dessous
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "land", "110m",
                                     facecolor="lightgray", edgecolor="none"),
        zorder=4,
    )

    # Grille (sous les côtes)
    im = ax.pcolormesh(
        Xc, Yc, param,
        transform=crs_data,
        cmap="Spectral_r", vmin=VMIN_IC, vmax=VMAX_IC,
        shading="nearest",
        zorder=2,
    )

    # Traits de côte au-dessus
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="k", zorder=4)

    # Emprise Arctique
    ax.set_extent([-180, 180, 58, 90], crs=ccrs.PlateCarree())

    # Graticule
    gl = ax.gridlines(draw_labels=False, x_inline=False, y_inline=False,
                      linewidth=0.6, color="gray", alpha=0.6, linestyle="--", zorder=3)

    ax.set_aspect("equal")
    cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cb.set_label("Land concentration [0–1]")

    ax.set_title("Land concentration")
    plt.tight_layout()
    ZOOM_SCALE = 1.2  # facteur de zoom par cran ( >1 )

    def on_scroll(event):
        if event.inaxes != ax:
            return
        cur_xmin, cur_xmax = ax.get_xlim()
        cur_ymin, cur_ymax = ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return
        scale = 1/ZOOM_SCALE if event.button == 'up' else ZOOM_SCALE
        new_w = (cur_xmax - cur_xmin) * scale
        new_h = (cur_ymax - cur_ymin) * scale
        ax.set_xlim([xdata - new_w/2, xdata + new_w/2])
        ax.set_ylim([ydata - new_h/2, ydata + new_h/2])
        ax.figure.canvas.draw_idle()

    cid = fig.canvas.mpl_connect('scroll_event', on_scroll)
    plt.show()

def plot_map_and_save(param, Xc, Yc, sigma, min_map, max_map, title, output_file):
    """
    Génère une carte polaire et la sauvegarde dans un fichier.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from scipy.ndimage import gaussian_filter

    if sigma != 0:
        param = gaussian_filter(param, sigma=sigma)

    crs_data = ccrs.epsg(3413)
    proj_plot = ccrs.NorthPolarStereo(central_longitude=0)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": proj_plot})

    # Fond terres
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "land", "110m",
                                     facecolor="lightgray", edgecolor="none"),
        zorder=4,
    )

    # Grille de données
    im = ax.pcolormesh(
        Xc, Yc, param,
        transform=crs_data,
        cmap="Spectral_r", vmin=min_map, vmax=max_map,
        shading="nearest",
        zorder=2,
    )

    # Traits de côte
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="k", zorder=4)

    # Emprise Arctique
    ax.set_extent([-180, 180, 58, 90], crs=ccrs.PlateCarree())

    # Graticule
    ax.gridlines(draw_labels=False, x_inline=False, y_inline=False,
                 linewidth=0.6, color="gray", alpha=0.6, linestyle="--", zorder=3)

    ax.set_aspect("equal")
    
    # Colorbar
    cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cb.set_label("Land concentration [0–1]")

    # Titre
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Sauvegarde
    fig.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"[INFO] Carte sauvegardée: {output_file}")
