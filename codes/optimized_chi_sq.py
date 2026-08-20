"""
Optimized chi-squared computation module for binary star analysis.
This module provides optimized functions for computing chi-squared values
for large numbers of stars across multiple cores.
"""
import os
import numpy as np
import bottleneck as bn
import multiprocessing as mp
import time
from functools import partial
from tqdm import tqdm  # For progress bars, install with: pip install tqdm

# Numba support - optional but recommended
try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Numba not found. Install with 'pip install numba' for additional speedups.")

# Cache for memoization
_cache = {}

def memoize(func):
    """Simple memoization decorator to cache results."""
    def wrapper(*args, **kwargs):
        # Create a key from the function arguments
        key = (func.__name__, args, frozenset(kwargs.items()))
        if key not in _cache:
            _cache[key] = func(*args, **kwargs)
        return _cache[key]
    return wrapper

def clear_cache():
    """Clear the memoization cache."""
    global _cache
    _cache = {}
    
if HAS_NUMBA:
    @nb.njit(parallel=True)
    def compute_residuals_numba(model_chunk, x_obs, x_sig, weights):
        """
        Compute residuals using numba for better performance.
        
        Parameters:
        -----------
        model_chunk : numpy.ndarray
            Chunk of the model grid
        x_obs : numpy.ndarray
            Observed values
        x_sig : numpy.ndarray
            Uncertainties on observed values
        weights : numpy.ndarray
            Weights for each observation
        
        Returns:
        --------
        numpy.ndarray
            Squared residuals
        """
        # Preallocate output array
        output_shape = model_chunk.shape
        residuals = np.empty(output_shape, dtype=np.float32)
        
        # Calculate residuals in parallel
        for i in nb.prange(output_shape[0]):
            for j in range(output_shape[1]):
                for k in range(output_shape[2]):
                    for l in range(output_shape[3]):
                        for m in range(output_shape[4]):
                            x_sig_weights = x_sig * weights
                            for n in range(output_shape[5]):
                                if np.isnan(model_chunk[i,j,k,l,m,n]) or np.isnan(x_obs[n]) or np.isnan(x_sig_weights[n]):
                                    residuals[i,j,k,l,m,n] = np.nan
                                else:
                                    residuals[i,j,k,l,m,n] = ((model_chunk[i,j,k,l,m,n] - x_obs[n]) / x_sig_weights[n]) ** 2
        
        return residuals

def compute_chi_sq_optimized(star_index, MODEL, VALID, THETA, Del_Z, Del_magb, Del_mi, Del_mf,
                           metal_Z, agb_masses, initial_masses, final_masses,
                           master_table, get_star_params, get_stats, write_parameter_stats=None,
                           save_results=True, output_dir=None):
    """
    Optimized version of the compute_chi_sq function.
    
    Parameters:
    -----------
    star_index : int
        Index of the star in master_table
    MODEL, VALID, THETA, etc. : numpy.ndarray
        Model arrays needed for computation
    master_table : astropy.table
        Table containing star data
    get_star_params : function
        Function to extract star parameters
    get_stats : function
        Function to compute statistics
    write_parameter_stats : function, optional
        Function to write parameter statistics to file
    save_results : bool, default=True
        Whether to save results to file
    output_dir : str, optional
        Directory to save output files, defaults to current directory
    
    Returns:
    --------
    tuple
        (models, tracks, chis, probs)
    """
    if output_dir is None:
        output_dir = '/nexus/posix0/MIA-astro-env/hxr/adimoff/BinaryStars/Accretion_New/chi_sq_info/'
    
    # Ensure output directory exists
    if save_results and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Get star parameters
    star_name, obs_names_II, obs_elems_nums, obs_elems_II, obs_errrs_II, X_obs, X_sig, type_flag = get_star_params(master_table, star_index)
    
    # Handle NaN values efficiently
    nan_mask = np.isnan(X_obs)
    len_nans = np.sum(nan_mask)
    non_nan_obs_length = len(X_obs) - len_nans
    
    # Handle NaN uncertainties
    nan_sig_mask = np.isnan(X_sig) & ~np.isnan(X_obs)
    X_sig = X_sig.copy()  # Create a copy to avoid modifying the original
    X_sig[nan_sig_mask] = X_obs[nan_sig_mask] / 2
    X_obs = X_obs.copy()
    X_obs[nan_sig_mask] = 0.00
    
    # Apply metallicity constraint
    arr_metal = [-2.1522884, -1.6751671, -1.1522883, -0.6751671, -0.3741371, -0.15228835]
    if X_obs[0] < arr_metal[0]:
        metal_VALID = np.array([True, True, False, False, False, False])
    else:
        metal_VALID = (np.abs(arr_metal - X_obs[0]) < 3.0 * X_sig[0])
        
    # Apply validation mask
    new_VALID = metal_VALID[:, None, None, None, None] * VALID
    
    # Define spacing
    new_spacing = 1
    
    # Compute weights
    weights = np.ones(len(X_obs))
    weights[:3] = 3./non_nan_obs_length  # Surface parameters
    # Add specific weights for special elements
    for i, special_idx in enumerate([4, 8, 10, 18]): # elements carbon, ... 
        if special_idx < len(weights):
            weights[special_idx] = [1.5, 2.0, 2.0, 2.0][i]
            
    # Use strided views for efficiency
    subsample_MODEL = MODEL[:,:,:,:,::new_spacing,:]
    subsample_VALID = new_VALID[:,:,:,:,::new_spacing]
    subsample_THETA = THETA[:,:,:,:,::new_spacing]
    
    # Compute residuals
    start_time = time.time()
    
    if HAS_NUMBA and subsample_MODEL.size < 1e9:  # Only use Numba for reasonably sized arrays
        # Reshape for Numba function
        X_obs_array = np.asarray(X_obs, dtype=np.float32)
        X_sig_array = np.asarray(X_sig, dtype=np.float32)
        weights_array = np.asarray(weights, dtype=np.float32)
        
        # Use Numba for the computation-intensive part
        residuals_sq = compute_residuals_numba(subsample_MODEL, X_obs_array, X_sig_array, weights_array)
    else:
        # Use vectorized NumPy operations
        X_obs_expanded = X_obs.reshape(1, 1, 1, 1, 1, -1)
        X_sig_weights = (X_sig * weights).reshape(1, 1, 1, 1, 1, -1)
        residuals_sq = np.square((subsample_MODEL - X_obs_expanded) / X_sig_weights)
    
    # Compute chi squared
    chi_sq = bn.nansum(residuals_sq, axis=-1)
    chi_sq[~subsample_VALID] = np.inf
    
    # Save results if requested
    if save_results:
        chi_file = os.path.join(output_dir, f'{star_name}_rough_chis_weights.npy')
        with open(chi_file, 'wb') as chisq_handler:
            np.save(chisq_handler, chi_sq)
    
    # Find best models
    best_inds_all = np.array(np.unravel_index(np.argsort(chi_sq.flatten()), 
                                             chi_sq.shape, order='C')).T
    
    # Compute likelihood
    chi_sq_normalized = chi_sq - np.nanmin(chi_sq)
    log_like = -chi_sq_normalized
    like = np.exp(log_like)
    like_norm = like / np.nansum(like)
    
    # Combine with priors
    log_THETA = np.log(subsample_THETA)
    prob = like_norm * np.exp(log_THETA - np.log(np.nansum(subsample_THETA)))
    
    # Collect results
    probs = []
    chis = []
    models = []
    tracks = []
    
    # Use a limit to avoid collecting too many models
    max_models = 5000
    counter = 0
    
    for ind in best_inds_all:
        if prob[tuple(ind)] > 0.0 and chi_sq[tuple(ind)] < np.inf:
            probs.append(prob[tuple(ind)])
            chis.append(chi_sq[tuple(ind)])
            models.append(ind[-1]*new_spacing)
            tracks.append(ind[0:-1])
            counter += 1
            if counter >= max_models:
                break
    
    # Convert to arrays
    probs = np.array(probs)
    chis = np.array(chis)
    models = np.array(models)
    tracks = np.array(tracks)
    
    # Compute marginalized probabilities for each parameter
    subsample_Del_Z = Del_Z[:,:,:,:,::new_spacing]
    subsample_Del_magb = Del_magb[:,:,:,:,::new_spacing]
    subsample_Del_mi = Del_mi[:,:,:,:,::new_spacing]
    subsample_Del_mf = Del_mf[:,:,:,:,::new_spacing]
    
    # Sum over different dimensions to marginalize
    prob_Z = np.sum(np.sum(np.sum(np.sum(like, axis=4), axis=3), axis=2), axis=1) * \
             subsample_Del_Z[:,0,0,0,0] / np.sum(subsample_Del_Z)
    
    prob_magb = np.sum(np.sum(np.sum(np.sum(like, axis=4), axis=3), axis=2), axis=0) * \
                subsample_Del_magb[0,:,0,0,0] / np.sum(subsample_Del_magb)
    
    prob_mi = np.sum(np.sum(np.sum(np.sum(like, axis=4), axis=3), axis=1), axis=0) * \
              subsample_Del_mi[0,0,:,0,0] / np.sum(subsample_Del_mi)
    
    prob_mf = np.sum(np.sum(np.sum(np.sum(like, axis=4), axis=2), axis=1), axis=0) * \
              subsample_Del_mf[0,0,0,:,0] / np.sum(subsample_Del_mf)
    
    # Calculate statistics for each parameter
    metal_stats = get_stats(metal_Z, prob_Z)
    agb_stats = get_stats(agb_masses, prob_magb)
    mi_stats = get_stats(initial_masses, prob_mi)
    mf_stats = get_stats(final_masses, prob_mf)
    
    # Write statistics if requested and function provided
    if save_results and write_parameter_stats is not None:
        write_parameter_stats(star_name, metal_Z, agb_masses, initial_masses, final_masses,
                         prob_Z, prob_magb, prob_mi, prob_mf,
                         metal_stats, agb_stats, mi_stats, mf_stats)
    
    print(f"Processed {star_name} in {time.time() - start_time:.2f} seconds")
    return models, tracks, chis, probs

def compute_chi_sq_chunked(star_index, MODEL, VALID, THETA, Del_Z, Del_magb, Del_mi, Del_mf,
                         metal_Z, agb_masses, initial_masses, final_masses,
                         master_table, get_star_params, get_stats, write_parameter_stats=None,
                         save_results=True, output_dir=None, chunk_size=10):
    """
    Process the model grid in chunks to reduce memory usage.
    
    Parameters:
    -----------
    star_index : int
        Index of the star in master_table
    MODEL, VALID, etc. : numpy.ndarray
        Model arrays needed for computation
    chunk_size : int, default=10
        Number of models to process in each chunk
    
    Returns:
    --------
    tuple
        (models, tracks, chis, probs)
    """
    if output_dir is None:
        output_dir = '/nexus/posix0/MIA-astro-env/hxr/adimoff/BinaryStars/Accretion_New/chi_sq_info/'
        
    # Ensure output directory exists
    if save_results and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # Get star parameters
    star_name, obs_names_II, obs_elems_nums, obs_elems_II, obs_errrs_II, X_obs, X_sig, type_flag = \
        get_star_params(master_table, star_index)
    
    # Handle NaN values efficiently
    nan_mask = np.isnan(X_obs)
    len_nans = np.sum(nan_mask)
    non_nan_obs_length = len(X_obs) - len_nans
    
    # Handle NaN uncertainties
    nan_sig_mask = np.isnan(X_sig) & ~np.isnan(X_obs)
    X_sig = X_sig.copy()
    X_sig[nan_sig_mask] = X_obs[nan_sig_mask] / 2
    X_obs = X_obs.copy()
    X_obs[nan_sig_mask] = 0.00
    
    # Apply metallicity constraint
    arr_metal = [-2.1522884, -1.6751671, -1.1522883, -0.6751671, -0.3741371, -0.15228835]
    if X_obs[0] < arr_metal[0]:
        metal_VALID = np.array([True, True, False, False, False, False])
    else:
        metal_VALID = (np.abs(arr_metal - X_obs[0]) < 3.0 * X_sig[0])
    
    # Apply validation mask
    new_VALID = metal_VALID[:, None, None, None, None] * VALID
    new_spacing = 1
    
    # Compute weights
    weights = np.ones(len(X_obs))
    weights[:3] = 3./non_nan_obs_length  # Surface parameters
    # Add specific weights for special elements
    for i, special_idx in enumerate([4, 8, 10, 18]):
        if special_idx < len(weights):
            weights[special_idx] = [1.5, 2.0, 2.0, 2.0][i]
            
    # Initialize full result array
    chi_sq_shape = list(VALID.shape)
    chi_sq_shape[4] = chi_sq_shape[4] // new_spacing  # Adjust for spacing
    all_chi_sq = np.full(chi_sq_shape, np.inf, dtype=np.float32)
    
    # Get the size of the last dimension (number of models per parameter combination)
    total_models = VALID.shape[4]
    
    # Process in chunks along the model dimension
    print(f"Processing {total_models} models in chunks of {chunk_size}...")
    start_time = time.time()
    
    # Create a progress bar
    pbar = tqdm(total=total_models, desc=f"Processing {star_name}")
    
    for chunk_start in range(0, total_models, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_models)
        chunk_size_actual = chunk_end - chunk_start
        
        # Extract model chunk - only load necessary data into memory
        model_chunk = MODEL[:,:,:,:,chunk_start:chunk_end:new_spacing,:]
        valid_chunk = new_VALID[:,:,:,:,chunk_start:chunk_end:new_spacing]
        
        # Compute residuals for this chunk
        X_obs_expanded = X_obs.reshape(1, 1, 1, 1, 1, -1)
        X_sig_weights = (X_sig * weights).reshape(1, 1, 1, 1, 1, -1)
        residuals_sq_chunk = np.square((model_chunk - X_obs_expanded) / X_sig_weights)
        
        # Compute chi squared for this chunk
        chi_sq_chunk = bn.nansum(residuals_sq_chunk, axis=-1)
        chi_sq_chunk[~valid_chunk] = np.inf
        
        # Store in the full array
        all_chi_sq[:,:,:,:,chunk_start//new_spacing:(chunk_end-1)//new_spacing+1] = chi_sq_chunk
        
        pbar.update(chunk_size_actual)
        
    pbar.close()
    
    # The rest of the processing is the same as in compute_chi_sq_optimized
    if save_results:
        chi_file = os.path.join(output_dir, f'{star_name}_rough_chis_weights.npy')
        with open(chi_file, 'wb') as chisq_handler:
            np.save(chisq_handler, all_chi_sq)
    
    # Find best models
    best_inds_all = np.array(np.unravel_index(np.argsort(all_chi_sq.flatten()),
                                             all_chi_sq.shape, order='C')).T
    
    # Compute likelihood
    chi_sq_normalized = all_chi_sq - np.nanmin(all_chi_sq)
    log_like = -chi_sq_normalized
    like = np.exp(log_like)
    like_norm = like / np.nansum(like)
    
    # Use THETA with adjusted shape
    subsample_THETA = THETA[:,:,:,:,::new_spacing]
    
    # Combine with priors
    log_THETA = np.log(subsample_THETA)
    prob = like_norm * np.exp(log_THETA - np.log(np.nansum(subsample_THETA)))
    
    # Collect results
    probs = []
    chis = []
    models = []
    tracks = []
    
    # Limit to avoid collecting too many models
    max_models = 5000
    counter = 0
    
    for ind in best_inds_all:
        if counter >= max_models:
            break
        if prob[tuple(ind)] > 0.0 and all_chi_sq[tuple(ind)] < np.inf:
            probs.append(prob[tuple(ind)])
            chis.append(all_chi_sq[tuple(ind)])
            models.append(ind[-1]*new_spacing)
            tracks.append(ind[0:-1])
            counter += 1
    
    # Convert to arrays
    probs = np.array(probs)
    chis = np.array(chis)
    models = np.array(models)
    tracks = np.array(tracks)
    
    # Compute marginalized probabilities
    subsample_Del_Z = Del_Z[:,:,:,:,::new_spacing]
    subsample_Del_magb = Del_magb[:,:,:,:,::new_spacing]
    subsample_Del_mi = Del_mi[:,:,:,:,::new_spacing]
    subsample_Del_mf = Del_mf[:,:,:,:,::new_spacing]
    
    prob_Z = np.sum(np.sum(np.sum(np.sum(like, axis=4), axis=3), axis=2), axis=1) * \
             subsample_Del_Z[:,0,0,0,0] / np.sum(subsample_Del_Z)
    
    prob_magb = np.sum(np.sum(np.sum(np.sum(like, axis=4), axis=3), axis=2), axis=0) * \
                subsample_Del_magb[0,:,0,0,0] / np.sum(subsample_Del_magb)
    
    prob_mi = np.sum(np.sum(np.sum(np.sum(like, axis=4), axis=3), axis=1), axis=0) * \
              subsample_Del_mi[0,0,:,0,0] / np.sum(subsample_Del_mi)
    
    prob_mf = np.sum(np.sum(np.sum(np.sum(like, axis=4), axis=2), axis=1), axis=0) * \
              subsample_Del_mf[0,0,0,:,0] / np.sum(subsample_Del_mf)
    
    # Calculate statistics for each parameter
    metal_stats = get_stats(metal_Z, prob_Z)
    agb_stats = get_stats(agb_masses, prob_magb)
    mi_stats = get_stats(initial_masses, prob_mi)
    mf_stats = get_stats(final_masses, prob_mf)
    
    if save_results and write_parameter_stats is not None:
        write_parameter_stats(star_name, metal_Z, agb_masses, initial_masses, final_masses,
                         prob_Z, prob_magb, prob_mi, prob_mf,
                         metal_stats, agb_stats, mi_stats, mf_stats)
    
    print(f"Processed {star_name} in {time.time() - start_time:.2f} seconds")
    return models, tracks, chis, probs

def process_stars_in_parallel(star_indices, MODEL, VALID, THETA, Del_Z, Del_magb, Del_mi, Del_mf,
                           metal_Z, agb_masses, initial_masses, final_masses,
                           master_table, get_star_params, get_stats, write_parameter_stats=None,
                           save_results=True, output_dir=None, use_chunking=False, chunk_size=10,
                           num_processes=None):
    """
    Process multiple stars in parallel using multiprocessing.
    
    Parameters:
    -----------
    star_indices : list
        List of star indices to process
    MODEL, VALID, etc. : numpy.ndarray
        Model arrays needed for computation
    num_processes : int, optional
        Number of processes to use, defaults to number of CPU cores
    use_chunking : bool, default=False
        Whether to use chunked processing for lower memory usage
    
    Returns:
    --------
    list
        List of results for each star [(models, tracks, chis, probs), ...]
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"Processing {len(star_indices)} stars using {num_processes} processes")
    
    # Choose the processing function based on memory requirements
    if use_chunking:
        process_func = partial(compute_chi_sq_chunked, 
                             MODEL=MODEL, VALID=VALID, THETA=THETA,
                             Del_Z=Del_Z, Del_magb=Del_magb, Del_mi=Del_mi, Del_mf=Del_mf,
                             metal_Z=metal_Z, agb_masses=agb_masses, 
                             initial_masses=initial_masses, final_masses=final_masses,
                             master_table=master_table, get_star_params=get_star_params,
                             get_stats=get_stats, write_parameter_stats=write_parameter_stats,
                             save_results=save_results, output_dir=output_dir,
                             chunk_size=chunk_size)
    else:
        process_func = partial(compute_chi_sq_optimized, 
                             MODEL=MODEL, VALID=VALID, THETA=THETA,
                             Del_Z=Del_Z, Del_magb=Del_magb, Del_mi=Del_mi, Del_mf=Del_mf,
                             metal_Z=metal_Z, agb_masses=agb_masses, 
                             initial_masses=initial_masses, final_masses=final_masses,
                             master_table=master_table, get_star_params=get_star_params,
                             get_stats=get_stats, write_parameter_stats=write_parameter_stats,
                             save_results=save_results, output_dir=output_dir)
    
    # Run in parallel
    start_time = time.time()
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_func, star_indices), 
                          total=len(star_indices), 
                          desc="Processing stars"))
    
    print(f"Processed {len(star_indices)} stars in {time.time() - start_time:.2f} seconds")
    return results

def profile_compute_chi_sq(star_index, MODEL, VALID, THETA, Del_Z, Del_magb, Del_mi, Del_mf,
                         metal_Z, agb_masses, initial_masses, final_masses,
                         master_table, get_star_params, get_stats, write_parameter_stats=None):
    """
    Profile the compute_chi_sq function to identify bottlenecks.
    
    Parameters:
    -----------
    star_index : int
        Index of the star to profile
    MODEL, VALID, etc. : numpy.ndarray
        Model arrays needed for computation
        
    Returns:
    --------
    pstats.Stats
        Profiling statistics
    """
    import cProfile
    import pstats
    import io
    
    # Create profiler
    pr = cProfile.Profile()
    pr.enable()
    
    # Run the function
    compute_chi_sq_optimized(star_index, MODEL, VALID, THETA, Del_Z, Del_magb, Del_mi, Del_mf,
                           metal_Z, agb_masses, initial_masses, final_masses,
                           master_table, get_star_params, get_stats, write_parameter_stats,
                           save_results=False)
    
    pr.disable()
    
    # Print stats sorted by time
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())
    
    return ps