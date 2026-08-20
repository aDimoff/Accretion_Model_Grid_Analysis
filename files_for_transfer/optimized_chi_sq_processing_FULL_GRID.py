# Optimizing Chi-Square Computation for Binary Star Models

# Import Required Libraries
import numpy as np
import bottleneck as bn
import os
import time
import matplotlib.pyplot as plt
from astropy.table import Table, join, vstack, hstack
from astropy.io import fits, ascii
import multiprocessing as mp
from functools import partial, lru_cache
import multiprocessing as mp
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings("ignore")

# Import parallel processing tools
from concurrent.futures import ProcessPoolExecutor
try:
    from joblib import Parallel, delayed, Memory
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    print("joblib not found. Install with: pip install joblib")

# Import Numba for JIT compilation
try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("numba not found. Install with: pip install numba")

# Import tqdm for progress tracking
try:
    from tqdm import tqdm
    from tqdm.notebook import tqdm as tqdm_notebook
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("tqdm not found. Install with: pip install tqdm")
# Import the key functions from the original notebook
import sys
sys.path.append('/nexus/posix0/MIA-astro-env/hxr/adimoff/BinaryStars/Accretion_New')

# Import the original functions
# Note: This assumes you've converted your notebook to a Python module,
# or we can extract the functions from the notebook

# set OG data dir
original_data_dir = '/nexus/posix0/MIA-astro-env/hxr/adimoff/BinaryStars/Accretion_New'

# This guard is required for multiprocessing to work properly
if __name__ == '__main__':
    # Load model grid and validation mask (standard way...)
    MODEL = np.load(os.path.join(original_data_dir, 'model_grid_full_size.dat'))
    VALID = np.load(os.path.join(original_data_dir, 'valid_grid_full_size.dat'))

    # read compressed .npz grid of models and validation mask
    # data = np.load('model_grid_full_size.npz')
    # MODEL = data['MODEL']
    # VALID = data['VALID']

    print(f"MODEL shape: {MODEL.shape}")
    print(f"VALID shape: {VALID.shape}")
    
    # Load the master table and other necessary data
    # This is a simplified version - you may need to adjust according to your actual data loading process
    try:
        # Read in observational data
        Obs_Data_Table = ascii.read(os.path.join(original_data_dir, '../../abund_data/X_Fe_table_3.dat'), delimiter='&')
        param_mass_table = ascii.read(os.path.join(original_data_dir, '../../abund_data/param_mass_table.dat'), format='latex')
        
        full_obs_data_table = join(param_mass_table, Obs_Data_Table, keys='stars')
        
        cristallo_data = ascii.read(os.path.join(original_data_dir, '../../abund_data/Cristallo_2016.dat'))
        goswami_2020_data = ascii.read(os.path.join(original_data_dir, '../../abund_data/Gos_Rat_Gos_2020.dat'))
        goswami_2021_data = ascii.read(os.path.join(original_data_dir, '../../abund_data/Gos_Rat_Gos_CEMPs_2021.dat'))
        DeCastro_Roriz = ascii.read(os.path.join(original_data_dir, '../../abund_data/DeCastro_Roriz_Table.dat'))

        #SAGA_MP = ascii.read(os.path.join(original_data_dir, '../../abund_data/SAGA_metalpoor_heavies_APTable.dat'))
        #SAGA_MR = ascii.read(os.path.join(original_data_dir, '../../abund_data/SAGA_metalrich_heavies_APTable.dat'))

        # Stack all tables
        master_table = vstack([full_obs_data_table, cristallo_data, goswami_2020_data, goswami_2021_data, DeCastro_Roriz], join_type='inner') # SAGA_MP, SAGA_MR
        obs_names = np.array(['C','C_sig','FeH','FeH_sig','Sr','Sr_sig','Y','Y_sig','Zr','Zr_sig','Nb','Nb_sig','Mo','Mo_sig','Ru','Ru_sig','Ba','Ba_sig','La','La_sig','Ce','Ce_sig','Pr','Pr_sig','Nd','Nd_sig','Sm','Sm_sig','Eu','Eu_sig','Dy','Dy_sig','Pb','Pb_sig'])
        
        print(f"Loaded master_table with {len(master_table)} rows")       
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Will need to recreate all arrays from the original notebook.")

    # If arrays are missing, recreate them from the original notebook
    # This is a simplified version that recreates the core arrays needed for computation

    # Define split_surf_file function (needed for parameter creation)
    def split_surf_file(surf_file):
        """
        from file string, extract
        model name,
        metallicity, 
        AGB mass
        inital mass
        final mass
        """
        model_name = surf_file.split('/')[-1][12:]
        metalZ = model_name.split('_')[-1][1:]
        AGB_mass = model_name.split('_')[-2][3:] 
        init_mass = (model_name.split('_')[0][1]+'.'+model_name.split('_')[0][3:5])
        final_mass = (model_name.split('_')[1][1]+'.'+model_name.split('_')[1][3:5])
        return metalZ, AGB_mass, init_mass, final_mass

    # Define create_delta function
    def create_delta(input_list):
        """
        Create delta array for given input list - allows computation of grid cell volume
        """
        delta_list = []
        for i in range(len(input_list)-1):
            diff = (float(input_list[i+1]) - float(input_list[i])) / 2
            delta_list.append(diff)
        delta_list.insert(0, delta_list[0])
        return np.array(delta_list)

    # Load surface files list
    list_surf_files = np.genfromtxt(os.path.join(original_data_dir, 'list_surface_files_full'), dtype=str)

    # Generate indices
    split = [split_surf_file(sf) for sf in list_surf_files if split_surf_file(sf) is not None]

    # Select unique indices
    unique = [list(np.unique(spl)) for spl in np.array(split).T]

    # Convert unique values to floats
    # Convert Z values to floats
    metal_Z = []
    for z in unique[0]:
        capZ = np.log10(float('0.'+z)/0.0142)
        metal_Z.append(capZ)
    metal_Z = np.array(metal_Z)

    # Convert agb values to floats
    agb_masses = []
    for m_agb in unique[1]:
        value = float(m_agb[0] + '.' + m_agb[-1])
        agb_masses.append(value)
    agb_masses = np.array(agb_masses)

    # Convert initial and final masses to floats
    initial_masses = []
    for m_initial in unique[2]:
        value = float(m_initial)
        initial_masses.append(value)
    initial_masses = np.array(initial_masses)

    final_masses = []
    for m_final in unique[3]:
        value = float(m_final)
        final_masses.append(value)
    final_masses = np.array(final_masses)

    # Create delta arrays for the grid
    delta_metal_Z = create_delta(metal_Z)
    delta_m_agb = create_delta(agb_masses)
    delta_m_init = create_delta(initial_masses)
    delta_m_final = create_delta(final_masses)
    delta_time = np.ones(30000)

    # Create Del_* arrays using meshgrid
    Del_Z, Del_magb, Del_mi, Del_mf, Del_t = np.meshgrid(
        delta_metal_Z, delta_m_agb, delta_m_init, delta_m_final, delta_time, indexing="ij")

    # Create THETA grid volume array
    # Divide by sums for proper priors
    top_prod = Del_magb * Del_mi * VALID
    THETA = Del_Z/np.sum(Del_Z) * top_prod/(np.sum(top_prod)) * Del_mf/np.sum(Del_mf) * Del_t/np.sum(Del_t)
    print(f"THETA shape: {THETA.shape}")


def get_star_params(table_in_question, star_index):
    star_name = table_in_question['stars'][star_index]
    # parse observed abunds and errors into array
    obs_data = [table_in_question['C'][star_index],   table_in_question['C_sig'][star_index],   # C
                table_in_question['FeH'][star_index], table_in_question['FeH_sig'][star_index], # (Fe)
                table_in_question['Sr'][star_index],  table_in_question['Sr_sig'][star_index],  # Sr
                table_in_question['Y'][star_index],   table_in_question['Y_sig'][star_index],   # Y
                table_in_question['Zr'][star_index],  table_in_question['Zr_sig'][star_index],  # Zr
                table_in_question['Nb'][star_index],  table_in_question['Nb_sig'][star_index],  # Nb
                table_in_question['Mo'][star_index],  table_in_question['Mo_sig'][star_index],  # Mo
                table_in_question['Ru'][star_index],  table_in_question['Ru_sig'][star_index],  # Ru
                table_in_question['Ba'][star_index],  table_in_question['Ba_sig'][star_index],  # Ba
                table_in_question['La'][star_index],  table_in_question['La_sig'][star_index],  # La
                table_in_question['Ce'][star_index],  table_in_question['Ce_sig'][star_index],  # Ce
                table_in_question['Pr'][star_index],  table_in_question['Pr_sig'][star_index],  # Pr
                table_in_question['Nd'][star_index],  table_in_question['Nd_sig'][star_index],  # Nd
                table_in_question['Sm'][star_index],  table_in_question['Sm_sig'][star_index],  # Sm
                table_in_question['Eu'][star_index],  table_in_question['Eu_sig'][star_index],  # Eu
                table_in_question['Dy'][star_index],  table_in_question['Dy_sig'][star_index],  # Dy
                table_in_question['Pb'][star_index],  table_in_question['Pb_sig'][star_index]]  # Pb
    # surface parameters
    logg = table_in_question['logg'][star_index]
    e_logg = table_in_question['e_logg'][star_index]
    Teff = table_in_question['Teff'][star_index]
    e_Teff = table_in_question['e_Teff'][star_index]
    log_Teff = np.log10(Teff); log_e_Teff = (e_Teff**2/(Teff*np.log10(e_Teff))**2)**(1/2)
    FeH = table_in_question['FeH'][star_index]
    eFeH = table_in_question['FeH_sig'][star_index]
# -------------------------------
    # add a flag for classification purposes
    flag = 'empty'
    # identify elemental groups for classification purposes
    ls = bn.nanmean([table_in_question['Sr'][star_index],table_in_question['Y'][star_index],
                     table_in_question['Zr'][star_index]])#, table_in_question['Nb'][star_index]
                     #table_in_question['Mo'][star_index]])#,table_in_question['Ru'][star_index]])
    hs = bn.nanmean([table_in_question['Ba'][star_index],table_in_question['La'][star_index],
                     table_in_question['Ce'][star_index],#table_in_question['Pr'][star_index],
                     table_in_question['Nd'][star_index]])    

    C_enhanced = table_in_question['C'][star_index] > 0.50 # or 0.70? # or 0.50?
    Metal_Poor = table_in_question['FeH'][star_index] < -1.00

    dash_s = (hs - table_in_question['Eu'][star_index] > 0.50) and hs > 0.90 ## Beers + Christlieb 2005 / Jorissen 2016
    #dash_s = (hs > 1.00) and (hs - table_in_question['Eu'][star_index] > 0.50) ## Sivarani 2006
    #dash_s = (hs > 1.00) and (hs - table_in_question['Eu'][star_index] > 0.00) and (table_in_question['Eu'][star_index] < 1.00) ## Jonsell 2006
    #dash_s = (hs - table_in_question['Eu'][star_index] > 0.00) and hs > 0.90 ## Abate 2016
    
    dash_rs = (hs - table_in_question['Eu'][star_index] < 0.50) and (hs - table_in_question['Eu'][star_index] > 0.10) ## Beers + Christlieb 2005 / Jorissen 2016
    #dash_rs = (hs > 0.0) and (hs < 0.50) ## Sivarani 2006
    #dash_rs = (table_in_question['Eu'][star_index] > 1.00) and (hs > 1.00) and (hs - table_in_question['Eu'][star_index] > 0.0) ## Abate 2016
    #dash_rs = (table_in_question['Eu'][star_index] > 1.00) and (hs - table_in_question['Eu'][star_index] > 0.0) and (hs - table_in_question['Eu'][star_index] < 1.0) ## Hansen 2019
    
    #dash_r = table_in_question['Eu'][star_index] > 1.00 ## Beers + Christlieb 2005
    dash_r = table_in_question['Eu'][star_index] > 1.00 and hs < 0.0 ## Abate 2016
    
    ## CEMP: [C/Fe] ≥ 0.7
    ## CEMP-r/s: [Ba/Fe] ≥ 1.0, [Eu/Fe] ≥ 1.0
    ## i) 0.0 ≤ [Ba/Eu] ≤ 1.0 and/or 0.0 ≤ [La/Eu] ≤ 0.7;
    ## CEMP-s: [Ba/Fe] ≥ 1.0
    ## i.) [Eu/Fe] < 1.0, [Ba/Eu] > 0.0 and/or [La/Eu] > 0.5;
    ## ii.) [Eu/Fe] ≥ 1.0, [Ba/Eu] > 1.0 and/or [La/Eu] > 0.7.

    ## CAMILLA'S CRITERION BASED ON SR AND BA -- LS AND HS
    #CEMP-no [Sr/Ba] > 0.75 New classification
    #if C_enhanced and Metal_Poor and (ls - hs > 0.75): # Hansen 2019
    #    flag = 'CEMP-no'
    #if C_enhanced and Metal_Poor and hs < 0.0: # Beers + Christlieb 2005 / Jorissen 2016
    #    flag = 'CEMP-no'
    ## THE ORDER OF THESE MATTERS
    #if C_enhanced and Metal_Poor and (ls < -1.5): # Hansen 2019
    #    flag = 'CEMP-r'
    #if C_enhanced and Metal_Poor and dash_r: 
    #    flag = 'CEMP-r'
    if C_enhanced and Metal_Poor and (ls - hs > -0.60) and (ls - hs < 0.60): # Hansen 2019
        flag = 'CEMP-s'
    #elif C_enhanced and Metal_Poor and (ls - hs > -1.5) and (ls - hs > -0.5): # Hansen 2019
    #    flag = 'CEMP-r/s'
    elif C_enhanced and Metal_Poor and dash_rs: 
        flag = 'CEMP-r/s'
    elif C_enhanced and Metal_Poor and dash_s: 
        flag = 'CEMP-s'
    ## add flag for CH stars as those at higher metallicities but share same characteristics
    elif Metal_Poor and C_enhanced: # ...? These stars are metal poor, but not SO carbon enriched... this is tough, some could be r/s?
        flag = 'CH'
    # Escorza's strong / weak Ba star criterion:
    elif not Metal_Poor and ls > 0.2 or hs > 0.2 and not C_enhanced:
        if ls > 0.8 or hs > 0.8:
            flag = 'SBa'
        elif (ls > 0.2 and ls < 0.8) or (hs > 0.2 and hs < 0.8):
            flag = 'WBa'
# -------------------------------
    ## assign masses based on observations
    ## or from previous population studies
    for col_name in table_in_question.columns:
        if col_name == 'vis':   
            m_star = table_in_question['vis'][star_index]
            e_m_star = table_in_question['e_vis'][star_index]
        # for tagged Barium stars, avg mass is ~2.50
        # -- doing this for these stars almost guarantees a fit. we don't want to start with a mass constraint
        elif flag == 'WBa' or flag == 'SBa':
            m_star = np.nan # 2.50
            e_m_star = np.nan # 1.00
        # CEMP stars, avg mass is ~0.80
        elif flag == 'CEMP-s' or flag == 'CEMP-r/s' or flag == 'CH':
            m_star = np.nan # 0.80
            e_m_star = np.nan # 1.00
        # if not, give it a nan
        else:
            m_star = np.nan #1.50
            e_m_star = np.nan #1.50
# -------------------------------
    # element names (skip errors in names list)
    obs_names_I_ = obs_names[0:-1:2]
    obs_elems_nums = [6,38,39,40,41,42,44,56,57,58,59,60,62,63,66,82]    
    # observed abundances and errors go into obs_data array
    obs_elems = obs_data[0:-1:2]
    obs_errrs = obs_data[1::2]
    # collect observational data and uncertanties in arrays
    # skip Fe (index 1) in these lists...
    obs_names_II = []
    obs_elems_II = []
    obs_errrs_II = []
    for i in range(len(obs_elems)): 
        if i != 1:
            obs_names_II.append(obs_names_I_[i])
            obs_elems_II.append(obs_elems[i])
            obs_errrs_II.append(obs_errrs[i])
    #print('Collect observational data for the star in an array')
    X_obs = np.concatenate([[FeH,log_Teff,logg,m_star],obs_elems_II])
    X_sig = np.concatenate([[eFeH,log_e_Teff,e_logg,e_m_star],obs_errrs_II])
            
    return star_name, obs_names_II, obs_elems_nums, obs_elems_II, obs_errrs_II, X_obs, X_sig, flag

# Function to calculate statistics for each parameter
def get_stats(values, weights):
    # Normalize weights
    weights = weights / np.sum(weights)
    # Calculate mean
    mean = np.sum(values * weights)
    # Calculate median (50th percentile)
    cumsum = np.cumsum(weights[np.argsort(values)])
    median = values[np.argsort(values)][np.searchsorted(cumsum, 0.5)]
    # Calculate 16th and 84th percentiles (68% confidence interval)
    p16 = values[np.argsort(values)][np.searchsorted(cumsum, 0.16)]
    p84 = values[np.argsort(values)][np.searchsorted(cumsum, 0.84)]
    
    return mean, median, p16, p84

def write_parameter_stats(star_name, metal_Z, agb_masses, initial_masses, final_masses, 
                         prob_Z, prob_magb, prob_mi, prob_mf,
                         metal_stats, agb_stats, mi_stats, mf_stats,
                         best_model_params=None):
    
    stats_dir = '/nexus/posix0/MIA-astro-env/hxr/adimoff/BinaryStars/Accretion_New/statistics_numba_FULL_GRID/'
    #os.makedirs(stats_dir, exist_ok=True)

    # Create output filename in statistics directory
    outfile = os.path.join(stats_dir, f'parameter_stats_{star_name}.txt')
    # do the writing
    with open(outfile, 'w') as f:
        f.write(f'Parameter Statistics for {star_name}\n')
        f.write('=' * 50 + '\n\n')
        # Write metallicity stats
        f.write('Metallicity [Fe/H]\n')
        f.write('-' * 20 + '\n')
        f.write(f'Mean: {metal_stats[0]:.3f}\n')
        f.write(f'Median: {metal_stats[1]:.3f}\n')
        f.write(f'68% confidence interval: [{metal_stats[2]:.3f}, {metal_stats[3]:.3f}]\n')
        f.write('Probability distribution:\n')
        for z, p in zip(metal_Z, prob_Z):
            f.write(f'{z:.3f}: {p:.3e}\n')
        f.write('\n')
        # Write AGB mass stats  
        f.write('AGB Mass (M⊙)\n')
        f.write('-' * 20 + '\n')
        f.write(f'Mean: {agb_stats[0]:.3f}\n')
        f.write(f'Median: {agb_stats[1]:.3f}\n')
        f.write(f'68% confidence interval: [{agb_stats[2]:.3f}, {agb_stats[3]:.3f}]\n')
        f.write('Probability distribution:\n')
        for m, p in zip(agb_masses, prob_magb):
            f.write(f'{m:.3f}: {p:.3e}\n')
        f.write('\n')
        # Write initial mass stats
        f.write('Initial Mass (M⊙)\n') 
        f.write('-' * 20 + '\n')
        f.write(f'Mean: {mi_stats[0]:.3f}\n')
        f.write(f'Median: {mi_stats[1]:.3f}\n')
        f.write(f'68% confidence interval: [{mi_stats[2]:.3f}, {mi_stats[3]:.3f}]\n')
        f.write('Probability distribution:\n')
        for m, p in zip(initial_masses, prob_mi):
            f.write(f'{m:.3f}: {p:.3e}\n')
        f.write('\n')
        # Write final mass stats
        f.write('Final Mass (M⊙)\n')
        f.write('-' * 20 + '\n') 
        f.write(f'Mean: {mf_stats[0]:.3f}\n')
        f.write(f'Median: {mf_stats[1]:.3f}\n')
        f.write(f'68% confidence interval: [{mf_stats[2]:.3f}, {mf_stats[3]:.3f}]\n')
        f.write('Probability distribution:\n')
        for m, p in zip(final_masses, prob_mf):
            f.write(f'{m:.3f}: {p:.3e}\n')
        f.write('\n')
        # Write mass ratio and accreted mass stats from individual accepted models
        if best_model_params is not None:
            metallicity_vals = best_model_params['metallicity']
            initial_m_vals = best_model_params['initial_mass']
            agb_m_vals = best_model_params['agb_mass']
            final_m_vals = best_model_params['final_mass']
            probs_vals = best_model_params['probabilities']
            # Compute mass ratio for each model
            mass_ratios = initial_m_vals / agb_m_vals  # Element-wise division
            f.write('Mass Ratio (M_initial / M_AGB)\n')
            f.write('-' * 20 + '\n')
            f.write(f'Mean: {np.average(mass_ratios, weights=probs_vals):.3f}\n')
            f.write(f'Median: {np.median(mass_ratios):.3f}\n')
            ratios_16 = np.percentile(mass_ratios, 16)
            ratios_84 = np.percentile(mass_ratios, 84)
            f.write(f'68% confidence interval: [{ratios_16:.3f}, {ratios_84:.3f}]\n')
            f.write('Probability distribution:\n')
            for ratio, p in zip(mass_ratios, probs_vals):
                f.write(f'{ratio:.3f}: {p:.3e}\n')
            f.write('\n')
            # Compute accreted mass for each model
            accreted_m = final_m_vals - initial_m_vals
            f.write('Accreted Mass (M_final - M_initial) (M⊙)\n')
            f.write('-' * 20 + '\n')
            f.write(f'Mean: {np.average(accreted_m, weights=probs_vals):.3f}\n')
            f.write(f'Median: {np.median(accreted_m):.3f}\n')
            accr_16 = np.percentile(accreted_m, 16)
            accr_84 = np.percentile(accreted_m, 84)
            f.write(f'68% confidence interval: [{accr_16:.3f}, {accr_84:.3f}]\n')
            f.write('Probability distribution:\n')
            for acc_mass, p in zip(accreted_m, probs_vals):
                f.write(f'{acc_mass:.3f}: {p:.3e}\n')


## 4. Implement Numba for Just-in-Time Compilation    
@nb.njit(parallel=True)
def compute_residuals_and_sum_numba(model_data, obs_data, sig_weights, valid_mask):
    """
    Compute squared residuals AND sum them in one pass using Numba
    This avoids creating the large intermediate residuals array
    """
    shape = model_data.shape[:-1]  # All dimensions except last
    result = np.zeros(shape, dtype=np.float32)
    
    # Use parallel loops for better performance
    for i in nb.prange(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[3]):
                    for m in range(shape[4]):
                        if not valid_mask[i, j, k, l, m]:
                            result[i, j, k, l, m] = np.inf
                            continue
                            
                        chi_sum = 0.0
                        valid_count = 0
                        
                        for n in range(model_data.shape[-1]):
                            model_val = model_data[i, j, k, l, m, n]
                            obs_val = obs_data[n]
                            sig_weight = sig_weights[n]
                            
                            # Skip NaN values
                            if (not np.isnan(model_val) and 
                                not np.isnan(obs_val) and 
                                not np.isnan(sig_weight) and
                                sig_weight != 0.0):
                                
                                diff = model_val - obs_val
                                residual_sq = (diff / sig_weight) ** 2
                                chi_sum += residual_sq
                                valid_count += 1
                        
                        # Set result based on whether we have valid data
                        if valid_count > 0:
                            result[i, j, k, l, m] = chi_sum
                        else:
                            result[i, j, k, l, m] = np.inf
                            
    return result

@nb.njit(parallel=True)
def compute_residuals_numba(model_data, obs_data, sig_weights):
    """
    Compute squared residuals using Numba for acceleration
    
    Parameters:
    -----------
    model_data : numpy.ndarray
        Model data array
    obs_data : numpy.ndarray
        Observed data array
    sig_weights : numpy.ndarray
        Sigma * weights array
        
    Returns:
    --------
    numpy.ndarray
        Squared residuals
    """
    # Get the shape of the model data
    shape = model_data.shape
    result = np.empty(shape, dtype=np.float32)
    
    # Use parallel loops for better performance
    for i in nb.prange(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[3]):
                    for m in range(shape[4]):
                        for n in range(shape[5]):
                            # Handle NaN values
                            if np.isnan(model_data[i,j,k,l,m,n]) or np.isnan(obs_data[n]) or np.isnan(sig_weights[n]):
                                result[i,j,k,l,m,n] = np.nan
                            else:
                                # Compute squared residual
                                diff = model_data[i,j,k,l,m,n] - obs_data[n]
                                result[i,j,k,l,m,n] = (diff / sig_weights[n]) ** 2
                                
    return result
    

# Add this after defining your Numba functions
def warmup_numba_functions():
    """Pre-compile Numba functions with representative data"""
    print("Warming up Numba functions...")
    
    # Create dummy arrays with the same shape/dtype as your real data
    dummy_model = np.random.random((6, 4, 4, 4, 100, 20)).astype(np.float32)
    dummy_obs = np.random.random(20).astype(np.float32)
    dummy_weights = np.random.random(20).astype(np.float32)
    dummy_mask = np.random.choice(a=[False, True], size=(6, 4, 4, 4, 100))
    # Force compilation
    _ = compute_residuals_numba(dummy_model, dummy_obs, dummy_weights)
    print("Numba warmup complete.")

def compute_chi_sq_numba(star_index, save_chis=False, save_models=True, save_stats=True):
    """Complete Numba-optimized version that computes chi-squared directly"""
    start_time = time.time()
    
    # Get star parameters
    star_name, obs_names_II, obs_elems_nums, obs_elems_II, obs_errrs_II, X_obs, X_sig, type_flag = get_star_params(master_table, star_index)
    print(star_name)
    # Efficient NaN handling
    nan_mask = np.isnan(X_obs)
    len_nans = np.sum(nan_mask)
    non_nan_obs_length = len(X_obs) - len_nans
    
    # Handle NaN uncertainties
    nan_sig_mask = np.isnan(X_sig) & ~np.isnan(X_obs)
    X_sig = X_sig.copy()
    X_obs = X_obs.copy()
    X_sig[nan_sig_mask] = X_obs[nan_sig_mask] / 2
    X_obs[nan_sig_mask] = 0.00
    
    # Metallicity constraint
    arr_metal = np.array([-2.1522884, -1.6751671, -1.1522883, -0.6751671, -0.3741371, -0.15228835])
    if X_obs[0] < arr_metal[0]:
        metal_VALID = np.array([True, True, True, False, False, False])
    else:
        metal_VALID = (np.abs(arr_metal - X_obs[0]) < 4.0 * X_sig[0])
    
    new_VALID = metal_VALID[:, None, None, None, None] * VALID
    
    # Define spacing
    new_spacing = 1
    subsample_MODEL = MODEL[:,:,:,:,::new_spacing,:]
    subsample_VALID = new_VALID[:,:,:,:,::new_spacing]
    subsample_THETA = THETA[:,:,:,:,::new_spacing]
    
    subsample_Del_Z = Del_Z[:,:,:,:,::new_spacing]
    subsample_Del_magb = Del_magb[:,:,:,:,::new_spacing]
    subsample_Del_mi = Del_mi[:,:,:,:,::new_spacing]
    subsample_Del_mf = Del_mf[:,:,:,:,::new_spacing]
    
    # Create weights array
    weights = np.ones(len(X_obs))
    weights[:3] = 3./non_nan_obs_length
    
    for i, w in enumerate([1.5, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0]):
        if 4+i < len(weights):
            weights[4+i] = w
    
    #print('Computing chi-squared directly using optimized Numba...')
    
    # Prepare arrays for Numba
    X_obs_array = np.ascontiguousarray(X_obs, dtype=np.float32)
    X_sig_weights = np.ascontiguousarray(X_sig * weights, dtype=np.float32)
    model_data = np.ascontiguousarray(subsample_MODEL, dtype=np.float32)
    valid_mask = np.ascontiguousarray(subsample_VALID, dtype=bool)
    
    # Compute chi-squared directly without intermediate arrays
    chi_sq = compute_residuals_and_sum_numba(model_data, X_obs_array, X_sig_weights, valid_mask)
    
    # Save results if requested
    if save_chis:
        output_dir = '/nexus/posix0/MIA-astro-env/hxr/adimoff/BinaryStars/Accretion_New/chi_sq_info_numba_FULL_GRID/'
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'{star_name}_rough_chis.npy'), 'wb') as chisq_handler:
            np.save(chisq_handler, chi_sq)
    
    # Find best indices
    best_inds_all = np.array(np.unravel_index(np.argsort(chi_sq.ravel()), chi_sq.shape)).T
    #print('Best track and model via chi sq', best_inds_all[0])
    
    #print('Compute likelihood and probability value')
    chi_sq_min = np.nanmin(chi_sq)
    log_like = -(chi_sq - chi_sq_min)
    like = np.exp(log_like)
    like_sum = np.nansum(like)
    like_norm = like / like_sum if like_sum > 0 else like
    
    # Combine with priors
    log_THETA = np.log(subsample_THETA)
    log_THETA_sum = np.log(np.nansum(subsample_THETA))
    prob = like_norm * np.exp(log_THETA - log_THETA_sum)
    
    # OPTIMIZED: Pre-filter and limit the search space
    #print('Extracting best models efficiently...')
    
    # Find finite chi-squared values first
    finite_mask = np.isfinite(chi_sq)
    finite_prob_mask = np.isfinite(prob) & (prob > 1e-20)  # Higher threshold
    
    # Combine masks for valid entries
    valid_entries_mask = finite_mask & finite_prob_mask & subsample_VALID
    
    # If no valid entries, return empty results quickly
    if not np.any(valid_entries_mask):
        print(f"No valid models found for {star_name}")
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Get indices of valid entries
    valid_indices = np.where(valid_entries_mask)
    valid_chi_values = chi_sq[valid_indices]
    valid_prob_values = prob[valid_indices]
    
    # Sort by chi-squared (limit to top 5000 to avoid memory issues)
    max_models = min(5000, len(valid_chi_values))
    sort_indices = np.argsort(valid_chi_values)[:max_models]
    
    # Extract sorted results efficiently
    sorted_indices = tuple(vi[sort_indices] for vi in valid_indices)
    
    models = np.array([sorted_indices[4][i] * new_spacing for i in range(len(sort_indices))])
    tracks = np.column_stack([sorted_indices[i] for i in range(4)])
    chis = valid_chi_values[sort_indices]
    probs = valid_prob_values[sort_indices]
    
    #print(f"Found {len(models)} valid models for {star_name}")
   
    # Fill arrays with valid results
    valid_count = 0
    for i, ind in enumerate(best_inds_all):
        if valid_count >= max_models:
            break
            
        ind_tuple = tuple(ind)
        if prob[ind_tuple] > 0.0 and chi_sq[ind_tuple] < np.inf:
            probs[valid_count] = prob[ind_tuple]
            chis[valid_count] = chi_sq[ind_tuple]
            models[valid_count] = ind[-1] * new_spacing
            tracks[valid_count, :] = ind[:-1]
            valid_count += 1
    
    # Trim arrays to actual valid count
    probs = probs[:valid_count]
    chis = chis[:valid_count]
    models = models[:valid_count]
    tracks = tracks[:valid_count]
    
    # sort by probability
    sorted_inds = np.argsort(-probs)
    probs = probs[sorted_inds]
    chis = chis[sorted_inds]
    models = models[sorted_inds]
    tracks = tracks[sorted_inds]
    
    # write out best tracks and model numbers to a file
    tracks_and_mods = np.zeros((len(models),5),dtype=int)
    tracks_and_mods[:,0:4] = tracks
    tracks_and_mods[:,4] = models

    if save_models == True:
        np.savetxt('models_full_numba_FULL_GRID/best_tracks_and_models_'+star_name+'_numba.dat', tracks_and_mods, fmt='%i')

    # always write out histogram data to the file
    #print('Writing out best models for plotting histograms')
    hist_out_file = 'master_model_hist_list_numba_FULL_GRID.dat'
    # Write top 3 models to histogram file
    with open(hist_out_file,'a+') as hist_file:
        hist_file.write(str(star_name+'  0.'+unique[0][tracks[0][0]]+' '+unique[1][tracks[0][1]][0]+'.'+unique[1][tracks[0][1]][-1]+' '+unique[2][tracks[0][2]]+' '+unique[3][tracks[0][3]])+' '+str(X_obs[0])+' '+type_flag+'\n')
        hist_file.write(str(star_name+'  0.'+unique[0][tracks[1][0]]+' '+unique[1][tracks[1][1]][0]+'.'+unique[1][tracks[1][1]][-1]+' '+unique[2][tracks[1][2]]+' '+unique[3][tracks[1][3]])+' '+str(X_obs[0])+' '+type_flag+'\n')
        hist_file.write(str(star_name+'  0.'+unique[0][tracks[2][0]]+' '+unique[1][tracks[2][1]][0]+'.'+unique[1][tracks[2][1]][-1]+' '+unique[2][tracks[2][2]]+' '+unique[3][tracks[2][3]])+' '+str(X_obs[0])+' '+type_flag+'\n')

    # Compute marginal probabilities
    prob_Z = np.sum(np.sum(np.sum(np.sum(like, axis=4), axis=3), axis=2), axis=1)
    prob_Z = prob_Z * subsample_Del_Z[:,0,0,0,0] / np.sum(subsample_Del_Z)
    
    prob_magb = np.sum(np.sum(np.sum(np.sum(like, axis=4), axis=3), axis=2), axis=0)
    prob_magb = prob_magb * subsample_Del_magb[0,:,0,0,0] / np.sum(subsample_Del_magb)
    
    prob_mi = np.sum(np.sum(np.sum(np.sum(like, axis=4), axis=3), axis=1), axis=0)
    prob_mi = prob_mi * subsample_Del_mi[0,0,:,0,0] / np.sum(subsample_Del_mi)
    
    prob_mf = np.sum(np.sum(np.sum(np.sum(like, axis=4), axis=2), axis=1), axis=0)
    prob_mf = prob_mf * subsample_Del_mf[0,0,0,:,0] / np.sum(subsample_Del_mf)
    
    # Calculate statistics
    metal_stats = get_stats(metal_Z, prob_Z)
    agb_stats = get_stats(agb_masses, prob_magb)
    mi_stats = get_stats(initial_masses, prob_mi)
    mf_stats = get_stats(final_masses, prob_mf)
    

    # Save parameter statistics
    #if save_stats:
    #    write_parameter_stats(star_name, metal_Z, agb_masses, initial_masses, final_masses,
    #                     prob_Z, prob_magb, prob_mi, prob_mf,
    #                     metal_stats, agb_stats, mi_stats, mf_stats)

    if save_stats:
        # Build best_model_params dictionary from accepted models for derived quantities
        # tracks shape: (n_accepted, 4) where each row is (z_idx, agb_idx, track_idx, final_idx)
        # models shape: (n_accepted,) with initial mass grid indices
        best_model_params = {
            'metallicity': np.array([metal_Z[tracks[i, 0]] for i in range(len(tracks))]),
            'agb_mass': np.array([agb_masses[tracks[i, 1]] for i in range(len(tracks))]),
            'initial_mass': np.array([initial_masses[tracks[i, 2]] for i in range(len(tracks))]),
            'final_mass': np.array([final_masses[tracks[i, 3]] for i in range(len(tracks))]),
            'probabilities': probs
        }

        # Create the output directory if it doesn't exist
        os.makedirs(os.path.join(original_data_dir, 'statistics_numba_FULL_GRID'), exist_ok=True)

        write_parameter_stats(star_name, metal_Z, agb_masses, initial_masses, final_masses,
                         prob_Z, prob_magb, prob_mi, prob_mf,
                         metal_stats, agb_stats, mi_stats, mf_stats,
                         best_model_params=best_model_params)
    
    end_time = time.time()
    print(f"Numba function completed in {end_time - start_time:.2f} seconds")
    
    return models, tracks, chis, probs

def process_star_with_progress(args):
    """Star processing with explicit progress reporting"""
    star_index, warmup_done, progress_queue = args
    
    if not warmup_done:
        warmup_numba_functions()
    
    start_time = time.time()
    star_name = master_table['stars'][star_index]
    
    try:
        # Send progress update at start
        if progress_queue:
            progress_queue.put(f"Starting {star_name}")
        
        models, tracks, chis, probs = compute_chi_sq_numba(star_index, save_chis=False, save_models=True, save_stats=True)
        
        processing_time = time.time() - start_time
        
        result = {
            'star_name': star_name,
            'star_index': star_index,
            'processing_time': processing_time,
            'status': 'success',
            'num_models': len(models),
            'best_chi': chis[0] if len(chis) > 0 else None
        }
        
        # Send completion update
        if progress_queue:
            progress_queue.put(f"Completed {star_name} in {processing_time:.1f}s")
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        result = {
            'star_name': star_name,
            'star_index': star_index,
            'processing_time': processing_time,
            'status': 'failed',
            'error': str(e)
        }
        
        if progress_queue:
            progress_queue.put(f"Failed {star_name}: {str(e)}")
        
        return result


def process_all_stars_parallel_with_progress(num_processes=None):
    """Parallel processing with proper progress tracking"""
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)
    
    print(f"Processing {len(master_table)} stars using {num_processes} processes")
    
    # Warm up Numba functions once
    warmup_numba_functions()
    
    # Create a manager for progress communication
    manager = mp.Manager()
    progress_queue = manager.Queue()
    
    # Create arguments with progress queue
    star_args = [(i, True, progress_queue) for i in range(len(master_table))]
    
    # Process in smaller chunks for better progress tracking
    chunk_size = 15  # Smaller chunks for more frequent updates
    all_results = []
    
    total_stars = len(master_table)
    processed_count = 0
    
    # Create overall progress bar
    overall_pbar = tqdm(total=total_stars, desc="Overall Progress", position=0)
    
    for chunk_start in range(0, len(star_args), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(star_args))
        chunk_args = star_args[chunk_start:chunk_end]
        
        chunk_num = chunk_start // chunk_size + 1
        print(f"\nProcessing chunk {chunk_num}: stars {chunk_start} to {chunk_end-1}")
        
        # Create chunk progress bar
        chunk_pbar = tqdm(total=len(chunk_args), 
                         desc=f"Chunk {chunk_num}", 
                         position=1, 
                         leave=False)
        
        # Start the multiprocessing pool
        ctx = mp.get_context('fork')
        with ctx.Pool(processes=num_processes) as pool:
            # Start async processing
            async_result = pool.map_async(process_star_with_progress, chunk_args)
            
            # Monitor progress while processing
            chunk_completed = 0
            while not async_result.ready():
                try:
                    # Check for progress updates (non-blocking)
                    while not progress_queue.empty():
                        message = progress_queue.get_nowait()
                        if "Completed" in message or "Failed" in message:
                            chunk_completed += 1
                            chunk_pbar.update(1)
                            overall_pbar.update(1)
                            overall_pbar.set_postfix_str(message.split()[1])  # Show star name
                        
                    time.sleep(0.1)  # Small delay to prevent busy waiting
                    
                except:
                    pass
            
            # Get results
            chunk_results = async_result.get()
            
            # Update progress bars for any remaining items
            remaining = len(chunk_args) - chunk_completed
            if remaining > 0:
                chunk_pbar.update(remaining)
                overall_pbar.update(remaining)
        
        chunk_pbar.close()
        all_results.extend(chunk_results)
        
        # Process any remaining progress messages
        while not progress_queue.empty():
            try:
                message = progress_queue.get_nowait()
            except:
                break
        
        # Force garbage collection between chunks
        import gc
        gc.collect()
    
    overall_pbar.close()
    
    # Summarize results
    successful = [r for r in all_results if r['status'] == 'success']
    failed = [r for r in all_results if r['status'] == 'failed']
    
    print(f"\n\nProcessing Summary:")
    print(f"- Successfully processed: {len(successful)} stars")
    print(f"- Failed: {len(failed)} stars")
    
    if successful:
        avg_time = sum(r['processing_time'] for r in successful) / len(successful)
        print(f"- Average processing time per star: {avg_time:.2f} seconds")
        print(f"- Total processing time: {sum(r['processing_time'] for r in successful):.2f} seconds")
    return all_results

if __name__ == '__main__':
    ### run the parallelized process on the full master_table sample
    all_results = process_all_stars_parallel_with_progress(num_processes=15)

#####
# MxN - OxN = MxNxO
# sum over N gives NxO
# 
# numpy einsum ?
# 