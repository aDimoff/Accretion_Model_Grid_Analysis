print('Comparing observed data to model grid')
print('(( currently parallelized, one star per core on MPCDF ))')
print('(( memory allows 30  at a time...                     ))')
#print('(( try a distributed memory implemenetation?          ))')

import numpy as np
import pandas as pd
import re
import os
#import matplotlib.pyplot as plt
from astropy.table import Table, join, vstack
from astropy.io import fits, ascii
import pickle
from PyAstronomy import pyasl
from time import sleep
from tqdm.notebook import tqdm
import bottleneck as bn
from multiprocessing import Pool, shared_memory
import warnings
warnings.filterwarnings("ignore")

#!rm master_model_hist_list_0.dat
#touch master_model_hist_list_0.dat

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
    AGB_mass = model_name.split('_')[-2][3:] #(model_name.split('_')[-2][3:4]+'.'+model_name.split('_')[-2][5:6])
    init_mass = (model_name.split('_')[0][1]+'.'+model_name.split('_')[0][3:5])
    final_mass = (model_name.split('_')[1][1]+'.'+model_name.split('_')[1][3:5])
    #if AGB_mass < init_mass: 
    #    return None
    return metalZ,AGB_mass,init_mass,final_mass
#------------------------------------------------------------------------------------------------------------
# read the list of surface files and break up thier names into the different evo track parameters
list_surf_files = np.genfromtxt('list_surface_files',dtype=str)
# generate indicies for each of the parameters
split = [split_surf_file(sf) for sf in list_surf_files if split_surf_file(sf) is not None]
# select unique indicies to define each track
unique = [list(np.unique(spl)) for spl in np.array(split).T]
#print('unique',unique)
#------------------------------------------------------------------------------------------------------------
def create_delta(input_list):
    delta_list = []
    for i in range(len(input_list)-1):
        diff = (float(input_list[i+1]) - float(input_list[i])) / 2
        #print(i, diff)
        delta_list.append(diff)
    delta_list.insert(0,delta_list[0])

    return np.array(delta_list)
#------------------------------------------------------------------------------------------------------------
# convert unique values to floats to compute distances between grid points
# convert Z values to floats... 0.0001...
metal_Z = []
for z in unique[0]:
    capZ = np.log10(float('0.'+z)/0.0142)
    metal_Z.append(capZ)
metal_Z = np.array(metal_Z)
#metal_Z
# convert agb values to floats... 1p3 = 1.3
agb_masses = []
for m_agb in unique[1]:
    value = float(m_agb[0] + '.' + m_agb[-1])
    agb_masses.append(value)
agb_masses = np.array(agb_masses)
#agb_masses
# convert initial and final masses to floats from the unique array
initial_masses = []
for m_initial in unique[2]:
    value = float(m_initial)
    initial_masses.append(value)
initial_masses = np.array(initial_masses)
#initial_masses
final_masses = []
for m_final in unique[3]:
    value = float(m_final)
    final_masses.append(value)
final_masses = np.array(final_masses)
#final_masses
# create midpoint arrays / delta arrays for the grid...
delta_metal_Z = create_delta(metal_Z)
delta_m_agb = create_delta(agb_masses)
delta_m_init = create_delta(initial_masses)
delta_m_final = create_delta(final_masses)
delta_time = np.ones(30000) # dummy array, not used in the model grid

#------------------------------------------------------------------------------------------------------------
# establish data directory on this machine
base_dir = '/nexus/posix0/MIA-astro-env/hxr/adimoff/'
data_dir = 'BinaryStars/Accretion_New/'
print('accessing directories:')
print(base_dir)
print(data_dir)
# establish output file naming convention for this run
# out_file_tag = ''

print('Reading MODEL grid from file: "model_grid_big_file.dat" and VALID file "valid_grid_big_file.dat"')
#with open(data_dir + BinaryStars/Accretion_New/model_grid_big_file.dat','rb') as file:
#model_file = data_dir + BinaryStars/Accretion_New/model_grid_big_file.dat'
MODEL = np.load(base_dir + data_dir + 'model_grid_big_file_mass_limit_full.dat')
VALID = np.load(base_dir + data_dir + 'valid_grid_big_file_mass_limit_full.dat')
# generate priors based on size of grid cells
print('Generating priors based on relative size of grid cells')
Del_Z, Del_magb, Del_mi, Del_mf, Del_t = np.meshgrid(delta_metal_Z,delta_m_agb,delta_m_init,delta_m_final,delta_time, indexing="ij")
top_prod = Del_magb * Del_mi * VALID
THETA = Del_Z/np.sum(Del_Z) * top_prod/(np.sum(top_prod)) * Del_mf/np.sum(Del_mf) * Del_t/np.sum(Del_t) # divide by sums for proper priors
print('MODEL, VALID, THETA shapes:')
print(np.shape(MODEL), np.shape(VALID), np.shape(THETA))

print('Reading in observational data')
# read abundance infomation, parameters and mass info for my sample...
Obs_Data_Table = ascii.read(base_dir + 'abund_data/X_Fe_table_3.dat',delimiter='&')
param_mass_table = ascii.read(base_dir + 'abund_data/param_mass_table.dat',format='latex')
full_obs_data_table = join(param_mass_table,Obs_Data_Table,keys='stars')
# read in literature data for CEMP / CH / Ba stars
cristallo_data = ascii.read(base_dir + 'abund_data/Cristallo_2016.dat')
goswami_2020_data = ascii.read(base_dir + 'abund_data/Gos_Rat_Gos_2020.dat')
goswami_2021_data = ascii.read(base_dir + 'abund_data/Gos_Rat_Gos_CEMPs_2021.dat')
DeCastro_Roriz = ascii.read(base_dir + 'abund_data/DeCastro_Roriz_Table.dat')

# test for Borbala on one of her stars...
HD49641_data = ascii.read(base_dir + 'abund_data/HD49641_Cseh.dat')

# read in big fat SAGA database data... trim this to just CEMP and CH stars... metal poor 
#SAGA_MP_table = ascii.read(base_dir + 'abund_data/SAGA_metalpoor_heavies_APTable.dat')
#SAGA_MR_table = ascii.read(base_dir + 'abund_data/SAGA_metalrich_heavies_APTable.dat')

# stack them all up in the master table
# skip SAGA for now, too many of them
#master_table = vstack([full_obs_data_table,cristallo_data,goswami_2020_data,goswami_2021_data,SAGA_MP_table,DeCastro_Roriz,SAGA_MR_table],join_type='inner')
master_table = vstack([full_obs_data_table,cristallo_data,goswami_2020_data,goswami_2021_data,DeCastro_Roriz,HD49641_data],join_type='inner')
print('master table is ',len(master_table),' items long\n')

obs_names = np.array(['C','C_sig','FeH','FeH_sig','Sr','Sr_sig','Y','Y_sig','Zr','Zr_sig','Nb','Nb_sig','Mo','Mo_sig','Ru','Ru_sig','Ba','Ba_sig','La','La_sig','Ce','Ce_sig','Pr','Pr_sig','Nd','Nd_sig','Sm','Sm_sig','Eu','Eu_sig','Dy','Dy_sig','Pb','Pb_sig'])
star_indices = np.arange(len(master_table))
# ------------------------------------------------------------------------------------------------------------
def get_star_params(table_in_question, star_index):
    star_name = table_in_question['stars'][star_index]
    
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
    gstar = 10**logg; e_gstar = 10**(logg-e_logg)
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
                     table_in_question['Zr'][star_index]])#,table_in_question['Nb'][star_index], 
                     #table_in_question['Mo'][star_index],table_in_question['Ru'][star_index]])
    hs = bn.nanmean([table_in_question['Ba'][star_index],table_in_question['La'][star_index],
                     table_in_question['Ce'][star_index],table_in_question['Nd'][star_index]])    

    C_enhanced = table_in_question['C'][star_index] > 0.50 # or 0.70? or 0.50?
    Metal_Poor = table_in_question['FeH'][star_index] < -1.00
    dash_s = (hs - table_in_question['Eu'][star_index] > 0.50) and hs > 0.90 ## Beers + Christlieb 2005 / Jorissen 2016
    dash_rs = (hs - table_in_question['Eu'][star_index] < 0.50) and (hs - table_in_question['Eu'][star_index] > 0.10) ## Beers + Christlieb 2005 / Jorissen 2016
    dash_r = table_in_question['Eu'][star_index] > 1.00 and hs < 0.0 ## Abate 2016

    ## CAMILLA'S CRITERION BASED ON SR AND BA -- LS AND HS
    #CEMP-no [Sr/Ba] > 0.75 New classification
    if C_enhanced and Metal_Poor and (ls - hs > 0.75): # Hansen 2019
        flag = 'CEMP-no'
    elif C_enhanced and Metal_Poor and hs < 0.0: # Beers + Christlieb 2005 / Jorissen 2016
        flag = 'CEMP-no'
    # THE ORDER OF THESE MATTERS...
    elif C_enhanced and Metal_Poor and (ls < -1.5): # Hansen 2019
        flag = 'CEMP-r'
    elif C_enhanced and Metal_Poor and dash_r: 
        flag = 'CEMP-r'
    elif C_enhanced and Metal_Poor and dash_rs:
        flag = 'CEMP-r/s'
    elif C_enhanced and Metal_Poor and (ls - hs > -0.60) and (ls - hs < 0.60): # Hansen 2019
        flag = 'CEMP-s'
    elif C_enhanced and Metal_Poor and (dash_s or (table_in_question['Ba'][star_index] > 1.0 and table_in_question['Ba'][star_index] < 2.10)):
        flag = 'CEMP-s'
    ## add flags for CH stars as those at higher metallicities but share same characteristics
    elif C_enhanced: #and (ls - hs > -0.60) and (ls - hs < 0.60):
        flag = 'CH'
    #elif Metal_Poor:
    #    flag = 'CH'
    # Escorza's strong / weak Ba star criterion:
    elif not Metal_Poor and (ls > 0.2 or hs > 0.2 and not C_enhanced):
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
            break
        # Barium stars, avg mass is ~2.50
        elif flag == 'WBa' or flag == 'SBa':
            m_star = np.nan # = 2.50
            e_m_star = np.nan # = 0.50
        # CEMP stars, avg mass is ~0.80
        elif flag == 'CEMP-s' or flag == 'CEMP-r/s' or flag == 'CH':
            m_star = np.nan # = 0.80
            e_m_star = np.nan # = 0.50
        # if not, give it a nan
        else:
            m_star = np.nan #1.50
            e_m_star = np.nan #1.50
    
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
#------------------------------------------------------------------------------------------------------------
# Calculate statistics for each parameter
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
#------------------------------------------------------------------------------------------------------------
def write_parameter_stats(star_name, metal_Z, agb_masses, initial_masses, final_masses, 
                         prob_Z, prob_magb, prob_mi, prob_mf,
                         metal_stats, agb_stats, mi_stats, mf_stats):
    
    stats_dir = '/nexus/posix0/MIA-astro-env/hxr/adimoff/BinaryStars/Accretion_New/statistics_weights_FULL_GRID/'
    
    # Create output filename in statistics directory
    outfile = os.path.join(stats_dir, f'parameter_stats_{star_name}.txt')
        
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
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
# Compute chi-squared for each star in the master table
# This function will be run in parallel for each star
# It computes the chi-squared value for each star based on the model grid and observational data
# It also saves the chi-squared values to a file for later analysis
#------------------------------------------------------------------------------------------------------------
def compute_chi_sq(star_index):
    
    star_name, obs_names_II, obs_elems_nums, obs_elems_II, obs_errrs_II, X_obs, X_sig, type_flag = get_star_params(master_table, star_index)
    print('computing...',f'{star_index:4}',star_name)
    
    len_nans = 0
    for obs in X_obs:
        if np.isnan(obs):
            #print('nan')
            len_nans += 1
    #print(len_nans)
    non_nan_obs_length = len(X_obs) - len_nans
    
    # change nans and abundances for upper limits
    for i,(sig,data) in enumerate(zip(X_sig,X_obs)):
        #print(i,data,sig)
        if (np.isnan(sig) and not(np.isnan(data))):
            #print(i,data,sig)
            X_sig[i] = X_obs[i] / 2
            X_obs[i] = 0.00
            
    ## constrain by metallicity...
    ## limit the searched parameter space by metallicities 'close' to the observations
    arr_metal = [-2.1522884,  -1.6751671,  -1.1522883,  -0.6751671,  -0.3741371,  -0.15228835] #unique[0] had problems, do this by hand...   
    if X_obs[0] < -2.15:
        # if metallicity is below our lower limit, only allow the most metal poor models
        metal_VALID = np.array([True, False, False, False, False, False])
    else:
        metal_VALID = (np.abs(arr_metal - X_obs[0]) < 4.0*X_sig[0])
    # reset valid array with compatible metallicities
    new_VALID = metal_VALID[:, None, None, None, None] * VALID

    #print('Coarse search within the grid')
    # sub sample of the model array for speed
    new_spacing = 5
    subsample_MODEL = MODEL[:,:,:,:,::new_spacing,:]
    subsample_VALID = new_VALID[:,:,:,:,::new_spacing]
    subsample_THETA = THETA[:,:,:,:,::new_spacing]   
    subsample_Del_Z = Del_Z[:,:,:,:,::new_spacing]
    subsample_Del_magb = Del_magb[:,:,:,:,::new_spacing]
    subsample_Del_mi = Del_mi[:,:,:,:,::new_spacing]
    subsample_Del_mf = Del_mf[:,:,:,:,::new_spacing]

    ## apply weights to the surface parameters such that they are equally important as the abundances. (3. / length of abundance list)
    ## and to the abundances - the s-process dominant elements are more important...
    ## decrease the dependence on Nb. (2.0)
    ## decrease the dependence on Ru. (2.0)
    ## decrease the dependence on Dy. (2.0) - r-process contributions...
    weights = [3./non_nan_obs_length,3./non_nan_obs_length,3./non_nan_obs_length,1.0,
               1.5,1.0,1.0,1.0,2.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0]
    
    #print('Compute squared residuals between observational data and grid of models')
    residuals_sq = np.square((subsample_MODEL - X_obs) / (X_sig*weights))
    #print(np.shape(residuals_sq))

    #print('Compute chi_squared value by summing up all residuals')
    chi_sq = bn.nansum(residuals_sq,axis=-1) ## use bottleneck nansum here for speed, np.nansum() is slower
    chi_sq[np.invert(subsample_VALID)] = np.inf ## apply infinities to correspond to valid array

    #print('writing chi_squareds to file...')
    with open(base_dir + data_dir + 'chis_full_weights_stats/'+str(star_name)+'_weights_stats.npy','wb') as chisq_handler:
        np.save(chisq_handler,chi_sq)
    
    #print('Compute likelihood and probability value')
    peaked_param = 10.0 # tuneable parameter for peakedness of the likelihood function
    chi_sq_normalized = chi_sq - np.nanmin(chi_sq)
    log_like = -chi_sq_normalized / peaked_param ## compute log likelihood from chi sq
    #max_log_like = np.nanmax(log_like)
    like = np.exp(log_like) # - max_log_like) ## compute likelihood from log likelihood
    #print(np.shape(like))
    like_norm = like / np.nansum(like) ## normalize likelihood to sum to 1

    # combine likelihood with priors
    #prob = bn.nansum(like) * subsample_THETA / np.sum(subsample_THETA)
    log_THETA = np.log(subsample_THETA)
    prob = like_norm * np.exp(log_THETA - np.log(np.nansum(subsample_THETA))) ## compute probability from likelihood and priors

    # sort best indices by likelihood
    best_inds_all_prob = np.array(np.unravel_index(np.argsort(-prob,axis=None),prob.shape,order='C')).T

    # compute probabilities for each parameter :: Marginalize over all other parameters to get Z probability distribution
    prob_Z    = np.sum(np.sum(np.sum(np.sum(like, axis=4), axis=3), axis=2), axis=1) * subsample_Del_Z[:,0,0,0,0] / np.sum(subsample_Del_Z)
    prob_magb = np.sum(np.sum(np.sum(np.sum(like, axis=4), axis=3), axis=2), axis=0) * subsample_Del_magb[0,:,0,0,0] / np.sum(subsample_Del_magb)
    prob_mi   = np.sum(np.sum(np.sum(np.sum(like, axis=4), axis=3), axis=1), axis=0) * subsample_Del_mi[0,0,:,0,0] / np.sum(subsample_Del_mi)
    prob_mf   = np.sum(np.sum(np.sum(np.sum(like, axis=4), axis=2), axis=1), axis=0) * subsample_Del_mf[0,0,0,:,0] / np.sum(subsample_Del_mf)

    # Calculate statistics for each parameter
    metal_stats = get_stats(metal_Z, prob_Z)
    agb_stats = get_stats(agb_masses, prob_magb)
    mi_stats = get_stats(initial_masses, prob_mi)
    mf_stats = get_stats(final_masses, prob_mf)
    
    #print('Writing out parameter statistics for the star')
    write_parameter_stats(star_name, metal_Z, agb_masses, initial_masses, final_masses,
                     prob_Z, prob_magb, prob_mi, prob_mf,
                     metal_stats, agb_stats, mi_stats, mf_stats)

    #print('Zoom in on best model in rough sampling')
    # sort indices of best fitting models
    best_inds_all = np.array(np.unravel_index(np.argsort(chi_sq,axis=None),chi_sq.shape,order='C')).T
    bit_list = []; best_chis = []; reduced_chis = []; best_probs = []; best_models = []; best_tracks = []
    zoom_spacing = 500
    i = 0 ; Num_tracks = 3
    while len(best_models) < Num_tracks:
        #print(i)
        # make bitmap numbers to keep track of which models have been used already
        #bitmap_number = np.sum([best_inds_all[i][0]**0, best_inds_all[i][1]**1, best_inds_all[i][2]**2, best_inds_all[i][3]**3]) ## or the other way 'round? this works.
        bitmap_number_prob = np.sum([best_inds_all_prob[i][0]**0, best_inds_all_prob[i][1]**1, best_inds_all_prob[i][2]**2, best_inds_all_prob[i][3]**3]) ## or the other way 'round...

        # if a given model already has a 'best fit' at one of the time steps, move to the next one. 
        # this is to make sure no two evo-tracks have 'best fit' models; they would be right next to each other.
        if bitmap_number_prob in bit_list:
            i += 1
            continue
        else:
            # zoom in on the model
            zoomed_MODEL = MODEL[best_inds_all_prob[i][0],best_inds_all_prob[i][1],best_inds_all_prob[i][2],best_inds_all_prob[i][3],best_inds_all_prob[i][4]*new_spacing-zoom_spacing:best_inds_all_prob[i][4]*new_spacing+zoom_spacing,:]
            zoomed_VALID = VALID[best_inds_all_prob[i][0],best_inds_all_prob[i][1],best_inds_all_prob[i][2],best_inds_all_prob[i][3],best_inds_all_prob[i][4]*new_spacing-zoom_spacing:best_inds_all_prob[i][4]*new_spacing+zoom_spacing]
            zoomed_THETA = THETA[best_inds_all_prob[i][0],best_inds_all_prob[i][1],best_inds_all_prob[i][2],best_inds_all_prob[i][3],best_inds_all_prob[i][4]*new_spacing-zoom_spacing:best_inds_all_prob[i][4]*new_spacing+zoom_spacing]
            # compute residuals and chi_sq value on zoom in section
            residuals_sq_zoom = np.square((zoomed_MODEL - X_obs) / X_sig*weights)
            chi_sq_zoom = bn.nansum(residuals_sq_zoom,axis=-1)
            chi_sq_zoom[np.invert(zoomed_VALID)] = np.inf
            #
            with open(base_dir + data_dir + 'chis_full_mass_limit_weights_stats/'+str(star_name)+'_'+str(bitmap_number_prob)+'_mass_limit_weights_stats.npy','wb') as chisq_handler_zoom:
                np.save(chisq_handler_zoom,chi_sq_zoom)

            best_chi = chi_sq_zoom.min()
            reduced_chi = chi_sq_zoom.min() / non_nan_obs_length

            chi_sq_zoom_normalized = chi_sq_zoom - np.nanmin(chi_sq_zoom)
            like_zoom = np.exp(-chi_sq_zoom_normalized / peaked_param) # - max_log_like) ## compute likelihood from log likelihood
            #print(np.shape(like_zoom))
            like_zoom_norm = like_zoom / np.nansum(like_zoom) ## normalize likelihood to sum to 1
            log_THETA_zoom = np.log(zoomed_THETA)
            prob_zoom = like_zoom_norm * np.exp(log_THETA_zoom - np.log(np.nansum(zoomed_THETA))) # bn.nansum(like_zoom) * zoomed_THETA / np.sum(zoomed_THETA) 
            #print(np.shape(prob_zoom))
            best_prob = np.nanmax(prob_zoom,axis=None) ## find the best probability value

            #best_inds_all_zoom = np.array(np.unravel_index(np.argsort(chi_sq_zoom,axis=None),chi_sq_zoom.shape,order='C')).T
            #best_inds_all_master = [best_inds_all[i][0],best_inds_all[i][1],best_inds_all[i][2],best_inds_all[i][3],best_inds_all[i][4]*new_spacing - new_spacing*zoom_spacing+best_inds_all_zoom[i][0]]
            best_inds_all_master = [best_inds_all_prob[i][0],best_inds_all_prob[i][1],best_inds_all_prob[i][2],best_inds_all_prob[i][3],best_inds_all_prob[i][4]*new_spacing] #???

        # append them all to the lists
        best_probs.append(best_prob)
        best_chis.append(best_chi)
        reduced_chis.append(reduced_chi)
        bit_list.append(bitmap_number_prob)
        best_models.append(best_inds_all_master)
        best_tracks.append(best_inds_all_master[:-1])
        i += 1
    
    best_probs,best_chis,best_models,best_tracks = zip(*sorted(zip(best_probs,best_chis,best_models,best_tracks),reverse=True))
    
    #print('Writing out best models for plotting tracks')
    with open(base_dir + data_dir + 'models_full_mass_limit_weights_stats/best_models_'+star_name+'_mass_limit_weights_stats.dat','wb') as mod_file:
        np.save(mod_file,best_models)  
    #print('Writing out best chis')
    with open(base_dir + data_dir + 'chis_full_mass_limit_weights_stats/best_chis_'+star_name+'_mass_limit_weights_stats.dat','wb') as chi_file:
        np.save(chi_file,best_chis)
    #print('Writing out best probs')
    with open(base_dir + data_dir + 'probs_full_mass_limit_weights_stats/best_probs_'+star_name+'_mass_limit_weights_stats.dat','wb') as prob_file:
        np.save(prob_file,best_probs)

    #print('Writing out best models for plotting histograms')
    with open('master_model_hist_list_mass_limit_weights_stats.dat','a+') as hist_file:
        hist_file.write(str(star_name+'  0.'+unique[0][best_tracks[0][0]]+' '+unique[1][best_tracks[0][1]][0]+'.'+unique[1][best_tracks[0][1]][-1]+' '+unique[2][best_tracks[0][2]]+' '+unique[3][best_tracks[0][3]])+' '+str(X_obs[0])+'  '+str(best_chis[0])+' '+str(reduced_chis[0])+' '+type_flag+'\n')
        hist_file.write(str(star_name+'  0.'+unique[0][best_tracks[1][0]]+' '+unique[1][best_tracks[1][1]][0]+'.'+unique[1][best_tracks[1][1]][-1]+' '+unique[2][best_tracks[1][2]]+' '+unique[3][best_tracks[1][3]])+' '+str(X_obs[0])+'  '+str(best_chis[1])+' '+str(reduced_chis[1])+' '+type_flag+'\n')
        hist_file.write(str(star_name+'  0.'+unique[0][best_tracks[2][0]]+' '+unique[1][best_tracks[2][1]][0]+'.'+unique[1][best_tracks[2][1]][-1]+' '+unique[2][best_tracks[2][2]]+' '+unique[3][best_tracks[2][3]])+' '+str(X_obs[0])+'  '+str(best_chis[2])+' '+str(reduced_chis[2])+' '+type_flag+'\n')

    return
#------------------------------------------------------------------------------------------------------------
# set number of cpus across which to parallelize
num_cpus = 30 #multiprocessing.cpu_count()
# use pool and num_cpus to speed up and do multiple stars at once
#res_ult = Pool(min(num_cpus,len(star_indices))).map(compute_chi_sq,range(len(star_indices)))

from contextlib import contextmanager

@contextmanager
def poolcontext(*args, **kwargs):
    pool = Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    pool.join()

# Process in batches
batch_size = 30
for i in range(0, len(star_indices), batch_size):
    batch = range(i, min(i + batch_size, len(star_indices)))
    with poolcontext(processes=min(num_cpus, len(batch))) as pool:
        try:
            results = pool.map(compute_chi_sq, batch)
            print(f"Completed batch {i//batch_size + 1}, processed stars {i} to {i+len(batch)}")
        except Exception as e:
            print(f"Error in batch starting at index {i}: {str(e)}")

#from tqdm import tqdm

#def process_batch(batch):
#    with poolcontext(processes=min(num_cpus, len(batch))) as pool:
#        for _ in tqdm(pool.imap_unordered(compute_chi_sq, batch), 
#                     total=len(batch), 
#                     desc=f"Processing batch"):
#            pass

# Process in batches with progress bar
#for i in range(0, len(star_indices), batch_size):
#    batch = range(i, min(i + batch_size, len(star_indices)))
#    process_batch(batch)

print('End of Line')
