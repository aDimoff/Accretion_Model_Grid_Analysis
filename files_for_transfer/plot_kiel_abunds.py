import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import bottleneck as bn
from astropy.table import Table, join, vstack
from astropy.io import fits, ascii
from PyAstronomy import pyasl
from time import sleep
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings("ignore")

# plotting parameters
plt.rc('axes', labelsize=12)
plt.rc('axes', labelweight='regular')
plt.rc('axes', titleweight='regular')
plt.rc('axes', linewidth=1)
plt.rc('xtick',labelsize=10,direction='in',top=True)
plt.rc('ytick',labelsize=10,direction='in',right=True)
plt.rcParams['xtick.major.pad']='10'
plt.rcParams['ytick.major.pad']='10'

paper_params = {#'text.usetex' : True,
           'font.size' : 10,
           #'font.family' : 'lmodern',
           'figure.dpi' : 200
           }
plt.rcParams.update(paper_params)

# constants for converting model values for mass [6] and radius [3] into cgs values
# log = (G M / R^2)
bigG = 6.67430e-8
solar_mass_grams = 1.98847e33
solar_radius_cm = 6.95700e10

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

list_surf_files = np.genfromtxt('list_surface_files',dtype=str)
#print(list_surf_files[0:5])
# generate indicies
split = [split_surf_file(sf) for sf in list_surf_files if split_surf_file(sf) is not None]
# select unique indicies
unique = [list(np.unique(spl)) for spl in np.array(split).T]
#print((unique))

print('Reading MODEL grid from file: "model_grid_big_file.dat" and VALID file "valid_grid_big_file.dat"')
#with open('/nexus/posix0/MIA-astro-env/hxr/adimoff/BinaryStars/Accretion_New/model_grid_big_file.dat','rb') as file:
MODEL = np.load('/nexus/posix0/MIA-astro-env/hxr/adimoff/BinaryStars/Accretion_New/model_grid_big_file.dat')
VALID = np.load('/nexus/posix0/MIA-astro-env/hxr/adimoff/BinaryStars/Accretion_New/valid_grid_big_file.dat')

print('Reading in observational data')
# read from the X_Fe_table from abundance infomation
Obs_Data_Table = ascii.read('/nexus/posix0/MIA-astro-env/hxr/adimoff/abund_data/X_Fe_table_3.dat',delimiter='&')
param_mass_table = ascii.read('/nexus/posix0/MIA-astro-env/hxr/adimoff/abund_data/param_mass_table.dat',format='latex')
full_obs_data_table = join(param_mass_table,Obs_Data_Table,keys='stars')
# read in other literature data
cristallo_data = ascii.read('/nexus/posix0/MIA-astro-env/hxr/adimoff/abund_data/Cristallo_2016.dat')
goswami_2020_data = ascii.read('/nexus/posix0/MIA-astro-env/hxr/adimoff/abund_data/Gos_Rat_Gos_2020.dat')
goswami_2021_data = ascii.read('/nexus/posix0/MIA-astro-env/hxr/adimoff/abund_data/Gos_Rat_Gos_CEMPs_2021.dat')
DeCastro_Roriz = ascii.read('/nexus/posix0/MIA-astro-env/hxr/adimoff/abund_data/DeCastro_Roriz_Table.dat')
# stack them all up in the master table
master_table = vstack([full_obs_data_table,cristallo_data,goswami_2020_data,goswami_2021_data,DeCastro_Roriz],join_type='inner')

obs_names = np.array(['C','C_sig','FeH','FeH_sig','Sr','Sr_sig','Y','Y_sig','Zr','Zr_sig','Nb','Nb_sig','Mo','Mo_sig','Ru','Ru_sig','Ba','Ba_sig','La','La_sig','Ce','Ce_sig','Pr','Pr_sig','Nd','Nd_sig','Sm','Sm_sig','Eu','Eu_sig','Dy','Dy_sig','Pb','Pb_sig'])

star_indices = np.arange(len(master_table))

def get_star_params(table_in_question, star_index):
    star_name = table_in_question['stars'][star_index]
    
    #obs_data = []
    #for name in obs_names:
    #    obs_data.append(np.array(Obs_Data_Table[np.where(full_obs_data_table['stars']==star_name)][name])[0])
    
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
                     table_in_question['Zr'][star_index],table_in_question['Nb'][star_index], 
                     table_in_question['Mo'][star_index],table_in_question['Ru'][star_index]])
    hs = bn.nanmean([table_in_question['Ba'][star_index],table_in_question['La'][star_index],
                     table_in_question['Ce'][star_index],table_in_question['Nd'][star_index]])    

    C_enhanced = table_in_question['C'][star_index] > 0.50 # or 0.70? or 0.50?
    Metal_Poor = table_in_question['FeH'][star_index] < -1.00

    dash_s = (hs - table_in_question['Eu'][star_index] > 0.50) and hs > 1.00 ## Beers + Christlieb 2005 / Jorissen 2016
    dash_s = (hs > 1.00) and (hs - table_in_question['Eu'][star_index] > 0.50) ## Sivarani 2006
    dash_s = (hs > 1.00) and (hs - table_in_question['Eu'][star_index] > 0.00) and (table_in_question['Eu'][star_index] < 1.00) ## Jonsell 2006
    dash_s = (hs - table_in_question['Eu'][star_index] > 0.00) and hs > 0.90 ## Abate 2016
    
    dash_rs = (hs - table_in_question['Eu'][star_index] < 0.50) and (hs - table_in_question['Eu'][star_index] > 0.00) ## Beers + Christlieb 2005 / Jorissen 2016
    dash_rs = (hs > 0.0) and (hs < 0.50) ## Sivarani 2006
    dash_rs = (table_in_question['Eu'][star_index] > 1.00) and (hs - table_in_question['Eu'][star_index] > 0.0) and (hs - table_in_question['Eu'][star_index] < 1.0) ## Hansen 2019
    dash_rs = (table_in_question['Eu'][star_index] > 1.00) and (hs > 1.00) and (hs - table_in_question['Eu'][star_index] > 0.0) ## Abate 2016
    
    dash_r = table_in_question['Eu'][star_index] > 1.00 ## Beers + Christlieb 2005
    dash_r = table_in_question['Eu'][star_index] > 1.00 and hs < 0.0 ## Abate 2016

    ## CAMILLA'S CRITERION BASED ON SR AND BA -- LS AND HS
    #CEMP-no [Sr/Ba] > 0.75 New classification
    if C_enhanced and Metal_Poor and (ls - hs > 0.75): # Hansen 2019
        flag = 'CEMP-no'
    elif C_enhanced and Metal_Poor and hs < 0.0: # Beers + Christlieb 2005 / Jorissen 2016
        flag = 'CEMP-no'

    elif C_enhanced and Metal_Poor and (ls - hs > -0.5) and (ls - hs < 0.75): # Hansen 2019
        flag = 'CEMP-s'
    elif C_enhanced and Metal_Poor and dash_s: 
        flag = 'CEMP-s'

    elif C_enhanced and Metal_Poor and (ls - hs > -1.5) and (ls - hs > -0.5): # Hansen 2019
        flag = 'CEMP-rs'
    elif C_enhanced and Metal_Poor and dash_rs: 
        flag = 'CEMP-rs'

    elif C_enhanced and Metal_Poor and (ls < -1.5): # Hansen 2019
        flag = 'CEMP-r'
    elif C_enhanced and Metal_Poor and dash_r: 
        flag = 'CEMP-r'

    ## add flag for CH stars as those at higher metallicities but share same characteristics
    elif Metal_Poor: # ...? These stars are metal poor, but not SO carbon enriched... this is tough, some could be r/s?
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
            break
        # Barium stars, avg mass is ~2.50
        elif flag == 'WBa' or flag == 'SBa':
            m_star = 2.50
            e_m_star = 1.00
        # CEMP stars, avg mass is ~0.80
        elif flag == 'CEMP-s' or flag == 'CEMP-r/s' or flag == 'CH':
            m_star = 0.80
            e_m_star = 0.80
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


#####################################
# THIS IS WHERE THE PLOTTING HAPPENS #
#####################################
index_of_choice = 251

star_name, obs_names_II, obs_elems_nums, obs_elems_II, obs_errrs_II, X_obs, X_sig, type_flag = get_star_params(master_table,index_of_choice)
print(star_name)
adfsfads
# read in best models from comparison script
best_models = np.load('models_full_mass_limit/best_models_'+star_name+'_metal_con_mass_free_mass_limit_weights.dat') 
best_chis = np.load('chis_full_mass_limit/'+star_name+'_metal_con_mass_free_mass_limit_weights.dat'
print(best_models)
print(len(best_models))
print(np.shape(best_models))

Num_tracks = len(best_models); #print(Num_tracks) # should be 3...

# Plot the three best models written out to the file...
#cmap = plt.colormaps['rainbow']
#mod_colors = cmap(np.linspace(0.01,0.09,Num_tracks))
mod_colors = ['black','cyan','magenta'] #['darkorchid','royalblue','limegreen','darkorange','firebrick']
mod_sizes = np.linspace(8,32,Num_tracks)

fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(7,3))
    
for i,mod_color,mod_size in zip(np.arange(0,Num_tracks),mod_colors,mod_sizes):
    print(i)
    # find name of model track using unique array and best model indices
    mod_metal = str(round(np.log10(float(str('0.'+unique[0][best_models[i][0]]))/0.0142),2))
    mod_AGB = str(unique[1][best_models[i][1]].split('p')[0]+'.'+unique[1][best_models[i][1]].split('p')[1]+'0')
    mod_m_i = str(unique[2][best_models[i][2]])
    mod_m_f = str(unique[3][best_models[i][3]])
    track_name = str(r'M$_i$=' + mod_m_i + r' M$_f$=' + mod_m_f + r' M$_{AGB}$=' + mod_AGB +' [Fe/H]=' + mod_metal)
    #print(track_name)
    
    # find best model point, extract logT, logg, and abundances
    mod_logT = MODEL[best_models[i][0],best_models[i][1],best_models[i][2],best_models[i][3],best_models[i][4],1]
    mod_logg = MODEL[best_models[i][0],best_models[i][1],best_models[i][2],best_models[i][3],best_models[i][4],2]
    mod_abunds = MODEL[best_models[i][0],best_models[i][1],best_models[i][2],best_models[i][3],best_models[i][4]][4:]
    # find whole track
    valid = VALID[best_models[i][0],best_models[i][1],best_models[i][2],best_models[i][3],:]
    mod_logT_track = MODEL[best_models[i][0],best_models[i][1],best_models[i][2],best_models[i][3],:,1][valid]
    mod_logg_track = MODEL[best_models[i][0],best_models[i][1],best_models[i][2],best_models[i][3],:,2][valid]
    
    # plot the whole track, data points, and best model points along the track
    ax1.plot(mod_logT_track,mod_logg_track,zorder=Num_tracks-i,lw=1,alpha=0.75,color=mod_color,label=track_name)
    ax1.scatter(mod_logT,mod_logg,marker='o',facecolor='white',edgecolors=mod_color,s=20,zorder=Num_tracks-i+1,alpha=0.75)
    
    # plot abundances
    ax2.scatter(obs_names_II, mod_abunds,zorder=Num_tracks-i,marker='D',c=np.array([mod_color]),s=mod_size,alpha=0.75)

# plot observational data
ax1.errorbar(X_obs[1],X_obs[2],xerr=X_sig[1],yerr=X_sig[2],marker='o',capsize=2,elinewidth=1,markersize=4,color='black',markeredgecolor='black',markerfacecolor='white',zorder=Num_tracks+1,label=star_name)

for name, elem, err in zip(obs_names_II, obs_elems_II, obs_errrs_II):
    if np.isnan(err):
        ax2.scatter([name],[elem],marker='v',s=18,edgecolor='black',facecolor='white',zorder=Num_tracks+1)
    else:
        ax2.errorbar([name],[elem],yerr=[err],ls='',marker='o',capsize=2,elinewidth=1,markersize=4,color='black',markeredgecolor='black',markerfacecolor='white',zorder=Num_tracks+1)

ax1.set_xlabel(r'$\log(\rm{T_{eff}})$')
ax1.set_ylabel(r'$\log(g)$')
ax1.invert_xaxis()
ax1.invert_yaxis()
ax1.legend(fontsize=4.5,loc=2,markerscale=0.65)
ax1.set_xlim(4.0,3.4)
ax1.set_xlim(X_obs[1]+0.15,X_obs[1]-0.15)
ax1.set_ylim(5.1,-1.1); 
ax1.set_ylim(5.1,-0.6)
ax1.set_ylim(X_obs[2]+1.0,X_obs[2]-1.0)

ax2.set_xlabel('Atomic Species')
ax2.tick_params(axis='x', rotation=90)
ax2.set_ylabel('[X/Fe]')
ax2.set_ylim(-0.25,2.65)

plt.tight_layout()
#plt.show()
plt.savefig('new_output_'+star_name+'.png',dpi=200,bbox_inches=None,layout='constrained')
