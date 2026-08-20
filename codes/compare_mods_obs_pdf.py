print('Comparing observed data to model grid')
print('(( currently only works for 1 star at a time, but can be looped or parallelized ))')
print()

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from astropy.table import Table, join
from astropy.io import fits, ascii
import pickle
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

paper_params = {
       # 'text.usetex' : True,
        'font.size'   : 10,
       #'font.family' : 'lmodern',
        'figure.dpi'  : 200
        }
plt.rcParams.update(paper_params)

def split_surf_file(surf_file):
    """
    from file string, extract
    model name,
    metallicity, 
    AGB mass
    inital mass
    final mass
    """
    model_name = surf_file.split('/')[-1][8:]
    metalZ = model_name.split('_')[-1][1:]
    AGB_mass = model_name.split('_')[-2][3:] #(model_name.split('_')[-2][3:4]+'.'+model_name.split('_')[-2][5:6])
    init_mass = (model_name.split('_')[0][1]+'.'+model_name.split('_')[0][3:5])
    final_mass = (model_name.split('_')[1][1]+'.'+model_name.split('_')[1][3:5])
    #if AGB_mass < init_mass: 
    #    return None
    return metalZ,AGB_mass,init_mass,final_mass

list_surf_files = np.genfromtxt('list_surface_files',dtype=str)
# generate indicies
split = [split_surf_file(sf) for sf in list_surf_files if split_surf_file(sf) is not None]
# select unique indicies
unique = [list(np.unique(spl)) for spl in np.array(split).T]
#print(unique)

print('Reading MODEL grid from file: "model_grid_big_file.dat" and VALID file "valid_grid_big_file.dat"')
#with open('/data/beegfs/astro-storage/groups/rix/dimoff/BinaryStars/Accretion_New/model_grid_big_file.dat','rb') as file:
MODEL = np.load('/data/beegfs/astro-storage/groups/rix/dimoff/BinaryStars/Accretion_New/model_grid_big_file.dat')
VALID = np.load('/data/beegfs/astro-storage/groups/rix/dimoff/BinaryStars/Accretion_New/valid_grid_big_file.dat')
#print(np.shape(MODEL), np.shape(VALID))

print('Reading in observational data')
# read from the X_Fe_table from abundance infomation
Obs_Data_Table = ascii.read('/data/beegfs/astro-storage/groups/rix/dimoff/abund_data/X_Fe_table_3.dat',delimiter='&')
param_mass_table = ascii.read('/data/beegfs/astro-storage/groups/rix/dimoff/abund_data/param_mass_table.dat',format='latex')

obs_names = np.array(['C','C_sig','Mg','Mg_sig','FeH','FeH_sig','Sr','Sr_sig','Y','Y_sig','Zr','Zr_sig','Mo','Mo_sig','Ba','Ba_sig','La','La_sig','Ce','Ce_sig','Nd','Nd_sig','Eu','Eu_sig','Pb','Pb_sig'])

# collect all data for a star
star_index = 1
star_name = param_mass_table['stars'][star_index]
print('star:',star_name)

#print(Obs_Data_Table['stars']==star_name)[name]

obs_data = []
for name in obs_names:
    obs_data.append(np.array(Obs_Data_Table[np.where(Obs_Data_Table['stars']==star_name)][name])[0])

# surface parameters
logg = param_mass_table['logg'][star_index]
e_logg = param_mass_table['e_logg'][star_index]
gstar = 10**logg; e_gstar = 10**(logg-e_logg)

Teff = param_mass_table['Teff'][star_index]
e_Teff = param_mass_table['e_Teff'][star_index]
log_Teff = np.log10(Teff); log_e_Teff = (e_Teff**2/(Teff*np.log10(e_Teff))**2)**(1/2)

FeH = param_mass_table['Fe'][star_index]
eFeH = param_mass_table['e_Fe'][star_index]
#starZ = np.exp(FeH)*0.0142
#e_starZ = 
#print(starZ,FeH)

m_star = param_mass_table['vis'][star_index]
e_m_star = param_mass_table['e_vis'][star_index]
m_AGB = param_mass_table['AGB'][star_index]

# set parameters for plotting element comparison
# element names (skip errors in names list)
obs_names_I_ = obs_names[0:-1:2]
# elements from observed list, and errors
obs_elems = obs_data[0:-1:2]
obs_elems_nums = [6,12,38,39,40,42,56,57,58,60,63,82]
obs_errrs = obs_data[1::2]
# skip Fe in these lists...
obs_names_II = []
obs_elems_II = []
obs_errrs_II = []
for i in range(len(obs_elems)): 
    if i != 2:
        obs_names_II.append(obs_names_I_[i])
        obs_elems_II.append(obs_elems[i])
        obs_errrs_II.append(obs_errrs[i])

#print('data for a single star')
#print(star_name, m_star, e_m_star, FeH, eFeH, logg, e_logg, Teff, e_Teff)
#print(np.round(obs_elems_II,decimals=2))

print('Collect observational data for the star in an array')
X_obs = np.concatenate([[FeH,log_Teff,logg,m_star],obs_elems_II])
X_sig = np.concatenate([[eFeH,log_e_Teff,e_logg,e_m_star],obs_errrs_II])

print('Compute squared residuals between observational data and grid of models')
residuals_sq = np.square((MODEL - X_obs) / X_sig) 

#print('computing subset, lets see how fast')
#residuals_test = np.square((MODEL[:,:,:,:,-1] - X_obs) / X_sig)
#print(X_obs)
#print(X_sig)
#print(MODEL[5][1][8][1][2509])
#print(residuals_sq[5][1][8][1][2509])

print(np.shape(residuals_sq))
print('Compute chi squared value, probability distribution function, and log likelihood')
# sum across axis for chi_sq value
chi_sq = residuals_sq.sum(axis=-1)    
print(np.shape(chi_sq))
pdf = 1./(np.sqrt(2*np.pi*np.square(X_sig))) * np.exp((-1/2)*(residuals_sq)) # this is the likelihood
log_lik = (1./2.)*np.log(np.square(X_sig)) - (1./(2.*np.square(X_sig))*pdf)

log_lik = log_lik.sum(axis=-1)
log_lik[np.invert(VALID)] = np.inf
print(np.shape(log_lik))

#print('writing chi_squareds to file...')
#with open('chi_sq_'+star_name+'.dat','wb') as chisq_handler:
#    np.save(chisq_handler,chi_sq)

print('Finding best models, invert log_lik array')
best_inds_all = np.array(np.unravel_index(np.argsort(log_lik,axis=None),log_lik.shape,order='C')).T

Num_tracks = 5

best_tracks = []
best_models = []

for i,ind in enumerate(best_inds_all[:,:-1]):
    ind = list(ind)
    if ind not in best_tracks:
        best_tracks.append(ind)
        best_models.append(best_inds_all[i])
    if len(best_tracks) > Num_tracks:
        break

best_tracks = np.array(best_tracks)
best_models = np.array(best_models)
#print(best_tracks[0])
print(best_models[0])
#print(best_tracks)

print(log_lik[best_models[0][0]][best_models[0][1]][best_models[0][2]][best_models[0][3]][best_models[0][4]])
print(log_lik[best_models[1][0]][best_models[1][1]][best_models[1][2]][best_models[1][3]][best_models[1][4]])
print(log_lik[best_models[2][0]][best_models[2][1]][best_models[2][2]][best_models[2][3]][best_models[2][4]])
print(log_lik[best_models[3][0]][best_models[3][1]][best_models[3][2]][best_models[3][3]][best_models[3][4]])

#print('Plotting best models')
##cmap = plt.colormaps['rainbow']
##mod_colors = cmap(np.linspace(0.01,0.09,Num_tracks))
#mod_colors = ['darkorchid','royalblue','limegreen','darkorange','firebrick']
#mod_sizes = np.linspace(8,32,Num_tracks)
#
#fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(7,3))
#
##print(unique)
#
#for i,mod_color,mod_size in zip(np.arange(0,Num_tracks),mod_colors,mod_sizes):
##    print(i,best_models[i])
##    print(unique[0][best_models[i][0]],unique[1][best_models[i][1]],unique[2][best_models[i][2]],unique[3][best_models[i][3]])
#    #print(best_tracks[i])
## find name of model track using unique and best model indices
#    mod_metal = str(round(np.log10(float(str('0.'+unique[0][best_models[i][0]]))/0.0142),2))
#    mod_AGB = str(unique[1][best_models[i][1]].split('p')[0]+'.'+unique[1][best_models[i][1]].split('p')[1]+'0')
#    mod_m_i = str(unique[2][best_models[i][2]])
#    mod_m_f = str(unique[3][best_models[i][3]])
#
#    track_name = str('[Fe/H]=' + mod_metal + r' M$_{AGB}$=' + mod_AGB + r' M$_i$=' + mod_m_i + r' M$_f$=' + mod_m_f)
#
## find best model point, extract logT, logg, and abundances
#    mod_logT = MODEL[best_models[i][0],best_models[i][1],best_models[i][2],best_models[i][3],best_models[i][4],1]
#    mod_logg = MODEL[best_models[i][0],best_models[i][1],best_models[i][2],best_models[i][3],best_models[i][4],2]
#    mod_abunds = MODEL[best_models[i][0],best_models[i][1],best_models[i][2],best_models[i][3],best_models[i][4]][4:]
#    # find whole track
#    valid = VALID[best_models[i][0],best_models[i][1],best_models[i][2],best_models[i][3],:]
#    mod_logT_track = MODEL[best_models[i][0],best_models[i][1],best_models[i][2],best_models[i][3],:,1][valid]
#    mod_logg_track = MODEL[best_models[i][0],best_models[i][1],best_models[i][2],best_models[i][3],:,2][valid]
#
## plot the whole track, data points, and best model points along the track
#    ax1.plot(mod_logT_track,mod_logg_track,zorder=Num_tracks-i,alpha=0.75,color=mod_color,label=track_name)
#    ax1.scatter(mod_logT,mod_logg,marker='o',facecolor='white',edgecolors=mod_color,s=20,zorder=Num_tracks-i,alpha=0.75)
#
#    # plot abundances
#    ax2.scatter(obs_names_II, mod_abunds,zorder=Num_tracks-i,marker='D',c=np.array([mod_color]),s=mod_size,alpha=0.75)
#
#ax1.errorbar(log_Teff,logg,xerr=log_e_Teff,yerr=e_logg,marker='o',capsize=2,elinewidth=1,markersize=4,color='black',markeredgecolor='black',markerfacecolor='white',zorder=Num_tracks+1,label=star_name)
#ax2.errorbar(obs_names_II, obs_elems_II, yerr=obs_errrs_II,ls='',marker='o',capsize=2,elinewidth=1,markersize=4,color='black',markeredgecolor='black',markerfacecolor='white',zorder=Num_tracks+1)
#
#ax1.set_xlabel(r'$\log(\rm{T_{eff}})$')
#ax1.set_ylabel(r'$\log(g)$')
#ax1.invert_xaxis()
#ax1.invert_yaxis()
#ax1.legend(fontsize=4.5,loc=2,markerscale=0.65)
#
#ax1.set_xlim(4.0,3.4)
#ax1.set_xlim(log_Teff+0.15,log_Teff-0.15)
#ax1.set_ylim(5.1,-1.1); 
#ax1.set_ylim(5.1,-0.6)
#ax1.set_ylim(logg+1.5,logg-1.0)
#
#ax2.set_xlabel('Atomic Species')
#ax2.set_ylabel('[X/Fe]')
#ax2.set_ylim(-0.5,2.65)
#
#print('End of Line')
##plt.savefig('accretion_model'+star_name+'.png',dpi=200,bbox_inches=None,layout='constrained')
#plt.show()
