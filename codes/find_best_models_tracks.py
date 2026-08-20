print('Finding best models within the chi_sq grid')
print('importing packages...')
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

#paper_params = {'text.usetex' : True,
#           'font.size' : 10,
#           'font.family' : 'lmodern',
#           'figure.dpi' : 200
#           }
#plt.rcParams.update(paper_params)

print('read')
# read chi_sq data file
chi_sq = np.load('chi_sq_surface_file.dat')
#print(np.shape(chi_sq))

# read MODEL file with all info
MODEL = np.load('/data/beegfs/astro-storage/groups/rix/dimoff/BinaryStars/Accretion_New/model_grid_big_file.dat')
VALID = np.load('/data/beegfs/astro-storage/groups/rix/dimoff/BinaryStars/Accretion_New/valid_grid_big_file.dat')
print(np.shape(MODEL),np.shape(VALID))
# identify best model indicies in chi array
print('finding best fit')
best_inds_1 = np.unravel_index(np.argmin(chi_sq),chi_sq.shape)
print(best_inds_1)
print(best_inds_1[:-1])

# use best indicies to find best model on best track
#print('Best model shape',np.shape(MODEL[best_inds_1[0],best_inds_1[1],best_inds_1[2],best_inds_1[3],best_inds_1[4],:]))
#print('Best model parameters',MODEL[best_inds_1[0],best_inds_1[1],best_inds_1[2],best_inds_1[3],best_inds_1[4],:])

#valid = VALID[best_inds_1[0],best_inds_1[1],best_inds_1[2],best_inds_1[3],best_inds_1[4]] 
#best_logT = MODEL[best_inds_1[0],best_inds_1[1],best_inds_1[2],best_inds_1[3],best_inds_1[4],1]
#best_logg = MODEL[best_inds_1[0],best_inds_1[1],best_inds_1[2],best_inds_1[3],best_inds_1[4],2]
#best_abunds = MODEL[best_inds_1[0],best_inds_1[1],best_inds_1[2],best_inds_1[3],best_inds_1[4]][4:]
#print(best_abunds)
#best_logT_track = MODEL[best_inds_1[0],best_inds_1[1],best_inds_1[2],best_inds_1[3],:,1][valid]
#best_logg_track = MODEL[best_inds_1[0],best_inds_1[1],best_inds_1[2],best_inds_1[3],:,2][valid]

obs_names = ['C','Mg','Sr','Y','Zr','Mo','Ba','La','Ce','Nd','Eu','Pb']
#--------------------------------------------

fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(7,3))

#ax1.plot(best_logT_track,best_logg_track)
#ax1.scatter(best_logT,best_logg,marker='x',s=8)
#ax2.scatter(obs_names,best_abunds)

# save best model Kiel diagram info for plotting purposes
# find full evolutionary track of best model...
#np.save('best_kiel_track.dat',[MODEL[best_inds_1[0],best_inds_1[1],best_inds_1[2],best_inds_1[3],:,2], MODEL[best_inds_1[0],best_inds_1[1],best_inds_1[2],best_inds_1[3],:,3]])

print('finding other good fits')

best_inds_all = np.array(np.unravel_index(np.argsort(chi_sq,axis=None),chi_sq.shape)).T
#print(np.shape(best_inds_all))

#print(np.array(best_inds_all).T[16][0:-1])

# this takes time, we only want first 5 best mods/tracks
#unique, index = np.unique(best_inds_all,axis=0,return_index=True)
#best_tracks = best_inds_all[np.sort(index)]
#np.logical_and.reduce(best_tracks[] = best_inds_all[:,:-1],axis=1)

Num_tracks = 5

best_tracks = []
for ind in best_inds_all[:,:-1]:
    ind = list(ind)
    if ind not in best_tracks:
        best_tracks.append(ind)
    if len(best_tracks) > Num_tracks:
        break

best_tracks = np.array(best_tracks)
print(np.shape(best_tracks))
print(best_tracks)
#while len(seen) < 5:
for i in np.arange(0,Num_tracks):
    #print(best_tracks[i])
    
    logT = MODEL[best_inds_1[0],best_inds_1[1],best_inds_1[2],best_inds_1[3],best_inds_1[4],1]
    logg = MODEL[best_inds_1[0],best_inds_1[1],best_inds_1[2],best_inds_1[3],best_inds_1[4],2]
    abunds = MODEL[best_inds_1[0],best_inds_1[1],best_inds_1[2],best_inds_1[3],best_inds_1[4]][4:]

    valid = VALID[best_tracks[i][0],best_tracks[i][1],best_tracks[i][2],best_tracks[i][3]]
    logT_track = MODEL[best_tracks[i][0],best_tracks[i][1],best_tracks[i][2],best_tracks[i][3],:,1][valid]
    logg_track = MODEL[best_tracks[i][0],best_tracks[i][1],best_tracks[i][2],best_tracks[i][3],:,2][valid]
    ax1.plot(logT_track,logg_track)
    ax1.scatter(logT, logg)

    ax2.scatter(obs_names, abunds)

ax1.set_xlabel('T_eff')
ax1.set_ylabel('logg')
ax1.invert_xaxis()
ax1.invert_yaxis()
ax1.legend(fontsize=4.5,loc=2,markerscale=0.65)
ax1.set_ylim(5.1,-0.6)
ax1.set_xlim(4.0,3.4)

ax2.set_xlabel('Atomic Species')
ax2.set_ylabel('[X/Fe]')
ax2.set_ylim(-0.5,2.75)


print('End of Line')

plt.show()
