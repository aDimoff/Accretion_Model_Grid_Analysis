import numpy as np
import matplotlib.pyplot as plt

# look at rough chi file first...
rough_chi = 'BD+042466_rough_chis.npy'
chi_data = np.load(rough_chi)
print(np.min(chi_data))
print(np.where(chi_data == np.min(chi_data)))
print('')

# then zoom in on finer tracks
chi_files = ['BD+042466_545_zoom_chis.npy', 'BD+042466_846_zoom_chis.npy', 'BD+042466_897_zoom_chis.npy']

for chi_file in chi_files:

    chi_data = np.load(chi_file)

    print(np.min(chi_data))

    print(np.where(chi_data == np.min(chi_data)))

    #print(np.shape(chi_data))
#print(chi_data)
