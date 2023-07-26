import numpy as np

npy_file = np.load('data/sprite_labels_nc_1788_16x16.npy')

np.savetxt("trial.csv", npy_file, delimiter=",")