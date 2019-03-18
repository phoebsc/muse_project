import numpy as np
from matplotlib import pyplot as plt
from read import Recording
from analysis import simple_corr, butter_lowpass_filter

# initialization
config = {}
path = 'C:/Users/Phoebe Chen/Dropbox/MWM_Lowlands_2015_DATA/15-08-21-15-33-20'

# Config and preprocessing params for lowlands dataset
config['lowlands'] = {'srate': 128,  # sampling rate
                      'channels':['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4'],  # channel names
                      'epoch': [-0.5, 0.5],  # epochs of 1s length
                      'PCAparam': None,  # PCA params (not used)
                      'ICAparam': {'method': 'fastica',
                                   'n_components': 14,
                                   'decim': 3,
                                   'random_state': 23},  # ICA params
                      'notch': 60,  # notch filter frequency
                      'bandpass':[0.1,30],  # initial band pass range
                      'FFT': {'window_size':128,
                              'overlap': 0.5},  # FFT params (not used yet)
                      }

# Create the Recording object
recording = Recording(path, config, 'lowlands')

# Preprocess the Recording
recording.filter()
recording.epoch()
recording.autoreject()
#recording.identify_ICA()



ind = recording.config['autoreject'][0]
plv_corr = simple_corr(recording, 10, mode='plv', epoch_wise=True)
power_corr = simple_corr(recording, 10, mode='power', epoch_wise=True)
order = 6
fs = 50.0       # sample rate, Hz
cutoff = 2

for i in range(14):
    plt.plot(ind,butter_lowpass_filter(plv_corr[i], cutoff, fs, order),alpha=0.5)
    # plt.scatter(ind, butter_lowpass_filter(corr[i], cutoff, fs, order), alpha=0.5)
    #plt.plot(ind, corr[i], alpha=0.5)