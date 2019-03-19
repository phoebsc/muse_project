import numpy as np
from matplotlib import pyplot as plt
from read import Recording
from analysis import simple_corr, compute_single_freq, butter_lowpass_filter
import os
import json

# initialization
config = {}
path = 'C:/Users/Phoebe Chen/Dropbox/MWM_Lowlands_2015_DATA/15-08-21-15-33-20'

# Config and preprocessing params for lowlands dataset
config['lowlands'] = {'srate': 128,  # sampling rate
                      'channels':['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4'],  # channel names
                      'epoch': [-0.5, 0.5],  # epochs of 1s length
                      'ICAparam': {'method': 'fastica',
                                   'n_components': 14,
                                   'decim': 3,
                                   'random_state': 23},  # ICA params
                      'notch': 60,  # notch filter frequency
                      'bandpass':[0.1,30],  # initial band pass range

                      }

# Create the Recording object
recording = Recording(path, config, 'lowlands')

# Preprocess the Recording
recording.filter()
recording.epoch()
recording.autoreject()
#recording.identify_ICA()



ind = recording.config['autoreject'][0]
data = [recording.eegs[i].get_data() for i in range(2)]
plv_corr = simple_corr(data,recording.config['srate'], 10, mode='plv', epoch_wise=True)
power_corr = simple_corr(data,recording.config['srate'], 10, mode='power', epoch_wise=True)
order = 6
fs = 50.0
cutoff = 2

for i in range(14):
    plt.plot(ind,butter_lowpass_filter(plv_corr[i], cutoff, fs, order),alpha=0.5)
    # plt.scatter(ind, butter_lowpass_filter(corr[i], cutoff, fs, order), alpha=0.5)
    #plt.plot(ind, corr[i], alpha=0.5)

"""
compute average metrics for real and pseudo pairs
hypothesis: metrics are significantly higher for real pairs
"""

mode = 'plv'
datapath = 'C:/Users/Phoebe Chen/Dropbox/MWM_Lowlands_2015_DATA/'
jsonpath = 'C:/Users/Phoebe Chen/PycharmProjects/lowlands_benaki/json_files'
infopath = 'C:/Users/Phoebe Chen/PycharmProjects/lowlands_benaki/info_files'
for pair in os.listdir(datapath):
    filepath = os.path.join(datapath,pair)
    recording = Recording(filepath, config, 'lowlands')
    recording.filter()
    recording.epoch()
    recording.autoreject()
    if len(recording.eegs[0]) < 100:  # exclude data with less than 100 good epochs
        pass
    else:

        # save data
        filename0 = os.path.join(jsonpath, recording.config['pair_name'] + '_0')
        filename1 = os.path.join(jsonpath, recording.config['pair_name'] + '_1')
        with open(filename0, 'w') as outfile:
            json.dump(recording.eegs[0].get_data().tolist(), outfile)
        with open(filename1, 'w') as outfile:
            json.dump(recording.eegs[1].get_data().tolist(), outfile)

        # save autoreject and length info
        epoch_info = recording.config['autoreject'][0].tolist()
        length = recording.config['length']
        epoch_info.append(length)  # save a list of numbers, with list[0:-1]
        info = os.path.join(infopath, recording.config['pair_name'] + '_0')
        with open(info, 'w') as outfile:
            json.dump(epoch_info, outfile)




