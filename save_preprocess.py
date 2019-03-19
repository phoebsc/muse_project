"""
save preprocessed files
"""

from read import Recording
import os
import json

# initialization
config = {}

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
mode = 'plv'
datapath = 'C:/Users/Phoebe Chen/Dropbox/MWM_Lowlands_2015_DATA/'
jsonpath = './json_files'
infopath = './info_files'
for pair in os.listdir(datapath)[55:]:
    filepath = os.path.join(datapath, pair)
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