"""
Convert csv data to json data
"""

from read import Recording
import os
import json
import numpy as np
import pandas as pd

channels =['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
datapath = '/Users/averybedows/Dropbox (Personal)/MWM_Lowlands_2015_DATA/'
jsonpath = '/Users/averybedows/Desktop/json_files_raw'

for pair in os.listdir(datapath):
    # Make sure we don't do this for the ./DS_Store file
    if '.' in pair:
        continue

    pair_path = os.path.join(datapath, pair)

    directory = os.path.join(jsonpath, pair)
    paired_data = []
    filenames = []
    for subject in os.listdir(pair_path):
        # Each subject has their own data
        data = np.full([len(channels), 0], np.nan)

        subject_path = os.path.join(pair_path, subject)
        for filename in os.listdir(subject_path):
            path_to_csv = os.path.join(subject_path, filename)
            csv_cols = channels.copy()
            csv_cols.append('COUNTER')
            df = pd.read_csv(path_to_csv, usecols=csv_cols, index_col=False)
            data = np.concatenate((data, df[channels].get_values().T), axis=1)

        paired_data.append(data)
        filenames.append(os.path.join(directory, subject))

    # Now that we've extracted the CSV data to ndarrays for this pair, let's write to json
    if not os.path.exists(os.path.join(jsonpath, pair)):
        os.makedirs(directory)

    file_0 = open(filenames[0], 'w')
    json.dump(paired_data[0].tolist(), file_0)
    file_0.close()

    file_1 = open(filenames[1], 'w')
    json.dump(paired_data[1].tolist(), file_1)
    file_1.close()