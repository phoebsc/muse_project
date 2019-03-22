import numpy as np
import codecs, json, os
from matplotlib import pyplot as plt
from random import shuffle
from read import Recording
from read import DataImportType
from analysis import simple_corr, compute_single_freq, butter_lowpass_filter
import dickey_fuller as df
from datetime import datetime

n_iterations = 2
n_pairs = 2
#epoch_lengths = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
epoch_lengths = [1.0, 4.0]
jsonpath = './json_files_raw'
channels =['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

# For every epoch, select n_pairs n_iterations times.
# Average across n_pairs, then average those averages across n_iterations

# 1. Randomly select some files
# Shuffle the json files that end in "0"
#csv_files = [f for f in os.listdir(path) if not f.startswith('.')]
json_pairs = [f for f in os.listdir(jsonpath) if not f.startswith('.')]

# 2. Epoch them into x second segments and run autoreject
recordings = []

epoch_pcts = []
output_fname = '/Users/averybedows/Desktop/adf_output_' + datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + '.txt'
file = open(output_fname, 'w')

for epoch_length in epoch_lengths:
    avgs_per_iteration = []
    for iteration in range(n_iterations):

        # Shuffle the files so we can draw a random sample
        #shuffle(csv_files)
        shuffle(json_pairs)

        # Get the first n_pairs files
        #file_pairs_for_analysis = csv_files[0:n_pairs]
        pairs_for_analysis = json_pairs[0:n_pairs]

        fractions_per_pair = []

        for pair in pairs_for_analysis:

            # initialization
            config = {}

            # Config and preprocessing params for lowlands dataset
            config['lowlands'] = {'srate': 128,  # sampling rate
                                  'channels': ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8',
                                               'AF4'],  # channel names
                                  'epoch': [-epoch_length / 2.0, epoch_length / 2.0],  # epochs of epoch_length
                                  'ICAparam': {'method': 'fastica',
                                               'n_components': 14,
                                               'decim': 3,
                                               'random_state': 23},  # ICA params
                                  'notch': 60,  # notch filter frequency
                                  'bandpass': [0.1, 30],  # initial band pass range
                                  }

            recording = Recording(os.path.join(jsonpath, pair), config, 'lowlands', DataImportType.JSON)
            recording.filter()
            recording.epoch()
            #recording.autoreject()
            recordings.append(recording)

            # Assess stationarity
            original_data = recording.eegs[0].get_data()
            stationary_count = 0 # Tally up percentage of good epochs for this pair

            for epoch in range(original_data.shape[0]):
                data = original_data[epoch, 0, :].reshape((original_data.shape[2],))

                is_stationary = df.StationarityTests.ADF_Stationarity_Test(data, print_results=False)
                if (is_stationary):
                    stationary_count += 1

            # Calculate the fraction of stationary epochs
            fraction = (100 * float(stationary_count) / float(original_data.shape[0]))
            fractions_per_pair.append(fraction)

        # Average the percentages of good epochs
        pair_avg = sum(fractions_per_pair) / len(fractions_per_pair)
        iteration_output_str = 'Iteration ' + str(iteration) + ', epoch length ' + \
                               str(epoch_length) + ' seconds: ' + "%.2f" % pair_avg + '% good epochs.' + '\n'
        file.write(iteration_output_str)
        print('Iteration', iteration,
              ', epoch length', epoch_length, 'seconds: ',
              "%.2f" % pair_avg, '% good epochs.')
        avgs_per_iteration.append(pair_avg)

    # Average the n_pair-averaged epochs across iterations
    iteration_avg = sum(avgs_per_iteration) / len(avgs_per_iteration)
    epoch_output_str = 'Epoch of ' + str(epoch_length) + ' seconds: ' + "%.2f" % iteration_avg + '% good epochs.' + '\n\n'
    file.write(epoch_output_str)
    print('Epoch of', epoch_length, 'seconds: ', "%.2f" % iteration_avg, '% good epochs.')
    epoch_pcts.append(iteration_avg)

# Plot graph showing how the stationarity averages change as a function of epoch length
plt.clf()
plt.bar(epoch_lengths, epoch_pcts)
plt.title("Avg. % good epochs x epoch length")
plt.xlabel('Epoch Length (s)')
plt.ylabel('% Good Epochs')
figure_fname = '/Users/averybedows/Desktop/adf_test_figure_' + datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + '.png'
plt.savefig(figure_fname)
file.close()

