import mne
import numpy as np
import pandas as pd
import os
import autoreject as autor
from mne.preprocessing import ICA
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import kpss

class Recording:

    def __init__(self, experiment_path, config, experiment):
        """
        :param experiment_path: Path to the data from the experiment
        :param config: Configuration info for the data
        :param experiment: Name of the experiment
        """

        self.config = config[experiment]

        # self.eegs is a list containing eeg data (RawArray) from each subject.
        self.eegs = []

        # Each subject has their own sub-directory in the' experiment_path directory
        for subject in os.listdir(experiment_path):
            eeg = self._read_excel(os.path.join(experiment_path,subject))

            # update stream information with new information
            self.eegs.append(eeg)

        # Make sure EEG recordings are for the same amount of time
        if len(self.eegs[0]) != len(self.eegs[1]):
            print('ERROR – incompatible EEG lengths: ', len(self.eegs[0]), ' ', len(self.eegs[1]))

        # Update config settings for this new Recording data
        self.update_config({'pair_name': self.path_to_filename(experiment_path),
                            'length': len(eeg) / self.config['srate']})

    def _read_excel(self, path):
        """
        :param path: Path to the excel document we want to read
        :return: Returns an mne.io.RawArray object representing the EEG data
        """
        data = np.full([len(self.config['channels']), 0], np.nan)

        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            csv_cols = self.config['channels'].copy()
            csv_cols.append('COUNTER')
            df = pd.read_csv(filepath, usecols=csv_cols, index_col=False)
            # check if the counter is correct
            #if 128 - df.COUNTER[0] + df.COUNTER.iloc[-1] != self.config['srate'] - 2:
            #    print("COUNTER error in " + filepath)
            data = np.concatenate((data, df[self.config['channels']].get_values().T), axis=1)

        # Initialize mne raw
        mneInfo = mne.create_info(ch_names=self.config['channels'],
                               sfreq=self.config['srate'],
                               ch_types='eeg',
                               montage='standard_1020')
        eeg = mne.io.RawArray(data, mneInfo, first_samp=0, verbose=False)

        print("Data successfully read: ", path)
        return eeg

    def filter(self):
        """
        Apply notch and bandpass filters to eeg data
        :return: None
        """
        # notch filter and bandpass filter
        self.eegs[0].filter(self.config['bandpass'][0], self.config['bandpass'][1])
        self.eegs[1].filter(self.config['bandpass'][0], self.config['bandpass'][1])

    def epoch(self):
        """
        Epochs the data according to the config data for this Recording object
        :return: None
        """
        time_before = self.config['epoch'][0]
        time_after = self.config['epoch'][1]

        # changed data to epoch type here
        for i, eeg in enumerate(self.eegs):
            events = mne.make_fixed_length_events(eeg, duration=time_after - time_before)
            self.eegs[i] = mne.Epochs(eeg,
                                events,
                                detrend=0,
                                event_id=None,
                                tmin=time_before,
                                tmax=time_after,
                                baseline=None,
                                preload=True,
                                verbose=False)

        # Update config with epochs
        self.update_config({'epoched': 1})


    def autoreject(self):
        """
        Performs autorejection.
        :return: None
        """
        # Check if the data is epoched
        if 'epoched' not in self.config.keys():
            print('Data not epoched; autoreject failed')
            return

        before_len = len(self.eegs[0])

        # Find bad epochs for stream 0
        ar0 = autor.AutoReject(n_interpolate=[0], verbose=False)  # allow interpolating 1-2 channels
        ar0.fit(self.eegs[0])
        reject_log0 = ar0.get_reject_log(self.eegs[0])

        # Find bad epochs for stream 1
        ar1 = autor.AutoReject(n_interpolate=[0], verbose=False)
        ar1.fit(self.eegs[1])
        reject_log1 = ar1.get_reject_log(self.eegs[1])

        # If an epoch is good for both streams, then mark as good
        good_epochs = np.all([reject_log1.bad_epochs == False, reject_log0.bad_epochs == False], axis=0)

        self.eegs[0] = ar0.transform(self.eegs[0].copy()[good_epochs])
        self.eegs[1] = ar1.transform(self.eegs[1].copy()[good_epochs])

        # Update config
        self.update_config({'autoreject': np.where(good_epochs)})
        print('Good epochs:', np.sum(good_epochs), 'out of', before_len)

    def identify_ICA(self):
        """
        Interactively identify ICAs in the stream for EOG removal
        :return: None
        """
        self.ica = []
        config = self.config
        for i, eeg in enumerate(self.eegs):
            ica = ICA(n_components=config['lowlands']['ICAparam']['n_components'],
                      method=config['lowlands']['ICAparam']['method'],
                      random_state=config['lowlands']['ICAparam']['random_state'])
            ica.fit(eeg)

            ica.plot_components(inst=eeg)
            self.eegs[i] = ica.apply(eeg)


    def update_config(self, new_info):
        """
        Updates the Recording config
        :param new_info: new config info
        :return: None
        """
        self.config.update(new_info)

    def path_to_filename(self, path):
        """
        :param path: Filename whose path we want
        :return: Path to filename
        """
        return os.path.basename(os.path.abspath(path))

    def plot(self, index, show=False):
        """
        :param index: The stream we want to plot
        :param show: Whether or not to show this plot
        :return: The figure object
        """
        fig = self.eegs[index].plot(scalings=dict(eeg=100),
                                    title="Subject " + repr(index),
                                    n_channels=len(self.config['channels']))  # improves aesthetics of plot

        # Only show this plot if we want it to.
        # If we show the plot, it will pause execution until we close the plot.
        if (show):
            plt.show()

        return fig

# Test change for first commit