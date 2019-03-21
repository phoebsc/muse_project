"""
implemented: power/envelope correlation, projected power correlation, PLV
to implement: network analysis, granger causality
"""

import numpy as np
import scipy.signal as signal
from scipy.signal import butter, lfilter


def simple_corr(data, srate, freq, mode='psd', epoch_wise=True):
    """
    compute channel-wise, epoch-to-epoch comparison analysis on the two subjects.
    :param Recording: Recording object
    :param freq: int or list. a specific frequency or a frequency range of interest
    :param mode: {'envelope': envelope correlation,
                  'power': power correlation,
                  'plv': phase-locked value,
                  'proj': projected power correlation}

    :return: cor: either a list or an int object.
    """

    # extract data from Recording object.
    # Data consists of two lists of np.array (n_epochs, n_channels, sampling_rate)
    n_channel = data[0].shape[1]
    n_epoch = min(data[0].shape[0], data[1].shape[0])
    n_samp = data[0].shape[2]
    all_channels = []
    # compute correlation coefficient for all symmetrical channel pairs
    for channel in range(n_channel):
        values = compute_single_freq(data, n_samp, mode, freq, channel)

        # generate a list of per-epoch end values
        if epoch_wise:
            if mode in ['envelope', 'power']:
                result = [np.corrcoef(values[:, epoch, :])[0][1]
                          for epoch in range(n_epoch)]
            elif mode is 'plv':
                n_samples = values.shape[2]
                result = [abs(np.sum(np.exp(1j*(values[0, epoch, :]-values[1, epoch, :])))) / n_samples
                          for epoch in range(n_epoch)]
            elif mode is 'proj':
                result = [_proj_power_corr(values[0, epoch, :], values[1, epoch, :])
                          for epoch in range(n_epoch)]
        # generate a single correlation value
        else:
            strands = [np.concatenate(values[n]) for n in range(2)]  # concatenate values from all epochs
            if mode in ['envelope', 'power']:
                result = np.corrcoef(strands)[0][1]  # generate a corr coef overall
            elif mode is 'plv':
                n_samples = len(strands[0])
                result = abs(np.sum(np.exp(1j*(strands[0]-strands[1])))) / n_samples

        all_channels.append(result)

    return all_channels



def compute_single_freq(data, Fs, mode, freq, which_channel, plot=False):
    """

    :param data: a list of two arrays 2 x [n_epochs x n_channels x sampling_rate]
    :param mode: str analysis mode
    :param freq: frequency of interest
    :param which_channel:
    :return: values [2 x n_epochs x sampling_rate]
    """
    window_size = min(5 * freq, Fs)  # window size is frequency dependent: max(5*freq, sampling rate)
    n_epoch = min(data[0].shape[0], data[1].shape[0])

    # short-time Fourier transform

    complex_signal = np.array([[signal.stft(
            data[subject][epoch][which_channel],
            nperseg=window_size,
            nfft=Fs,  # equivalent to "pad_to". make sure the freq resolution is 1 Hz
            window='hanning',
            noverlap=window_size - 1)[2][freq]  # general output from stft is "f,t,Zxx", we only need Zxx[freq]
                   for epoch in range(n_epoch)  # for every epoch
                   ]
                  for subject in range(2)  # for each subject
                  ])

    # compute values
    if mode == 'envelope':
        values = np.abs(complex_signal)  # envelope
    elif mode == 'plv':
        values = np.angle(complex_signal)  # time-domain angle
    elif mode == 'power':
        values = np.abs(complex_signal)*2  # power
    elif mode == 'proj':
        values = complex_signal  # special case: projected power correlation is calculated later
    elif mode == 'imag':
        values = np.imag(complex_signal)  # imaginary part of the analytic signal
    return values


def _proj_power_corr(X, Y):
    # compute power proj corr
    X_abs = np.abs(X)
    Y_abs = np.abs(Y)

    X_unit = X / X_abs
    Y_unit = Y / Y_abs

    X_abs_norm = (X_abs - np.nanmean(X_abs)) / np.nanstd(X_abs)
    Y_abs_norm = (Y_abs - np.nanmean(Y_abs)) / np.nanstd(Y_abs)

    X_ = X_abs / np.nanstd(X_abs)
    Y_ = Y_abs / np.nanstd(Y_abs)

    X_z = X_ * X_unit
    Y_z = Y_ * Y_unit
    projX = np.imag(X_z * np.conjugate(Y_unit))
    projY = np.imag(Y_z * np.conjugate(X_unit))

    projX_norm = (projX - np.nanmean(projX)) / np.nanstd(projX)
    projY_norm = (projY - np.nanmean(projY)) / np.nanstd(projY)

    proj_corr = (np.nanmean(projX_norm * Y_abs_norm) + np.nanmean(projY_norm * X_abs_norm)) / 2
    return proj_corr

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


