"""
compare three time-frequency methods on an epoch of data
"""
import matplotlib.mlab as mlab
from matplotlib import pyplot as plt
import numpy as np
from read import Recording
from read import DataImportType

"""
initialization
"""
config = {}
path = '/Users/averybedows/Desktop/allthethings/MWM/Lowlands/15-08-21-15-52-48'

# Config and preprocessing params for lowlands dataset
config['lowlands'] = {'srate': 128,  # sampling rate
                      'channels':['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4'],  # channel names
                      'epoch': [-1.5, 1.5],  # epochs of 1s length #TODO changed to 3 sec
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
recording = Recording(path, config, 'lowlands', DataImportType.CSV)

# Preprocess the Recording
recording.filter()
recording.epoch()
recording.autoreject()

"""
trying out time-frequency methods
"""
x = recording.eegs[0].get_data()[10][5]  # some random data
freq = 10  # freq of interest is 10 Hz


# the matlab function equivalent. discrete-time Fourier transform with sliding window
complex_signal = mlab.specgram(
    x,
    NFFT=50,  # flexible. depending on frequency of interest
    Fs=128,
    window=mlab.window_hanning,
    noverlap= 49,  # moving window in a 1 sample step (so overlap length is one sample short of window size)
    pad_to=128, # making sure the resolution is 1 Hz
    mode='complex')  # mode can also be changed to 'complex'

plt.title('mlab.specgram')
plt.plot(np.real(complex_signal[0][freq]),alpha=0.6, label='real')  # real part
plt.plot(np.imag(complex_signal[0][freq]),alpha=0.6, label='imag')  # imaginary part
plt.plot(np.angle(complex_signal[0][freq]),alpha=0.3, label='angle')   # phase of the complex signal
plt.plot(np.abs(complex_signal[0][freq]),alpha=0.6, label='envelope/amplitude')  # envelope of the complex signal
plt.plot(np.abs(complex_signal[0][freq])*2,alpha=0.6, label='power = amplitude * 2')  # power
plt.legend()
plt.show()

# FIR filtering to get 10 Hz -11 Hz data, and then hilbert transform to get the analytic/complex signal
import scipy.signal as signal
# FIT filter design
cutoff=[10,11]
nyquist=128/2
order = int(128/freq*4)  # "For EEG signals, a useful rule of thumb is to ‘look at’
                    # about 4 to 5 cycles of the desired EEG rhythm."

b = signal.firwin(order, cutoff = [cutoff[0]/64,cutoff[1]/64],fs=128, window="hanning", pass_zero=False)
filtered = signal.lfilter(b, 1.0, x)
complex_signal = signal.hilbert(filtered)
plt.title('FIR filter and Hilbert transform')
plt.plot(np.real(complex_signal),alpha=0.6, label='real')  # real part
plt.plot(np.imag(complex_signal),alpha=0.6, label='imag')  # imaginary part
plt.plot(np.angle(complex_signal),alpha=0.3, label='angle')   # phase of the complex signal
plt.plot(np.abs(complex_signal),alpha=0.6, label='envelope/amplitude')  # envelope of the complex signal
plt.plot(np.abs(complex_signal)*2,alpha=0.6, label='power = amplitude * 2')  # power
plt.legend()
plt.show()


# short-time Fourier transform
f, t, Zxx = signal.stft(x, 128, nperseg=50, noverlap=49, window='hanning', nfft=128)
complex_signal = Zxx[10]
plt.title('FIR filter and Hilbert transform')
plt.plot(np.real(complex_signal),alpha=0.6, label='real')  # real part
plt.plot(np.imag(complex_signal),alpha=0.6, label='imag')  # imaginary part
plt.plot(np.angle(complex_signal),alpha=0.3, label='angle')   # phase of the complex signal
plt.plot(np.abs(complex_signal),alpha=0.6, label='envelope/amplitude')  # envelope of the complex signal
plt.plot(np.abs(complex_signal)*2,alpha=0.6, label='power = amplitude * 2')  # power
plt.legend()
plt.show()


