import numpy as np
import codecs, json, os
from matplotlib import pyplot as plt
from random import shuffle
from read import Recording
from analysis import simple_corr, compute_single_freq, butter_lowpass_filter
import dickey_fuller as df

"""
simple analysis
"""
jsonpath = './json_files'
infopath = './info_files'
channels =['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

'''
files = []
infos = []
for file in os.listdir(jsonpath):
    files.append(os.path.join(jsonpath, file))
    infos.append(os.path.join(infopath, file))

#reading json files
files.sort()
infos.sort()

f = codecs.open(files[0], 'r', encoding='utf-8').read()
sample0 = np.array(json.loads(f))
f = codecs.open(files[1], 'r', encoding='utf-8').read()
sample1 = np.array(json.loads(f))
f = codecs.open(infos[0], 'r', encoding='utf-8').read()
info = np.array(json.loads(f))
'''


"""
real pairs and pseudo-pairs
real pairs: matched epochs
pseudo pairs: randomly selected epoch pairs
"""
'''
ind = info[0:-1]
data = [sample0, sample1]

plv = simple_corr(data, 128, 10, mode='plv', epoch_wise=True)
power_corr = simple_corr(data, 128, 10, mode='power', epoch_wise=True)
order = 6
fs = 50.0
cutoff = 2

# Plot PLV over linearly ordered epochs (each one occurs later than the one previously, but at an unspecified time)
plt.title("PLV over linearly ordered epochs")
for channel in range(len(channels)):
    plt.plot(plv[channel])
plt.legend(channels)
#plt.show()

# Plot PLV over linearly ordered epochs (each one occurs later than the one previously, but at an unspecified time).
# Lowpass filter before plotting using a 6th-Butterworth with a 2Hz cutoff
colors=14*['b']
colors[4] = 'r'
for channel in range(len(channels)):
    plt.plot(butter_lowpass_filter(plv[channel], 5, 20, order), alpha=0.5, label=channels[channel],color=colors[channel])
plt.legend()
#plt.show()


plt.legend()
subdata = [data[i][20:30, :, :] for i in range(2)]
values = compute_single_freq(subdata, 128, 'proj', 10, 5)
values = np.imag(values)
plt.plot(np.concatenate(values[0]))
plt.plot(np.concatenate(values[1]))
#plt.show()

plt.plot(np.arange(0, 10*130, 130), np.array(plv[5][30:40])*10)
#plt.show()

# plv_corr = np.array(plv_corr)
# plt.imshow(plv_corr, cmap='icefire',interpolation='hanning')
'''








