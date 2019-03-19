import numpy as np
import codecs, json, os
from matplotlib import pyplot as plt
from read import Recording
from analysis import simple_corr, compute_single_freq, butter_lowpass_filter

"""
simple analysis
"""
jsonpath = './json_files'
infopath = './info_files'
channels =['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']


files = []
infos = []
for file in os.listdir(jsonpath):
    files.append(os.path.join(jsonpath, file))
    infos.append(os.path.join(infopath, file))

#reading json files
f = codecs.open(files[0], 'r', encoding='utf-8').read()
sample0 = np.array(json.loads(f))
f = codecs.open(files[1], 'r', encoding='utf-8').read()
sample1 = np.array(json.loads(f))
f = codecs.open(infos[0], 'r', encoding='utf-8').read()
info = np.array(json.loads(f))



"""
real pairs and pseudo-pairs
real pairs: matched epochs
pseudo pairs: randomly selected epoch pairs
"""


ind = info[0:-1]
data = [sample0, sample1]
plv_corr = simple_corr(data, 128, 10, mode='plv', epoch_wise=True)
power_corr = simple_corr(data, 128, 10, mode='power', epoch_wise=True)
order = 6
fs = 50.0
cutoff = 2

colors=14*['b']
colors[4] = 'r'
for i in range(14):
    plt.plot(butter_lowpass_filter(plv_corr[i], 5,20,order), alpha=0.5, label=channels[i],color=colors[i])
    # plt.scatter(ind, butter_lowpass_filter(corr[i], cutoff, fs, order), alpha=0.5)
    #plt.plot(ind, corr[i], alpha=0.5)
plt.legend()

plt.plot(np.mean(plv_corr, axis=1))
plt.plot(butter_lowpass_filter(np.mean(plv_corr, axis=0), 5, 20, order))

plt.legend()
subdata = [data[i][20:30, :, :] for i in range(2)]
values = compute_single_freq(subdata, 128, 'proj', 10, 5)
values = np.imag(values)
plt.plot(np.concatenate(values[0]))
plt.plot(np.concatenate(values[1]))

plt.plot(np.arange(0, 10*130, 130), np.array(plv_corr[5][30:40])*10)




plv_corr = np.array(plv_corr)
plt.imshow(plv_corr, cmap='icefire',interpolation='hanning')






