import numpy as np
import codecs, json, os, random
from matplotlib import pyplot as plt
from read import Recording
from analysis import simple_corr, compute_single_freq, butter_lowpass_filter
from scipy.stats import ttest_ind

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

"""
real pairs and pseudo-pairs
real pairs: matched epochs
pseudo pairs: randomly selected epoch pairs

(1)
Collect the trials of the two experimental conditions in a single set.
(2)
Randomly draw as many trials from this combined data set as there were trials in condition 1 and place those trials into subset 1. Place the remaining trials in subset 2. 
The result of this procedure is called a random partition.
(3)
Calculate the test statistic on this random partition.
a. For every frequency bin, calculate the coherence for each of the two experimental conditions.
b. For every frequency bin, evaluate the coherence difference by means of a test statistic, such as the coherence -statistic.
c. Select all frequency bins whose coherence -statistic is larger than some threshold. For instance, this threshold can be some quantile of the normal distribution with expected value 0 and variance 1.
d. Cluster the selected frequency bins in connected sets on the basis of adjacency; neighboring frequency bins are clustered in the same set.
e. Calculate cluster-level statistics by taking the sum of the coherence -statistics within a cluster.
f. Take the maximum of the cluster-level statistics.
(4)
Repeat steps 2 and 3 a large number of times and construct a histogram of the test statistics.
(5)
From the test statistic that was actually observed and the histogram in step 4, calculate the proportion of random partitions that resulted in a larger test statistic than the observed one. 
This proportion is called the Monte Carlo p-value.
(6)
If the Monte Carlo p-value is smaller than the critical alpha-level (typically, 0.05), then conclude that the data in the two experimental conditions are significantly different.

"""

srate = 128
frequency = 30  # range 1-30 Hz
real_count = 0
fake_total = 78

"""
(1)(2)
Collect the trials of the two experimental conditions in a single set.
condition 1: real pairs, 78 trials
condition 2: pseudo pairs, 78 trials (randomly chosen)
"""
# generate pseudo pairs
fake_pairs = []
fake_count = 0
while fake_count < fake_total:
    a, b = random.sample(files, 2)
    while a.split('_')[1] == b.split('_')[1]:  # if from the same pair, shuffle until we get two files from different pairs
        a, b = random.sample(files, 2)
    fake_pairs.append([a,b])
    fake_count += 1

# define real pairs
real_pairs = np.split(np.array(files), 78)
real_pairs = [list(p) for p in real_pairs]

# construct set of real + fake data
all_pairs = real_pairs + fake_pairs
"""
(3)
Calculate the test statistic on this random partition.
"""
# a. For every frequency bin and channel bin, calculate the plv for each of the two experimental conditions.
# compute for all pairs plv: [n_all_pairs x 14 channels x 30 freq]
fake_all = np.empty((156, frequency, len(channels)))
for i, [a,b] in enumerate(all_pairs):
    #reading json files
    f = codecs.open(a, 'r', encoding='utf-8').read()
    sample0 = np.array(json.loads(f))
    f = codecs.open(b, 'r', encoding='utf-8').read()
    sample1 = np.array(json.loads(f))
    # f = codecs.open(infos[0], 'r', encoding='utf-8').read()
    # info = np.array(json.loads(f))
    data = [sample0, sample1]
    perm_corrs = []
    for freq in np.arange(1, frequency+1, 1):
        perm_corr = np.array(simple_corr(data, srate, freq, mode='plv', epoch_wise=True))
        perm_corrs.append(np.mean(perm_corr, axis=1))
    perm_corrs = np.array(perm_corrs)  # PLV matrix [n_freq x n_channels]
    fake_all[i] = perm_corrs  # update matrix


# b. For every frequency bin and channel bin, evaluate the plv difference by means of a test statistics.
# here we use t-statistics
group1 = random.sample(list(np.arange(0,156,1)), 78)
group2 = fake_all-group1
ttest_ind(group1, group2, axis=2)

# c. Select all frequency bins whose t-statistic is larger than some threshold.

# d. Cluster the selected frequency bins in connected sets on the basis of adjacency.

# e. Calculate cluster-level statistics by taking the sum of the coherence -statistics within a cluster.
# options: size of the cluster, sum(t-values) for the cluster







#
# ind = info[0:-1]
#
# power_corr = simple_corr(data, 128, 10, mode='power', epoch_wise=True)
#


#
#
# colors=14*['b']
# colors[4] = 'r'
# for i in range(14):
#     plt.plot(butter_lowpass_filter(plv_corr[i], 5,20,order), alpha=0.5, label=channels[i],color=colors[i])
#     # plt.scatter(ind, butter_lowpass_filter(corr[i], cutoff, fs, order), alpha=0.5)
#     #plt.plot(ind, corr[i], alpha=0.5)
# plt.legend()
#
# plt.plot(np.mean(plv_corr, axis=1))
# plt.plot(butter_lowpass_filter(np.mean(plv_corr, axis=0), 5, 20, order))
#
# plt.legend()
# subdata = [data[i][20:30, :, :] for i in range(2)]
# values = compute_single_freq(subdata, 128, 'proj', 10, 5)
# values = np.imag(values)
# plt.plot(np.concatenate(values[0]))
# plt.plot(np.concatenate(values[1]))
#
# plt.plot(np.arange(0, 10*130, 130), np.array(plv_corr[5][30:40])*10)
#
#
#
#
# plv_corr = np.array(plv_corr)
# plt.imshow(plv_corr, cmap='icefire',interpolation='hanning')
#
#
#
#
#
#
# """
# distance matrix
# """
# # TODO not funished.
# # thinking maybe we shouldn't cluster electrodes - because we have so few!!
# channel_graph = {'AF3': ['F7', 'F3'],
#                  'F7': ['AF3', 'F3', 'FC5'],
#                  'T7': ['FC5'],
#                  'F3': ['AF3', 'F7', 'FC5'],
#                  'FC5': ['F7', 'F3', 'FC5'],
#                  }
#
