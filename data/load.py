import model.params as params

from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt

import math

import numpy as np

import mne

epoc_ch = ['af3', 'f7', 'f3', 'fc5', 't7', 'p7', 'o1', 'o2', 'p8', 't8', 'fc6', 'f4', 'f8', 'af4']

def load(csv_name, dstLabel, batchDim, ignoredIDs = (), cut = -1):

    dataset = loadFiltered(csv_name, ignoredIDs)
    #dataset = dataset[4*128:]
    #if cut > 0:
    #    dataset = dataset[:-cut*128]
    bins = sliceNumBatches(dataset, batchDim)
    #bins = sliceSequentialBins(dataset, batchDim, 150)

    return [list([dstLabel, x]) for x in bins]

def loadRaw(csv_name, ignoredIDs = ()):
    csv_file = open(csv_name, "r")

    if(not csv_file):
        print("error loading file")
        return

    contents = csv_file.readlines()
    dataset = []
    currentIgnore = False

    for line in contents:
        line = line[:-1] # -2 for windows /r/n
        if line == "":
            continue
        if line in params.emotionLabels:
            continue
        if line[0:7] == "Session":
            print(line)
            currentIgnore = line in ignoredIDs
            continue

        if not currentIgnore:
            lineSplit = line.split(",")
            # lineSplit[0] is time
            dataset.append(list(map(float, lineSplit[1:])))

    return dataset


def sliceTimeBatches(dataset, size, dim = "seconds"): #seconds or millis

    multiplier = 1

    if(dim == "seconds"):
        multiplier = 1000

    batchBegin = dataset[0][1]
    batchEnd = batchBegin + size * multiplier

def sliceNumBatches(dataset, size):

    nBatches = len(dataset) / size
    finalDim = nBatches*size

    dimDelta = len(dataset) - finalDim

    if int(math.ceil(dimDelta/2.0)):
        dataset = dataset[int(math.ceil(dimDelta/2.0)):]
    if int(-math.floor(dimDelta/2.0)):
        dataset = dataset[:int(-math.floor(dimDelta/2.0))]

    return np.array_split(dataset, len(dataset)/size)

def sliceSequentialBins(dataset, size, strides = 1):

    print("slicing size=", size, ", strides=", strides)
    bins = []

    for i in range(0, len(dataset) - size - 1, strides):
        bins.append(dataset[i:i+size])

    print("slice end")

    return np.array(bins)

def loadFiltered(csv_name, ignoredIDs = ()):

    sampling_freq = 128
    scaling = {'eeg' : 1}
    info = mne.create_info(ch_names=epoc_ch, sfreq=sampling_freq, ch_types='eeg')

    samples = loadRaw(csv_name, ignoredIDs)
    samples = np.transpose(samples)
    #samples *= 0.51

    print(samples.shape)

    mne_raw = mne.io.RawArray(samples, info)

    mne_raw.filter(l_freq=0.5, h_freq=4, n_jobs=4)

    #mne_raw.plot(n_channels=len(epoc_ch), scalings=scaling, show=True, title=csv_name, block=True)

    return np.transpose(mne_raw.get_data())

def saveSpectrogram(data, dataset_label, sample_label, sampling_rate):

    chan_data = np.transpose(data)

    fig = plt.figure()
    axs_grid = fig.subplots(3, 5, sharex='col', sharey='row')

    axs_seq = [axs_grid[0, 0], axs_grid[0, 1], axs_grid[0, 2], axs_grid[0, 3], axs_grid[0, 4],
               axs_grid[1, 0], axs_grid[1, 1], axs_grid[1, 2], axs_grid[1, 3], axs_grid[1, 4],
               axs_grid[2, 0], axs_grid[2, 1], axs_grid[2, 2], axs_grid[2, 3]]
    fig.suptitle('Spectrograms_' + str(dataset_label) + '_' + str(sample_label))

    # for ax in axs.flat:
    #     ax.label_outer()

    for channel_index in range(0, len(epoc_ch) - 1):
        axs_seq[channel_index].specgram(chan_data[channel_index], Fs=sampling_rate)
        axs_seq[channel_index].set_title(epoc_ch[channel_index])

    fig.text(0.5, 0.04, 'Time [sec]', ha='center', va='center')  # x
    fig.text(0.06, 0.5, 'Frequency [Hz]', ha='center', va='center', rotation='vertical')  # y

    plt.savefig('../spectrograms/' + str(dataset_label) + '/sample_' + str(sample_label) + '.png')
    plt.close(fig)
    #plt.show()

def saveBatchesSpectrogram(data, emotion_label):
    sampling_rate = 128 #samples/second
    batches = np.array(list(map(lambda arr: arr[1], data)))
    binIndex = 0

    for batch in batches:
        saveSpectrogram(batch, emotion_label, binIndex, sampling_rate)
        binIndex = binIndex + 1



def getDataset(batchDim, filenames, labels, ignoredIDs = ()):

    dataset = []

    for i in range(len(labels)):
        data = load(filenames[i], labels[i], batchDim, ignoredIDs)
        dataset += data
        # save loaded as spectrogram img
        saveBatchesSpectrogram(data, labels[i])

    np.random.shuffle(dataset)

    nSamples = len(dataset)

    nTrain = int(math.ceil(nSamples * 0.8))

    data_train_samples = [np.hstack(dataitem[1]) for dataitem in (dataset[:nTrain])]
    data_train_labels = [dataitem[0] for dataitem in (dataset[:nTrain])]

    print(data_train_labels)

    data_test_samples = [np.hstack(dataitem[1]) for dataitem in (dataset[nTrain:])]
    data_test_labels = [dataitem[0] for dataitem in (dataset[nTrain:])]
    print(data_test_labels)

    return (len(data_train_samples), len(data_test_samples)), \
            (np.hstack(data_train_samples), np.hstack(data_train_labels)), \
           (np.hstack(data_test_samples), np.hstack(data_test_labels))






