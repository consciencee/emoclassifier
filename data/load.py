import model.params as params


import math

import numpy as np

import mne

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

    for line in contents:
        line = line[:-2]
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
            dataset.append(map(float, lineSplit[1:]))

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
    epoc_ch = ['af3', 'f7', 'f3', 'fc5', 't7', 'p7', 'o1', 'o2', 'p8', 't8', 'fc6', 'f4', 'f8', 'af4']
    scaling = {'eeg' : 1}
    info = mne.create_info(ch_names=epoc_ch, sfreq=sampling_freq, ch_types='eeg')

    samples = loadRaw(csv_name, ignoredIDs)
    samples = np.transpose(samples)
    samples *= 0.51

    mne_raw = mne.io.RawArray(samples, info)

    mne_raw.filter(l_freq=8, h_freq=40, n_jobs=4)

    #mne_raw.plot(n_channels=len(epoc_ch), scalings=scaling, show=True, title=csv_name, block=True)

    return np.transpose(mne_raw.get_data())


def getDataset(batchDim, filenames, labels, ignoredIDs = ()):

    dataset = []

    for i in range(len(labels) - 1):
        dataset += load(filenames[i], labels[i], batchDim, ignoredIDs)

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






