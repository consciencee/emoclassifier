import model.params as params

import math

import numpy as np

import mne

def load(csv_name, dstLabel, batchDim, ignoredIDs = ()):

    dataset = loadFiltered(csv_name, ignoredIDs)
    bins = sliceNumBatches(dataset, batchDim)

    return [list([dstLabel, x]) for x in bins]

def loadRaw(csv_name, ignoredIDs = ()):
    csv_file = open(csv_name, "r")

    if(not csv_file):
        print "error loading file"
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
            print line
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


def getDataset(batchDim):

    dataset = []

    dataset += load("../samples/Alexey/2/Alexey1_2_eeg_log.csv", 1, batchDim, ("Session1"))
    dataset += load("../samples/Alexey/2/Alexey2_2_eeg_log.csv", 2, batchDim)
    dataset += load("../samples/Alexey/2/Alexey3_2_eeg_log.csv", 3, batchDim, ("Session4"))
    dataset += load("../samples/Alexey/2/Alexey4_2_eeg_log.csv", 4, batchDim)#[70:95]
    dataset += load("../samples/Alexey/2/Alexey5_2_eeg_log.csv", 5, batchDim)
    dataset += load("../samples/Alexey/2/Alexey6_2_eeg_log.csv", 0, batchDim)#[70:95]

    np.random.shuffle(dataset)

    #dataset = dataset[0:50]

    nSamples = len(dataset)

    nTrain = int(math.ceil(nSamples * 0.8))

    data_train_samples = [np.hstack(dataitem[1]) for dataitem in (dataset[:nTrain])]
    data_train_labels = [dataitem[0] for dataitem in (dataset[:nTrain])]

    print data_train_labels

    data_test_samples = [np.hstack(dataitem[1]) for dataitem in (dataset[nTrain:])]
    data_test_labels = [dataitem[0] for dataitem in (dataset[nTrain:])]
    print data_test_labels

    return (len(dataset[:nTrain]), len(dataset[nTrain:])), \
            (np.hstack(data_train_samples), np.hstack(data_train_labels)), \
           (np.hstack(data_test_samples), np.hstack(data_test_labels))



