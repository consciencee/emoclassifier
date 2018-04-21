import mne

import data.load as rawdata
import matplotlib

import numpy as np



def loadFiltered(csv_name, ignoredIDs = ()):

    sampling_freq = 128
    epoc_ch = ['af3', 'f7', 'f3', 'fc5', 't7', 'p7', 'o1', 'o2', 'p8', 't8', 'fc6', 'f4', 'f8', 'af4']
    scaling = {'eeg' : 1}
    info = mne.create_info(ch_names=epoc_ch, sfreq=sampling_freq, ch_types='eeg')

    samples = rawdata.loadRaw(csv_name, ignoredIDs)
    samples = np.transpose(samples)
    samples /= samples.max()

    mne_raw = mne.io.RawArray(samples, info)

    mne_raw.filter(l_freq=8, h_freq=40, n_jobs=4)



    mne_raw.plot(n_channels=len(epoc_ch), scalings=scaling, show=True, title='em1', block=True)

