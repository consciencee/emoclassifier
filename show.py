from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import numpy as np

from data.load import loadRaw, loadFiltered

sampling_rate = 128

epoc_ch = ['af3', 'f7', 'f3', 'fc5', 't7', 'p7', 'o1', 'o2', 'p8', 't8', 'fc6', 'f4', 'f8', 'af4']

def saveSpectrogram(data, channel):

    channel_index = epoc_ch.index(channel)

    chan_data = np.transpose(data)[channel_index]

    fig = plt.figure()
    fig.suptitle('Spectrogram ' + str(channel))

    # axs_seq = [axs_grid[0, 0]]

    plt.specgram(chan_data, Fs=sampling_rate)

    fig.text(0.5, 0.04, 'Time [sec]', ha='center', va='center')  # x
    fig.text(0.06, 0.5, 'Frequency [Hz]', ha='center', va='center', rotation='vertical')  # y

    #plt.savefig('../spectrograms/' + str(dataset_label) + '/sample_' + str(sample_label) + '.png')
    #plt.close(fig)
    plt.show()

#fileName = "ssvep_data/ssvep1_eeg_.csv"
fileName = "ssvep_data/ssvep1_20hz_eeg_.csv"
#fileName = "ssvep_data/ssvep1_30Hz_eeg_.csv"

data = loadFiltered(fileName)
for channel_name in epoc_ch:
    saveSpectrogram(data, channel_name)


