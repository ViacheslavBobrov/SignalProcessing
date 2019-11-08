# -*- coding: utf-8 -*-
"""
BCI Workshop Auxiliary Tools

Created on Fri May 08 15:34:59 2015

@author: Cassani
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

NOTCH_B, NOTCH_A = butter(4, np.array([55, 65]) / (256 / 2), btype='bandstop')  ### WRONG FILTER, NEEDS TO REJECT


def analyse_binary(calib_list, datas, condition_mask):
    condition_results = np.empty(len(calib_list), dtype=bool)
    for i in range(len(calib_list)):
        condition_results[i] = calib_list[i] < datas[i]
    condition_satisfied = np.array_equal(condition_results, condition_mask)
    return condition_satisfied


def analyse_float(calib_list, datas, condition_mask):
    deviations = np.empty(len(calib_list), dtype=bool)
    for i in range(len(calib_list)):
        deviations[i] = calib_list[i] - datas[i]
    deviations *= condition_mask
    return deviations.mean()


def plot_multichannel(data, params=None):
    """Create a plot to present multichannel data.

    Args:
        data (numpy.ndarray):  Multichannel Data [n_samples, n_channels]
        params (dict): information about the data acquisition device

    TODO Receive labels as arguments
    """
    fig, ax = plt.subplots()

    n_samples = data.shape[0]
    n_channels = data.shape[1]

    if params is not None:
        fs = params['sampling frequency']
        names = params['names of channels']
    else:
        fs = 1
        names = [''] * n_channels

    time_vec = np.arange(n_samples) / float(fs)

    data = np.fliplr(data)
    offset = 0
    for i_channel in range(n_channels):
        data_ac = data[:, i_channel] - np.mean(data[:, i_channel])
        offset = offset + 2 * np.max(np.abs(data_ac))
        ax.plot(time_vec, data_ac + offset, label=names[i_channel])

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    plt.legend()
    plt.draw()


def epoch(data, samples_epoch, samples_overlap=0):
    """Extract epochs from a time series.

    Given a 2D array of the shape [n_samples, n_channels]
    Creates a 3D array of the shape [wlength_samples, n_channels, n_epochs]

    Args:
        data (numpy.ndarray or list of lists): data [n_samples, n_channels]
        samples_epoch (int): window length in samples
        samples_overlap (int): Overlap between windows in samples

    Returns:
        (numpy.ndarray): epoched data of shape
    """

    if isinstance(data, list):
        data = np.array(data)

    n_samples, n_channels = data.shape

    samples_shift = samples_epoch - samples_overlap

    n_epochs = int(np.floor((n_samples - samples_epoch) / float(samples_shift)) + 1)

    # Markers indicate where the epoch starts, and the epoch contains samples_epoch rows
    markers = np.asarray(range(0, n_epochs + 1)) * samples_shift
    markers = markers.astype(int)

    # Divide data in epochs
    epochs = np.zeros((samples_epoch, n_channels, n_epochs))

    for i in range(0, n_epochs):
        epochs[:, :, i] = data[markers[i]:markers[i] + samples_epoch, :]

    return epochs


def compute_feature_vector(eegdata, fs, bands):
    """Extract the features from the EEG.

    Args:
        eegdata (numpy.ndarray): array of dimension [number of samples,
                number of channels]
        fs (float): sampling frequency of eegdata

    Returns:
        (numpy.ndarray): feature matrix of shape [number of feature points,
            number of different features]
    """
    # 1. Compute the PSD
    winSampleLength, nbCh = eegdata.shape

    # Apply Hamming window
    w = np.hamming(winSampleLength)
    dataWinCentered = eegdata - np.mean(eegdata, axis=0)  # Remove offset
    dataWinCenteredHam = (dataWinCentered.T * w).T

    NFFT = nextpow2(winSampleLength)
    Y = np.fft.fft(dataWinCenteredHam, n=NFFT, axis=0) / winSampleLength
    PSD = 2 * np.abs(Y[0:int(NFFT / 2), :])
    f = fs / 2 * np.linspace(0, 1, int(NFFT / 2))

    # SPECTRAL FEATURES
    # Average of band powers
    feature_vector = compute_band_means(f, bands, PSD)  #### 12 (3 bands for 4 electrodes

    feature_vector = np.log10(feature_vector)

    return feature_vector


def compute_band_means(frequencies_axis, bands, PSD):
    n_bands = len(bands)
    n_channels = PSD.shape[1]
    band_means = np.zeros(n_bands * n_channels)

    for i in range(n_bands):
        lower_bound, upper_bound = bands[i, 0], bands[i, 1]
        ind_band, = np.where((frequencies_axis >= lower_bound) & (frequencies_axis < upper_bound))
        band_means[i * n_channels:(i + 1) * n_channels] = np.mean(PSD[ind_band, :], axis=0)

    return band_means.ravel()


def nextpow2(i):
    """
    Find the next power of 2 for number i
    """
    n = 1
    while n < i:
        n *= 2
    return n


def compute_feature_matrix(epochs, fs, bands):
    """
    Call compute_feature_vector for each EEG epoch
    """
    n_epochs = epochs.shape[2]
    print(n_epochs)
    for i_epoch in range(n_epochs):
        if i_epoch == 0:
            feat = compute_feature_vector(epochs[:, :, i_epoch], fs, bands).T
            feature_matrix = np.zeros((n_epochs, feat.shape[0]))  # Initialize feature_matrix

        feature_matrix[i_epoch, :] = compute_feature_vector(epochs[:, :, i_epoch], fs, bands).T

    return feature_matrix


def get_feature_names(ch_names):
    """Generate the name of the features.

    Args:
        ch_names (list): electrode names

    Returns:
        (list): feature names
    """
    bands = ['delta', 'theta', 'alpha', 'beta']

    feat_names = []
    for band in bands:
        for ch in range(len(ch_names)):
            feat_names.append(band + '-' + ch_names[ch])

    return feat_names


def update_buffer(data_buffer, new_data, notch=False, filter_state=None):
    """
    Concatenates "new_data" into "data_buffer", and returns an array with
    the same size as "data_buffer"
    """
    if new_data.ndim == 1:
        new_data = new_data.reshape(-1, data_buffer.shape[1])

    if notch:
        if filter_state is None:
            filter_state = np.tile(lfilter_zi(NOTCH_B, NOTCH_A),
                                   (data_buffer.shape[1], 1)).T
        new_data, filter_state = lfilter(NOTCH_B, NOTCH_A, new_data, axis=0,
                                         zi=filter_state)

    new_buffer = np.concatenate((data_buffer, new_data), axis=0)
    new_buffer = new_buffer[new_data.shape[0]:, :]

    return new_buffer, filter_state


def get_last_data(data_buffer, newest_samples):
    """
    Obtains from "buffer_array" the "newest samples" (N rows from the
    bottom of the buffer)
    """
    new_buffer = data_buffer[(data_buffer.shape[0] - newest_samples):, :]

    return new_buffer


class DataPlotter():
    """
    Class for creating and updating a line plot.
    """

    def __init__(self, nbPoints, chNames, fs=None, title=None):
        """Initialize the figure."""

        self.nbPoints = nbPoints
        self.chNames = chNames
        self.nbCh = len(self.chNames)

        self.fs = 1 if fs is None else fs
        self.figTitle = '' if title is None else title

        data = np.empty((self.nbPoints, 1)) * np.nan
        self.t = np.arange(data.shape[0]) / float(self.fs)

        # Create offset parameters for plotting multiple signals
        self.yAxisRange = 100
        self.chRange = self.yAxisRange / float(self.nbCh)
        self.offsets = np.round((np.arange(self.nbCh) + 0.5) * (self.chRange))

        # Create the figure and axis
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_yticks(self.offsets)
        self.ax.set_yticklabels(self.chNames)

        # Initialize the figure
        self.ax.set_title(self.figTitle)

        self.chLinesDict = {}
        for i, chName in enumerate(self.chNames):
            self.chLinesDict[chName], = self.ax.plot(
                self.t, data + self.offsets[i], label=chName)

        self.ax.set_xlabel('Time')
        self.ax.set_ylim([0, self.yAxisRange])
        self.ax.set_xlim([np.min(self.t), np.max(self.t)])

        plt.show()

    def update_plot(self, data):
        """ Update the plot """

        data = data - np.mean(data, axis=0)
        std_data = np.std(data, axis=0)
        std_data[np.where(std_data == 0)] = 1
        data = data / std_data * self.chRange / 5.0

        for i, chName in enumerate(self.chNames):
            self.chLinesDict[chName].set_ydata(data[:, i] + self.offsets[i])

        self.fig.canvas.draw()

    def clear(self):
        """ Clear the figure """

        blankData = np.empty((self.nbPoints, 1)) * np.nan

        for i, chName in enumerate(self.chNames):
            self.chLinesDict[chName].set_ydata(blankData)

        self.fig.canvas.draw()

    def close(self):
        """ Close the figure """

        plt.close(self.fig)
