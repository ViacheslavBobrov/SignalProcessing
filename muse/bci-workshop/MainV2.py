# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:32:48 2019

@author: Cyrielle Albert
"""

import numpy as np  # Module that simplifies computations on matrices
import matplotlib.pyplot as plt  # Module used for plotting
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data

import bci_workshop_tools2 as BCIw  # functions for the workshop

#### POSTERIOR RYTHM IS ALPHA!!!!!!!!!!!

#FOR PLOTTING PUT SMR FIRST
bands = np.asarray([
    [4, 8, True],
    [12, 15, True]
    #[16, 30, False]

])
condition_mask = bands.T[-1]

if __name__ == "__main__":

    """ 1. CONNECT TO EEG STREAM """

    # Search for active LSL stream
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=10)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info, description, sampling frequency, number of channels
    info = inlet.info()
    description = info.desc()
    fs = int(info.nominal_srate())
    n_channels = info.channel_count()

    # Get names of all channels
    ch = description.child('channels').first_child()
    ch_names = [ch.child_value('label')]
    for i in range(1, n_channels):
        ch = ch.next_sibling()
        ch_names.append(ch.child_value('label'))

    print('Child channels names: ', ch_names)
    """ 2. SET EXPERIMENTAL PARAMETERS """

    # Length of the EEG data buffer (in seconds)
    # This buffer will hold last n seconds of data and be used for calculations
    buffer_length = 15  ###### Important

    # Length of the epochs used to compute the FFT (in seconds)
    epoch_length = 1

    # Amount of overlap between two consecutive epochs (in seconds)
    overlap_length = 0.8

    # Amount to 'shift' the start of each next consecutive epoch
    shift_length = epoch_length - overlap_length

    # Index of the channel (electrode) to be used
    # 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
    index_channel = [0, 1, 2, 3]
    # Name of our channel for plotting purposes
    ch_names = [ch_names[i] for i in index_channel]
    n_channels = len(index_channel)

    # Get names of features
    # ex. ['delta - CH1', 'pwr-theta - CH1', 'pwr-alpha - CH1',...]
    feature_names = BCIw.get_feature_names(ch_names)
    print('Feature names: ', feature_names)

    # Number of seconds to collect training data for (one class)
    training_length = 20

    # Number of frequency band 
    n_freq_band = len(bands)  # Theta SMR and Beta

    """ 3. RECORD TRAINING DATA """

    # Record data for mental activity 0

    print('\nStart initialisation')
    print('\nClose your eyes!\n')
    eeg_data1, timestamps1 = inlet.pull_chunk(
        timeout=training_length + 1, max_samples=fs * training_length)
    eeg_data1 = np.array(eeg_data1)[:, index_channel]
    print('Training data has been collected')

    # Divide data into epochs
    eeg_epochs1 = BCIw.epoch(eeg_data1, epoch_length * fs,
                             overlap_length * fs)

    print('Epochs shape: ', eeg_epochs1.shape)
    """ 4. COMPUTE FEATURES AND TRAIN CLASSIFIER """
    # Calibrate matrix
    feat_matrix1 = BCIw.compute_feature_matrix(eeg_epochs1, fs, bands)
    print('Feature matrix:', feat_matrix1.shape)
    xlen, ylen = np.shape(feat_matrix1)
    matrix = np.zeros((xlen, n_freq_band, n_channels))  #### 96 x 3 x 4
    calibration_list = []
    for i_epoch in range(xlen):
        matrix[i_epoch] = feat_matrix1[i_epoch].reshape(n_freq_band, n_channels)
    average_mat = np.zeros((n_freq_band, n_channels))
    for j_freq in range(n_freq_band):
        for k_channel in range(n_channels):
            average_mat[j_freq][k_channel] = np.mean(matrix[:, j_freq, k_channel],
                                                     axis=0)  ###avg of all epochs for each band of each electrode
        calibration_list.append(float(np.mean(average_mat[j_freq, :], axis=0)))
    print('average_mat:', average_mat.shape)
    print('calibration_list:', calibration_list)
    plt.plot(calibration_list[0])

    print('\nEnd of initialisation')

    """ 5. USE THE CLASSIFIER IN REAL-TIME"""

    # Initialize the buffers for storing raw EEG and decisions
    eeg_buffer = np.zeros((int(fs * buffer_length), n_channels))
    filter_state = None  # for use with the notch filter
    avgRTsmr_buffer = np.zeros((30, 1))
    plotter_avgRTsmr = BCIw.DataPlotter(30, ['realtime smr (avg)'])

    # The try/except structure allows to quit the while loop by aborting the
    # script with <Ctrl-C>
    print('Press Ctrl-C in the console to break the while loop.')

    try:
        while True:

            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=int(shift_length * fs))  #### 0.2 for appending new data to buffer
            ### eeg_data = n_samples x n_electrodes, timestamp = n_samples
            # Only keep the channel we're interested in
            ch_data = np.array(eeg_data)[:, index_channel]

            # Update EEG buffer
            eeg_buffer, filter_state = BCIw.update_buffer(  ### n_samples_per_buffer x n_channels
                eeg_buffer, ch_data, notch=True,
                filter_state=filter_state)

            """ 3.2 COMPUTE FEATURES AND CLASSIFY """
            # Get newest samples from the buffer
            data_epoch = BCIw.get_last_data(eeg_buffer,
                                            epoch_length * fs)

            # Compute features
            realtime_datas = BCIw.compute_feature_vector(data_epoch, fs, bands)
            realtime_datas = realtime_datas.reshape(n_freq_band, 4)
            avgRealtime = []
            for i_freq in range(n_freq_band):
                avgRealtime.append(np.mean(realtime_datas[i_freq, :], axis=0))
            #### Builder can obtain calibration list
            condition_satisfied = BCIw.analyse_binary(calibration_list, avgRealtime, condition_mask)
            if condition_satisfied:
                print('Ok')
            else:
                print('Need to relax')

            avgRTsmr_buffer, _ = BCIw.update_buffer(avgRTsmr_buffer, np.reshape(avgRealtime[0], (-1, 1)))

            """3.3 VISUALISATION"""

            plotter_avgRTsmr.update_plot(avgRTsmr_buffer)
            plt.pause(0.00001)



    except KeyboardInterrupt:

        print('Closed!')
