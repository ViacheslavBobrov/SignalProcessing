import os
import random
from time import time, strftime, gmtime

import numpy as np
import pandas as pd
from playsound import playsound
from psychopy import core


def present_audio_sounds(experiment_number, record_duration=120):
    iti = 0.3
    jitter = 0.2
    recording_path = os.path.join("data",
                                  "%s_a_signal_timestamps_%s_%ssec.csv" % (strftime("%Y-%m-%d-%H.%M.%S", gmtime()),
                                                                         experiment_number, record_duration,))

    start_time = time()
    signal_timestamps = []

    while True:
        inter_trial_wait = iti + np.random.rand() * jitter
        core.wait(inter_trial_wait)
        n_normal_signals = random.randint(4, 9)
        for _ in range(n_normal_signals):
            playsound('audio/odd_signal2.mp3')
            core.wait(0.4)
        playsound('audio/normal_signal.mp3')
        odd_signal_time = time()
        signal_timestamps.append(odd_signal_time)
        print(odd_signal_time)
        core.wait(0.8)
        if odd_signal_time - start_time > record_duration:
            break
    pd.DataFrame(signal_timestamps).to_csv(recording_path)
