import os
import random
from time import time, strftime, gmtime

import numpy as np
import pandas as pd
from playsound import playsound
from psychopy import core, visual


def draw_signal(mywin, grating, fixation, angle, wait_time):
    grating.ori = angle
    grating.draw()
    fixation.draw()
    mywin.flip()
    core.wait(wait_time)
    fixation.draw()
    mywin.flip()


def present_visual_signals(experiment_number, record_duration=120):
    iti = 0.4
    soa = 0.3
    jitter = 0.2
    recording_path = os.path.join("data",
                                  "%s_v_signal_timestamps_%s_%ssec.csv" % (strftime("%Y-%m-%d-%H.%M.%S", gmtime()),
                                                                         experiment_number, record_duration,))

    start_time = time()
    signal_timestamps = []

    mywin = visual.Window([1920, 1080], monitor="testMonitor", units="deg", fullscr=True)
    grating = visual.GratingStim(win=mywin, mask='circle', size=20, sf=2)
    fixation = visual.GratingStim(win=mywin, size=0.2, pos=[0, 0], sf=0,
                                  rgb=[1, 0, 0])

    while True:
        inter_trial_wait = iti + np.random.rand() * jitter
        core.wait(inter_trial_wait)

        odd_signal_probability = np.random.rand()
        signal_time = time()
        if odd_signal_probability > 0.9:
            signal_timestamps.append(signal_time)
            print(signal_time)
            draw_signal(mywin, grating, fixation, 180, soa)
        else:
            draw_signal(mywin, grating, fixation, 90, soa)
        if signal_time - start_time > record_duration:
            break
    pd.DataFrame(signal_timestamps).to_csv(recording_path)
