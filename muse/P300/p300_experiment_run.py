import os
from threading import Thread
from time import strftime, gmtime

from muselsl import record
from audio_generator import present_audio_sounds

duration = 120
experiment_number = 1
recording_path = os.path.join("data", "recording_%s_%ssec_%s.csv" % (
    experiment_number, duration, strftime("%Y-%m-%d-%H.%M.%S", gmtime())))
print("Experiment %s has been started" % experiment_number)
recording_process = Thread(target=record, args=(duration, recording_path))
signaling_process = Thread(target=present_audio_sounds, args=(experiment_number, duration))
recording_process.start()
signaling_process.start()

recording_process.join()
signaling_process.join()
print("Experiment %s has been finished" % experiment_number)
