import os
import math
from datetime import datetime
from datetime import timedelta

import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

ROOT_PATH = '/drive/MyDrive'

def to_seconds(dt):
    dt = datetime.fromisoformat(dt)
    t = timedelta(minutes = dt.minute, seconds = dt.second, microseconds = dt.microsecond)
    return t.total_seconds()

dataset = pd.read_csv("conan(1).csv")

intervals = list(map(lambda x: int(x.split('.')[0]), os.listdir(os.path.join(ROOT_PATH, 'keypoints'))))

for interval in intervals:
    num_of_features = np.array(np.load(os.path.join(ROOT_PATH, 'keypoints', str(interval) + '.npy'), allow_pickle = True)[0]).shape[0]
    mfcc_features = np.empty((0, 13))

    record = dataset[dataset['interval_id'] == interval]
    start_time = to_seconds(record.iloc[0]['start_time'])
    duration = to_seconds(record.iloc[0]['end_time']) - start_time
    file_name = record.iloc[0]['video_fn'][:-4] + '.mp3'

    audio_file = os.path.join(ROOT_PATH, 'audio data', file_name)

    signal, sr = librosa.load(audio_file, sr = None,
                                  offset = start_time,
                                  duration = duration)

    temp = np.transpose(librosa.feature.mfcc(y = signal, n_mfcc = 13, sr = sr, hop_length = 441))

    for i in range(0, num_of_features):
        j = 5 * i
        mfcc_features = np.vstack((mfcc_features, np.mean(temp[j : j+5], axis = 0)))

    np.save(os.path.join(ROOT_PATH, 'mfccs', str(interval)), mfcc_features)