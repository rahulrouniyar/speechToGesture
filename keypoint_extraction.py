# From Python
# It requires OpenCV installed for Python
# from logging import captureWarnings
import sys
import cv2
import os
from datetime import datetime, timedelta
from sys import platform
import argparse

import numpy as np
import pandas as pd

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
#---------------
ROOT_PATH = '../../../../drive/MyDrive'

def to_seconds(dt):
    dt = datetime.fromisoformat(dt)
    t = timedelta(minutes = dt.minute, seconds = dt.second, microseconds = dt.microsecond)
    return t.total_seconds()

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e


    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000241.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"
    # params["face"] = True
    params["hand"] = True

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image

    dataset = pd.read_csv('conan.csv',sep=",", encoding='cp1252')
    # print(dataset)

    sample_rate = 20    #fps

    #duration of each sampled frame
    sample_time = 1 / sample_rate

    videos = []
    with open('video_name.txt', 'r', encoding = 'utf-8') as f:
        for i in range(100):
            videos.append(str(f.readline()[:-1]))

    datum = op.Datum()
    i = 85
    # #iterate through first 100 videos
    for video_fn in dataset['video_fn'].unique()[i:i+1]:
        file_name = videos[i]
        video = cv2.VideoCapture(os.path.join(ROOT_PATH, 'video data', file_name))

        width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        vid_dim = np.array([width, height])

        # print(width, height)
        for index, row in dataset[dataset['video_fn'] == video_fn].iterrows():
            start_time = to_seconds(row['start_time'])
            end_time = to_seconds(row['end_time'])
            t = start_time
            
            keypoints_seq = np.empty((0, 25, 2))
            lh_keypoints_seq = np.empty((0, 21, 2))
            rh_keypoints_seq = np.empty((0, 21, 2))

            bad_frame = False

            while(t <= end_time):
                video.set(cv2.CAP_PROP_POS_MSEC, 1000 * t)

                ret, frame = video.read()

                if ret == False:
                    break

                datum.cvInputData = frame
                opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                try:
                    keypoints_seq = np.vstack((keypoints_seq, [list(zip(datum.poseKeypoints[0, :, 0] / width, datum.poseKeypoints[0, :, 1] / height))]))
                    lh_keypoints_seq = np.vstack((lh_keypoints_seq, [list(zip(datum.handKeypoints[0][0, :, 0] / width, datum.handKeypoints[0][0, :, 1] / height))]))
                    rh_keypoints_seq = np.vstack((rh_keypoints_seq, [list(zip(datum.handKeypoints[1][0, :, 0] / width, datum.handKeypoints[1][0, :, 1] / height))]))
                except BaseException:
                    bad_frame = True
                    break

                t += sample_time
            if not bad_frame:
                np.save(os.path.join(ROOT_PATH, 'sunil_keypoints', str(row['interval_id'])), [keypoints_seq, lh_keypoints_seq, rh_keypoints_seq, vid_dim], allow_pickle = True)

except Exception as e:
    print(e)
    sys.exit(-1)










