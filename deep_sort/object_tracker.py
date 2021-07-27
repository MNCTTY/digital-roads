from deep_sort.tracker import Tracker
from deep_sort import nn_matching
from application_util import preprocessing

import os
import json
import random


def detect_cv2_camera():

    folder = "day1"
    images_day1 = os.listdir(folder)

    color_list = []
    for j in range(10000):
        color_list.append(((int)(random.randrange(255)),
                           (int)(random.randrange(255)),
                           (int)(random.randrange(255))))







if __name__ == '__main__':
    detect_cv2_camera()