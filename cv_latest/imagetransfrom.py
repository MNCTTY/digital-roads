from centroidtracker import CentroidTracker
import numpy as np
import random
import time
import os
import json
import cv2.cv2 as cv2

folder = "/Users/maxwhite/PycharmProjects/CV_latest/day1/"
images_day1 = sorted(os.listdir(folder))

# for line drawing
from _collections import deque
pts = [deque(maxlen=15) for _ in range(10000)]

# active three frames pathes
three_frames = []

# for different colors
color_list = []
for j in range(1000):
    color_list.append(((int)(random.randrange(255)),
                       (int)(random.randrange(255)),
                       (int)(random.randrange(255))))

for im_root in images_day1:
    if "_0_" in im_root or "_1_" in im_root or "_2_" in im_root:
        three_frames.append(im_root)
    elif "_3_" in im_root or "_4_" in im_root:
        continue
    else:
        fr0, fr1, fr2 = cv2.imread(os.path.join("day1", three_frames[0])),\
                        cv2.imread(os.path.join("day1", three_frames[1])),\
                        cv2.imread(os.path.join("day1", three_frames[2]))

        # horizontal concatenation
        image = cv2.hconcat([fr0, fr1, fr2])