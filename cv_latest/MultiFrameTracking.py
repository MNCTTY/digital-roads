from centroidtracker import CentroidTracker
import numpy as np
import random
import time
import os
import json
import cv2.cv2 as cv2

ct = CentroidTracker()

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

        rects = []
        for i, im_path in enumerate(three_frames):
            json_path = os.path.join("day_1JSON", im_path[:-4] + ".json")
            # READ JSON
            with open(json_path, 'r') as f:
                distros_dict = json.load(f)
                for el in distros_dict:
                    tmp = []
                    if el["score"] > 0:  # OBJECTS SCORE
                        for j in range(len(el["bbox"])):
                            # BIBOX ALTERATION
                            if j % 2 == 0:
                                tmp.append(el['bbox'][j] + 2048*i)  # i - 0/1/2 depends on which camera we are on
                            else:
                                tmp.append(el['bbox'][j])
                    else:
                        continue
                    el['bbox'] = tmp
                    rects.append(el['bbox'])
                    cv2.rectangle(image, (el['bbox'][0], el['bbox'][1]), (el['bbox'][2], el['bbox'][3]),
                                  (0, 255, 0), 2)

        print(rects)

        objects = ct.update(rects)

        # remained the same
        for (objectID, centroid) in objects.items():
            text = "ID {}".format(objectID)

            cv2.putText(image, str(objectID), (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(image, (centroid[0], centroid[1]), 10, color_list[objectID], 10)
            center = (centroid[0], centroid[1])

            pts[objectID].append(center)

            for j in range(1, len(pts[objectID])):
                if pts[objectID][j - 1] is None or pts[objectID][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                #cv2.line(image, (pts[objectID][j - 1]), (pts[objectID][j]), color_list[objectID], thickness)

        # IMSHOW
        resized = cv2.resize(image, (1000, 700), interpolation=cv2.INTER_AREA)
        cv2.imshow('demo', resized)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

        # need to reset for next three frames
        three_frames = []
