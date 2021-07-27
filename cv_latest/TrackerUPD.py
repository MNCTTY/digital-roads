from centroidtracker import CentroidTracker
import numpy as np
import random
import time
import os
import json
import cv2.cv2 as cv2

ct = CentroidTracker()

folder = "day1"
images_day1 = sorted(os.listdir(folder))

color_list = []
for j in range(1000):
    color_list.append(((int)(random.randrange(255)),
                       (int)(random.randrange(255)),
                       (int)(random.randrange(255))))

for imid, im in enumerate(images_day1):
    if im.split("_")[1] == "0":  # 0-5 разные камеры

        im_path = os.path.join(folder, im)
        json_path = os.path.join("day_1JSON", im[:-4] + ".json")

        img = cv2.imread(im_path)

        with open(json_path, 'r') as f:
            distros_dict = json.load(f)
            rects = []
            for distro in distros_dict:
                bbox = distro['bbox']
                bbox_class = distro['class']
                rects.append(bbox)

                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          (0, 255, 0), 2)

            objects = ct.update(rects)

            for (objectID, centroid) in objects.items():
                    text = "ID {}".format(objectID)

                    cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                #cv2.circle(img, (centroid[0], centroid[1]), 10, color_list[objectID], 10)

            # IMSHOW
            scale_percent = 100
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            cv2.imshow("Frame", resized)
            key = cv2.waitKey(0) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break