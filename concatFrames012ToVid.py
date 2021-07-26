import os
import json
import random
from functions import *
import cv2.cv2 as cv2
folder = "day1"
images_day1 = os.listdir(folder)

forConcat = []

size = (6144, 2448)
video_name = 'video.avi'
video = cv2.VideoWriter(video_name, 0, 15, size)


for im in images_day1:

    if "_0_" in im or "_1_" in im or "_2_" in im:
        forConcat.append(im)
    elif "_3_" in im or "_4_" in im:
        continue
    else:
        fr0, fr1, fr2 = cv2.imread(os.path.join("day1", forConcat[0])),\
                        cv2.imread(os.path.join("day1", forConcat[1])),\
                        cv2.imread(os.path.join("day1", forConcat[2]))
        image = cv2.hconcat([fr0, fr1, fr2])
        # print(image.shape)
        resized = cv2.resize(image, (1000,700), interpolation=cv2.INTER_AREA)
        cv2.imshow('demo', resized)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

        # video.write(image)

        forConcat = []

# cv2.destroyAllWindows()
# video.release()




