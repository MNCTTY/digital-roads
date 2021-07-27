from centroidtracker import CentroidTracker
import cv2
import imutils
import os
from imutils.video import VideoStream
import numpy as np
import json
from UliEngineering.Math.Coordinates import BoundingBox
from d2l import tensorflow as d2l
vs = cv2.VideoCapture('video.avi')


# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)
confidence = 0.5
rects = []

for file in sorted(os.listdir('day_1JSON')):
  path = '/Users/maxwhite/PycharmProjects/Centroid-Based_Object_Tracking/Python/day_1JSON/'
  imgpath = '/Users/maxwhite/PycharmProjects/Centroid-Based_Object_Tracking/Python/day1/'

  if '_0_' in file:
    #print(file)
    #print(os.path.join(imgpath, f'{os.path.split(file)[0]}.jpg'))
    #frame = cv2.imread(os.path.join(imgpath, f'{os.path.splitext(file)[0]}.jpg'), cv2.IMREAD_ANYCOLOR)

    file = os.path.join(path, file)
    with open(file, 'r') as f:
        i = 0
        frame = vs.read()
        frame = frame[1]
        (H, W) = frame.shape[:2]
        distros_dict = json.load(f)
        for distro in distros_dict:
          bbox = distro['bbox']
          b_class = distro['class']
          bbox_int = bbox[0], bbox[1], bbox[2], bbox[3]
          bbox_array = np.asarray(bbox_int)
          rects.append(bbox_array.astype('int'))
          # draw a bounding box surrounding the object so we can
          # visualize it
          print(bbox_array)
          (startX, startY, endX, endY) = bbox_array.astype('int')
          cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0, 255, 0), 2)
          # update our centroid tracker using the computed set of bounding
          # box rectangles
          objects = ct.update(rects)
          # loop over the tracked objects
          for (objectID, centroid) in objects.items():
                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                #cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        cv2.imshow('frame',frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
                break
