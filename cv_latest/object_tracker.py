
# import the necessary packages
from centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import os
import json
import cv2

# construct the argument parse and parse the arguments

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
print("[INFO] starting video stream...")
vs = cv2.VideoCapture('video.avi')
time.sleep(2.0)
while True:
	# loop over the frames from the video stream
	for file in sorted(os.listdir('day_1JSON')):
		path = '/Users/maxwhite/PycharmProjects/Centroid-Based_Object_Tracking/Python/day_1JSON/'
		#imgpath = '/Users/maxwhite/PycharmProjects/Centroid-Based_Object_Tracking/Python/day1/'
		#_0_ - all frames within 1 video
		file = os.path.join(path, file)
		if '_0_' in file:
			with open(file, 'r') as f:
				frame = vs.read()
				frame = frame[1]
				distros_dict = json.load(f)
				rects = []
				for distro in distros_dict:
					bbox = distro['bbox']
					bbox_class = distro['class']
					bbox_int = bbox[0], bbox[1], bbox[2], bbox[3]
					bbox_array = np.asarray(bbox_int)



						# update the bounding box rectangles list
					box = bbox_array

					rects.append(box.astype("int"))

					# draw a bounding box surrounding the object so we can
					# visualize it
					(startX, startY, endX, endY) = bbox_int
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
							cv2.putText(frame, bbox_class, (centroid[0] - 10, centroid[1] - 10),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
							cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

						# show the output frame
				cv2.imshow("Frame", frame)
				key = cv2.waitKey(1) & 0xFF

				# if the `q` key was pressed, break from the loop
				if key == ord("q"):
					break

						# do a bit of cleanup
				cv2.destroyAllWindows()
