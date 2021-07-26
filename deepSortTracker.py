from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort import nn_matching
from deep_sort.application_util import preprocessing
from deep_sort.deep_sort.detection import Detection
from deep_sort.tools import generate_detections as gdet

import numpy as np
import cv2.cv2 as cv2
import os
import json
import random

# for cv2.line display
from _collections import deque
pts = [deque(maxlen=15) for _ in range(10000)]

def detect_cv2_camera():

    max_cosine_distance = 0.7
    nn_budget = None
    nms_max_overlap = 1

    model_filename = 'deep_sort/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    folder = "day1"
    images_day1 = os.listdir(folder)

    color_list = []
    for j in range(10000):
        color_list.append(((int)(random.randrange(255)),
                           (int)(random.randrange(255)),
                           (int)(random.randrange(255))))


    for im in images_day1:
        if im.split("_")[1] == "0":  # 0-5 разные камеры

            im_path = os.path.join(folder, im)
            json_path = os.path.join("day_1JSON", im[:-4] + ".json")

            frame = cv2.imread(im_path)
            width, height, _ = frame.shape

            boxes = []
            confidences = []
            classNames = []
            try:
                with open(json_path) as json_file:
                    data = json.load(json_file)
            except Exception:
                data = []
            for obj in data:
                bb = obj["bbox"]
                bb[2] -= bb[0]
                bb[3] -= bb[1]
                boxes.append(bb)
                confidences.append(obj["score"])
                classNames.append(obj["class"])

            features = encoder(frame, boxes)

            detections = [Detection(bbox, score, feature) for bbox, score, feature in
                          zip(boxes, confidences, features)]

            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array(classNames)

            # boxes = np.array(boxes)
            # confidences = np.array(confidences)
            # classNames =

            indxs = preprocessing.non_max_suppression(boxs, nms_max_overlap, scores)
            detections = [detections[i] for i in indxs]


            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                color = color_list[int(track.track_id)]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
                pts[track.track_id].append(center)

                for j in range(1, len(pts[track.track_id])):
                    if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                        continue
                    thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                    cv2.line(frame, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), color, thickness)

            #IMSHOW
            scale_percent = 30
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow('demo', resized)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break


if __name__ == '__main__':
    detect_cv2_camera()
