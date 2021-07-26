from sort.sort import *
import os
import json
import random
import cv2.cv2 as cv2


def detect_cv2_camera():
    # Картинки
    folder = "day1"
    images_day1 = os.listdir(folder)

    # Трекер
    mot_tracker = Sort()

    # Случайные цвета
    color_list = []
    for j in range(10000):
        color_list.append(((int)(random.randrange(255)),
                           (int)(random.randrange(255)),
                           (int)(random.randrange(255))))


    for im in images_day1:
        if im.split("_")[1] == "0":  # 0-5 разные камеры

            im_path = os.path.join(folder, im)
            json_path = os.path.join("day_1JSON", im[:-4] + ".json")

            img = cv2.imread(im_path)

            # разбираем json
            boxes = []
            confidences = []
            className = []
            try:
                with open(json_path) as json_file:
                    data = json.load(json_file)
            except Exception:
                data = []
            for obj in data:
                boxes.append(obj["bbox"])
                confidences.append(obj["score"])
                className.append(obj["class"])

            # Performs non maximum suppression given boxes and corresponding scores.
            # тут отбрасываются некоторые боксы
            # idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

            # берем все боксы
            idxs = [[i] for i in range(len(boxes))]
            result_img = np.copy(img)
            count_detection = 0
            for j in range(len(idxs)):
                count_detection += 1
            detects = np.zeros((count_detection, 5))
            count = 0


            # Подготовим формат который ест трекер
            for j in range(len(idxs)):
                b = boxes[j]
                x1 = int(b[0])
                y1 = int(b[1])
                x2 = int(b[2])
                y2 = int(b[3])
                box = np.array([x1, y1, x2, y2, confidences[idxs[j][0]]])
                detects[count,:] = box[:]
                count += 1

            # Трекер + отрисовка
            if len(detects)!=0:
                trackers = mot_tracker.update(detects)
                for d in trackers:
                    result_img = cv2.rectangle(result_img, ((int)(d[0]), (int)(d[1])), ((int)(d[2]), (int)(d[3])), color_list[(int)(d[4])], 2)

            # тут вывод на экран
            scale_percent = 30
            width = int(result_img.shape[1] * scale_percent / 100)
            height = int(result_img.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(result_img, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow('demo', resized)
            key = cv2.waitKey(50) & 0xFF
            if key == ord("q"):
                break

if __name__ == '__main__':
    detect_cv2_camera()