import os
import cv2
import sys
from PIL import Image
import PIL
import numpy as np
from natsort import natsorted
def convert_to_video():
    image_folder = '/Users/maxwhite/PycharmProjects/CV_latest/images/'
    video_name = 'video.avi'
    images = [img for img in natsorted(os.listdir(image_folder)) if img.endswith('.jpg')]
    print(images)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 30, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
def show_video():
    cap = cv2.VideoCapture('video.avi')
    while (cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
def convert_to_video_2():
    from PIL import Image
    d = 0
    image_folder = '/Users/maxwhite/PycharmProjects/CV_latest/day1'
    save_path = '/Users/maxwhite/PycharmProjects/CV_latest/images/'
    list_im = []
    for file in sorted(os.listdir(image_folder)):
        if file.endswith('.jpg'):
            list_im.append(file)
            if len(list_im) == 3:
                imgs = [PIL.Image.open(os.path.join(image_folder,i)) for i in list_im]
                # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
                min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
                imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))

                # save that beautiful picture
                imgs_comb = PIL.Image.fromarray(imgs_comb)
                imgs_comb.save(f'{save_path}frame{d}.jpg')
                d += 1
                list_im = []

if __name__ == "__main__":
    convert_to_video()

