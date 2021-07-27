import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

src = cv.imread("znak2.jpg", cv.IMREAD_GRAYSCALE)
#Mean Filter
src_blur = cv.blur(src, (3, 3))
#Laplac sharpening
lapmask = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], np.float32)
src_lap = cv.filter2D(src_blur, -1, kernel=lapmask)
#Laplacian graph threshold segmentation (binary segmentation)
ret, th = cv.threshold(src_lap, 50, 255, cv.THRESH_BINARY)
#Subtract the Laplacian diagram from the blurred original image to make the defect more clear
img = src_blur - abs(50*th)
#Adaptive threshold segmentation
dst = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 5)
#Outline extraction and drawing
contours, hierarchy = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
src_rgb = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
cv.drawContours(src_rgb, contours, -1, (255, 0, 0), 1)
cv.imwrite("/Users/maxwhite/PycharmProjects/imageidentity/new_image.png", src_rgb)

cv.namedWindow("test", cv.WINDOW_AUTOSIZE)
cv.imshow("test", src_rgb)
plt.hist(src_rgb.ravel(), 256, [0, 256])


cv.waitKey(0)
cv.destroyAllWindows()

