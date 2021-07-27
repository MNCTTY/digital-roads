import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean

MIN_MATCH_COUNT = 10

img1 = cv2.imread('image.png',0)  # queryImage
img2 = cv2.imread('image2.png',0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
search_params = dict(checks = 100)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 1*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    kp1_matched=([ kp1[m.queryIdx] for m in good ])
    kp2_matched=([ kp2[m.trainIdx] for m in good ])

    kp1_miss_matched=[kp for kp in kp1 if kp not in kp1_matched]
    kp2_miss_matched=[kp for kp in kp2 if kp not in kp2_matched]

    # draw only miss matched or not matched keypoints location
    img1_miss_matched_kp = cv2.drawKeypoints(img1,kp1_miss_matched, None,color=(255,0,0), flags=2)
    plt.imshow(img1_miss_matched_kp),plt.show()

    img2_miss_matched_kp = cv2.drawKeypoints(img2,kp2_miss_matched, None,color=(255,0,0), flags=2)
    plt.imshow(img2_miss_matched_kp),plt.show()

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img3 = cv2.vconcat(img1_miss_matched_kp, img2_miss_matched_kp)
    cv2.imshow('image', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None