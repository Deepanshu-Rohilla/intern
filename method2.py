import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys
img1 = cv.imread(sys.argv[1],cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread(sys.argv[2],cv.IMREAD_GRAYSCALE) # trainImage
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
print(len(good))
print(len(matches))
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()

# 11 and 6 (only face)
# 23 and 27 (whole image)
# 125 for identical (only face)
# 687 for identical (whole image)
