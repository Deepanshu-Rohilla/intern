import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
goodArray = []
goodArrayName = []
for i in range(101,151):
    img1 = cv.imread('image5.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
    print('trump/donald trump speech' + str(i) + '.jpg')
    img2 = cv.imread('trump/donald trump speech' + str(i) + '.jpg',cv.IMREAD_GRAYSCALE) # trainImage
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    try:    
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
    except:
        continue
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
    goodArray.append(len(good))
    goodArrayName.append('trump/donald trump speech' + str(i) + '.jpg')
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3),plt.show()

for i in range(1,51):
    img1 = cv.imread('image5.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
    print('trump/trump' + str(i) + '.jpg')
    img2 = cv.imread('trump/trump' + str(i) + '.jpg',cv.IMREAD_GRAYSCALE) # trainImage
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    try:
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
    except:
        continue
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
    goodArray.append(len(good))
    goodArrayName.append('trump/trump' + str(i) + '.jpg')
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3),plt.show()


for i in range(51,101):
    img1 = cv.imread('image5.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
    print('trump/donald trump' + str(i) + '.jpg')
    img2 = cv.imread('trump/donald trump' + str(i) + '.jpg',cv.IMREAD_GRAYSCALE) # trainImage
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    try:
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
    except:
        continue
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
    goodArray.append(len(good))
    goodArrayName.append('trump/donald trump' + str(i) + '.jpg')
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3),plt.show()




for i in range(150):
    img1 = cv.imread('image5.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
    print('musk/musk' + str(i) + '.jpg')
    img2 = cv.imread('musk/musk' + str(i) + '.jpg',cv.IMREAD_GRAYSCALE) # trainImage
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    try:
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
    except:
        continue
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
    goodArray.append(len(good))
    goodArrayName.append('musk/musk' + str(i) + '.jpg')
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3),plt.show()


print("ANSWER IS ")
print(max(goodArray))
print("AND IT IS CORRESPONDING TO ")
index = goodArray.index(max(goodArray))
print(goodArrayName[index])
plt.imshow(cv.imread(goodArrayName[index]))
# 11 and 6 (only face)
# 23 and 27 (whole image)
# 125 for identical (only face)
# 687 for identical (whole image)
