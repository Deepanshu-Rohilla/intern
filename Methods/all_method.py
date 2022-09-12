import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys
import time


def m1(img1, img2, show):
    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    try:
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)
        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
    except:
        return 0

    print(len(matches))
    if(show):
        img3 = cv.drawMatches(img1,kp1,img2,kp2,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3),plt.show()
    return len(matches)

def m2(img1, img2, show):
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    try:    
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
    except:
        return 0
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
    if(show):
        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3),plt.show()
    return len(good)

def m3(img1, img2, show):
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    try:
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
    except:
        return 0
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's pape
    good=0
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good+=1
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv.DrawMatchesFlags_DEFAULT)
    print(len(matches))
    print(good)
    if(show):
        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
        plt.imshow(img3,),plt.show()
    return good + len(matches)

def method(img_path1, dir1,dir2,methodNum):
    goodArray1 = []
    goodArrayName1 = []
    goodArray2 = []
    goodArrayName2 = []
    img1 = cv.imread(img_path1,cv.IMREAD_GRAYSCALE) # queryImage
    for i in range(101,151):
        img_path2 = dir1 + '/donald trump speech' + str(i) + '.jpg'
        print(img_path2)
        img2 = cv.imread(img_path2,cv.IMREAD_GRAYSCALE) # trainImage
        l=0
        if(methodNum==1):
            l = m1(img1,img2,False)
        elif(methodNum==2):
            l = m2(img1,img2,False)
        elif(methodNum==3):
            l = m3(img1,img2,False)
        goodArray1.append(l)
        goodArrayName1.append(img_path2)

    for i in range(1,51):
        img_path2 = dir1 + '/trump' + str(i) + '.jpg'
        print(img_path2)
        img2 = cv.imread(img_path2,cv.IMREAD_GRAYSCALE) # trainImage
        l=0
        if(methodNum==1):
            l = m1(img1,img2,False)
        elif(methodNum==2):
            l = m2(img1,img2,False)
        elif(methodNum==3):
            l = m3(img1,img2,False)
        goodArray1.append(l)
        goodArrayName1.append(img_path2)

    for i in range(51,101):
        img_path2 = dir1 + '/donald trump' + str(i) + '.jpg'
        print(img_path2)
        img2 = cv.imread(img_path2,cv.IMREAD_GRAYSCALE) # trainImage
        l=0
        if(methodNum==1):
            l = m1(img1,img2,False)
        elif(methodNum==2):
            l = m2(img1,img2,False)
        elif(methodNum==3):
            l = m3(img1,img2,False)
        goodArray1.append(l)
        goodArrayName1.append(img_path2)

    for i in range(150):
        img_path2 = dir2 + '/musk' + str(i) + '.jpg'
        print(img_path2)
        img2 = cv.imread(img_path2,cv.IMREAD_GRAYSCALE) # trainImage
        l=0
        if(methodNum==1):
            l = m1(img1,img2,False)
        elif(methodNum==2):
            l = m2(img1,img2,False)
        elif(methodNum==3):
            l = m3(img1,img2,False)
        goodArray2.append(l)
        goodArrayName2.append(img_path2)
    print("The original images of this deepfake are: ")
    index1 = goodArray1.index(max(goodArray1))
    index2 = goodArray2.index(max(goodArray2))
    features1 = max(goodArray1)
    features2 = max(goodArray2)
    print(goodArrayName1[index1])
    print(goodArrayName2[index2])

    print("1. " + goodArrayName1[index1] + " with number of feautres matching: " + str(features1))
    print("2. " + goodArrayName2[index2] + " with number of feautres matching: " + str(features2))
    if(methodNum==1):
        m1(img1,cv.imread(goodArrayName1[index1],cv.IMREAD_GRAYSCALE), True)
        m1(img1,cv.imread(goodArrayName2[index2],cv.IMREAD_GRAYSCALE), True)
    elif(methodNum==2):
        m2(img1,cv.imread(goodArrayName1[index1],cv.IMREAD_GRAYSCALE), True)
        m2(img1,cv.imread(goodArrayName2[index2],cv.IMREAD_GRAYSCALE), True)
    elif(methodNum==3):
        m3(img1,cv.imread(goodArrayName1[index1],cv.IMREAD_GRAYSCALE), True)
        m3(img1,cv.imread(goodArrayName2[index2],cv.IMREAD_GRAYSCALE), True)


def seedOut(methodNum):
    for i in range(6600, 6626):
        goodArray1 = []
        goodArrayName1 = []
        goodArray2 = []
        goodArrayName2 = []
        img_path1 = 'seed/seed' + str(i) + '.png'
        img1 = cv.imread(img_path1,cv.IMREAD_GRAYSCALE) # queryImage
        for j in range(0,20):
            img_path2 = 'out/image' + str(j) + '.png'
            print(img_path2)
            img2 = cv.imread(img_path2,cv.IMREAD_GRAYSCALE) # trainImage
            l=0
            if(methodNum==1):
                l = m1(img1,img2,False)
            elif(methodNum==2):
                print("here")
                l = m2(img1,img2,False,i)
            elif(methodNum==3):
                l = m3(img1,img2,False)
            goodArray1.append(l)
            goodArrayName1.append(img_path2)
        print("The original images of this deepfake are: ")
        index1 = goodArray1.index(max(goodArray1))
        # index2 = goodArray2.index(max(goodArray2))
        features1 = max(goodArray1)
        # features2 = max(goodArray2)
        print(goodArrayName1[index1])
        # print(goodArrayName2[index2])

        print("1. " + goodArrayName1[index1] + " with number of feautres matching: " + str(features1))
        # print("2. " + goodArrayName2[index2] + " with number of feautres matching: " + str(features2))
        if(methodNum==1):
            m1(img1,cv.imread(goodArrayName1[index1],cv.IMREAD_GRAYSCALE), True)
            # m1(img1,cv.imread(goodArrayName2[index2],cv.IMREAD_GRAYSCALE), True)
        elif(methodNum==2):
            m2(img1,cv.imread(goodArrayName1[index1],cv.IMREAD_GRAYSCALE), True,i)
            # m2(img1,cv.imread(goodArrayName2[index2],cv.IMREAD_GRAYSCALE), True)
        elif(methodNum==3):
            m3(img1,cv.imread(goodArrayName1[index1],cv.IMREAD_GRAYSCALE), True)
            # m3(img1,cv.imread(goodArrayName2[index2],cv.IMREAD_GRAYSCALE), True)


def main():
    img_path1 = sys.argv[2]
    dir1 = sys.argv[3]
    dir2 = sys.argv[4]
    methodNum = int(sys.argv[1])
    # seedOut(methodNum)
    method(img_path1,dir1,dir2,methodNum)

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Runtime of the program is {end - start}")
