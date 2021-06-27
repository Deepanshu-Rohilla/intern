import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
goodArray = []
goodArrayName = []
for i in range(101,151):
    img1 = cv.imread('image5.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
    img2 = cv.imread('trump/donald trump speech' + str(i) + '.jpg',cv.IMREAD_GRAYSCALE) # trainImage 
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    print(len(matches))
    # Draw first 10 matches.
    img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()
    print(len(matches))



for i in range(1,150):
    img1 = cv.imread('musk/musk0.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
    img2 = cv.imread('musk/musk' + str(i) + '.jpg',cv.IMREAD_GRAYSCALE) # trainImage 
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    print(len(matches))
    # Draw first 10 matches.
    img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()
    print(len(matches))

# 104 and 94
'''136
136
134
134
119
119
118
118
129
129
138
138
129
129
117
117
125
125
117
117
110
110
128
128
142
142
119
119
137
137
146
146
107
107
131
131
132
132
127
127
130
130
118
118
131
131
119
119
106
106
136
136
117
117
135
135
152
152
123
123
125
125
123
123
128
128
121
121
130
130
156
156
129
129
114
114
116
116
137
137
141
141
129
129
134
134
129
129
119
119
74
74
123
123
117
117
133
133
500
500
137
137
128
128
110
110
153
153
118
118
118
118
119
119'''