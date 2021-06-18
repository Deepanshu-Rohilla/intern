# import sys

# # from scipy.misc import imread
# from imageio import imread

# from scipy.linalg import norm
# from scipy import sum, average
# import cv2

# def main():
#     file1, file2 = sys.argv[1:1+2]
#     # read images as 2D arrays (convert to grayscale for simplicity)
#     img1 = to_grayscale(imread(file1).astype(float))
#     img2 = to_grayscale(imread(file2).astype(float))
#     # compare
#     n_m, n_0 = compare_images(img1, img2)
#     print ("Manhattan norm:", n_m, "/ per pixel:", n_m/img1.size)
#     print ("Zero norm:", n_0, "/ per pixel:", n_0*1.0/img1.size)


# def compare_images(img1, img2):
#     # normalize to compensate for exposure difference, this may be unnecessary
#     # consider disabling it
#     img1 = normalize(img1)
#     img2 = normalize(img2)
#     # calculate the difference and its norms
#     width1  = img1.shape[1]
#     height1 = img1.shape[0]
#     width2  = img1.shape[1]
#     height2 = img1.shape[0]

#     width = min(width1,width2)
#     height = min(height1,height2)

#     img1 = cv2.rectangle(img1, width, height)
#     img2 = cv2.rectangle(img2, width, height)
	
#     diff = img1 - img2  # elementwise for scipy arrays
#     m_norm = sum(abs(diff))  # Manhattan norm
#     z_norm = norm(diff.ravel(), 0)  # Zero norm
#     return (m_norm, z_norm)

# def to_grayscale(arr):
#     "If arr is a color image (3D array), convert it to grayscale (2D array)."
#     if len(arr.shape) == 3:
#         return average(arr, -1)  # average over the last axis (color channels)
#     else:
#         return arr

# def normalize(arr):
#     rng = arr.max()-arr.min()
#     amin = arr.min()
#     return (arr-amin)*255/rng

# if __name__ == "__main__":
#     main()



import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('musk/musk0.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('musk/musk1.jpg',cv.IMREAD_GRAYSCALE) # trainImage
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
# Draw first 10 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()