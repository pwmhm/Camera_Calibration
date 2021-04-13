import numpy as np
import cv2 as cv
import glob
import pickle

with open("camera.matrices", "rb") as handle :
    cmtx = pickle.load(handle)


#redefine camera matrix
images = glob.glob('Images\*.bmp')


for fnames in images :
    filename = fnames.replace("Images", "")
    img = cv.imread(fnames)
    h , w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(cmtx['camera matrix'], cmtx['distortion coefficient'], (w,h), 1, (w,h))

    # undistort
    dst = cv.undistort(img, cmtx['camera matrix'], cmtx['distortion coefficient'], None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    filepath = "Undistorted" + filename
    print(filepath)
    cv.imwrite(filepath, dst)

