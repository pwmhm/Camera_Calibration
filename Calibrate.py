#Sourced from docs.opencv.org

import numpy as np
import cv2 as cv
import glob
import pickle

imgsize = (2048, 1536)
cb_width = 12
cb_height = 9
c_size = 1 #cm

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((cb_width*cb_height,3), np.float32)
objp[:,:2] = np.mgrid[0:cb_width,0:cb_height].T.reshape(-1,2)*c_size

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('Images\*.bmp')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(img, (cb_width,cb_height), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (cb_width,cb_height), corners2, ret)
        show = cv.resize(img, (800,600))
        cv.imshow('img', show)
        cv.waitKey(500)
cv.destroyAllWindows()

#Calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, imgsize, None, None)

#Put Camera Calibration parameters together
camera_matrices = {
    'camera matrix' : mtx,
    'distortion coefficient' : dist,
    'rotation vector' : rvecs,
    'translation vector' : tvecs
}

#Check Re-Projection Error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

#Save parameters for future use
with open("camera.matrices", "wb") as handle :
    pickle.dump(camera_matrices, handle)