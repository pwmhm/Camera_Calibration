#Sourced from docs.opencv.org


import argparse
import numpy as np
import cv2 as cv
import glob
import pickle




camera_type = ["mono", "stereo"]

parser = argparse.ArgumentParser(description="Mono/Stereo Calibration", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-dpath', '--dpath', metavar="Image Directory", default="M:\\CODE\\Dataset\\Calibration\\images_Rev04",
                    help = "Directories in which the images are stored")
parser.add_argument('-spath','--spath', metavar="Save Directory", default="M:\\CODE\\Dataset\\Calibration",
                    help="Directory in which the parameters will be saved for future use")
parser.add_argument('-cname', '--cname', metavar="Camera Name", default="Vaccinium",
                    help="Camera name to help distinguish different files")
parser.add_argument('-save','--save', metavar="Save Flag", default=False, type=bool)
parser.add_argument('-imsize', '--imsize', metavar="Image Size", default=(2048,1536), type=int, nargs=2,
                    help= "The dimensions of the image that will be used for calibration. Should be uniform")
parser.add_argument('-cdim', '--cdim', metavar="Chessboard Size", default=(12,9), type=int, nargs=2,
                    help= "Amount of squares (minus 1) going in the x and y direction")
parser.add_argument('-cl', '--cl', metavar="Square Size", default=1, type=int,
                    help="Length of the sides of the square in cm")
parser.add_argument('-mos', '--mos', metavar="Mono or Stereo", default="mono", choices=camera_type,
                    help="Type of the camera which will be calibrated")
parser.add_argument('-cbim', '--cbim', metavar="Show Chessboard Corners", default=False, type=bool,
                    help="Show image of corners to ensure proper functioning of code")
parser.add_argument('-mod', '--mod', metavar="Mode", default="calibrate")

def main() :
    imgsize = args.imsize
    cb_width = args.cdim[0]
    cb_height = args.cdim[1]
    c_size = args.cl

    objp = np.zeros((cb_width * cb_height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cb_width, 0:cb_height].T.reshape(-1, 2) * c_size

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    # Calibration
    if args.mos == "mono" :
        datapath = os.path.join(args.dpath, "*.png")

        obj_points, im_points = find_imagepoints(datapath, cb_width, cb_height, objp, criteria, args.cbim)
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, im_points, imgsize, None, None)
        repo_error = reproject_error(mtx,dist,rvecs,tvecs,obj_points,im_points)

        if args.save == True :
            filename = args.cname + "_" + args.mos + ".matrices"
            savepath = os.path.join(args.spath, filename)
            # Put Camera Calibration parameters together
            camera_matrices = {
                'camera matrix': mtx,
                'distortion coefficient': dist,
                'rotation vector': rvecs,
                'translation vector': tvecs,
                'reprojection error' : repo_error
            }
            # Save parameters for future use
            print(camera_matrices)
            print("saving to {0}".format(savepath))
            with open(savepath, "wb") as handle:
                pickle.dump(camera_matrices, handle)

    elif args.mos == "stereo" :
        datapath_l = os.path.join(args.dpath, "Left", "*.png")
        datapath_r = os.path.join(args.dpath, "Right", "*.png")
        objpl, impl = find_imagepoints(datapath_l, cb_width, cb_height, objp, criteria, args.cbim)
        objpr, impr = find_imagepoints(datapath_r, cb_width, cb_height, objp, criteria, args.cbim)

        ret, K1, D1, R1, T1 = cv.calibrateCamera(objpl,impl,imgsize, None, None)
        ret, K2, D2, R2, T2 = cv.calibrateCamera(objpr,impr,imgsize, None, None)

        _, K1, D1, K2, D2, R, T, E, F = cv.stereoCalibrate(objpl, impl, impr, K1, D1, K2, D2, imgsize,
                                        flags=cv.CALIB_FIX_INTRINSIC)
        if args.save == True :
            filename = args.cname + "_" + args.mos + ".matrices"
            savepath = os.path.join(args.spath, filename)
            camera_matrices = {
                'K_Left'    : K1    , 'D_Left'  : D1,
                'R_Left'    : R1    , 'T_Left'  : T1,
                'K_Right'   : K2    ,'D_Right'  : D2,
                'R_Left'    : R2    ,'T_Left'   : T2,
                'Rrect'     : R     ,'Trect'    : T,
                'E'         : E     , 'F'       : F
            }
            # Save parameters for future use
            print(camera_matrices)
            print("saving to {0}".format(savepath))
            with open(savepath, "wb") as handle:
                pickle.dump(camera_matrices, handle)






def reproject_error(K, D, R, T, OP, IP):
    # Check Re-Projection Error
    mean_error = 0
    for i in range(len(OP)):
        imgpoints2, _ = cv.projectPoints(OP[i], R[i], T[i], K, D)
        error = cv.norm(IP[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    return mean_error / len(OP)


def find_imagepoints(path,cb_width,cb_height,objp,criteria, showim) :

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for fname in sorted(glob.glob(path)):
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(img, (cb_width, cb_height), flags=cv.CALIB_CB_ADAPTIVE_THRESH)
        # If found, add object points, image points (after refining them)

        if ret == True:
            print(fname)
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (cb_width, cb_height), corners2, ret)
            if showim == True :
                show = cv.resize(img, (800, 600))
                cv.imshow('img', show)
                cv.waitKey(0)
                cv.destroyAllWindows()

    return objpoints, imgpoints

if __name__ == '__main__':
    import numpy as np
    import cv2 as cv
    import glob
    import pickle
    import os

    args = parser.parse_args()
    main()





