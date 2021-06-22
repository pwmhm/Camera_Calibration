import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open("M:\\CODE\\Dataset\\Calibration\\Vaccinium_stereo.matrices", "rb") as handle :
    cmtx = pickle.load(handle)


Left_rectified = []
Right_rectified = []

# Params = [                              #Relevant parameters for rectification, if you plan to use ctypes to iterate
#     'left_K', 'left_dist',              #the function getNodes, this would be useful
#     'right_K', "right, dist",
#     'relative_R', 'relative_tvec'
#     'left_res', 'right_res'
# ]

Params_dict = {}
impath_l = "M:\\CODE\\Dataset\\Vaccinium\\Color\\Left\\*.png"
impath_r = "M:\\CODE\\Dataset\\Vaccinium\\Color\\Right\\*.png"
rrect = "M:\\CODE\\Dataset\\Vaccinium\\Rectified\\Left\\*.png"
lrect = "M:\\CODE\\Dataset\\Vaccinium\\Rectified\\Right\\*.png"
savepath_l = "M:\\CODE\\Dataset\\Vaccinium\\Rectify\\Left"
savepath_r = "M:\\CODE\\Dataset\\Vaccinium\\Rectify\\Right"
disp_save = "M:\\CODE\\Dataset\\Vaccinium\\Rectify_Disp"

if not os.path.exists(savepath_l) :
    os.makedirs(savepath_l)
if not os.path.exists(savepath_r) :
    os.makedirs(savepath_r)
if not os.path.exists(disp_save):
    os.makedirs(disp_save)

K1 = cmtx['K_Left']
D1 = cmtx['D_Left']
K2 = cmtx['K_Right']
D2 = cmtx['D_Right']
imsize = (2048, 1536)
R = cmtx['Rrect']
T = cmtx['Trect']

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, imsize, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha= -1)

lmapx, lmapy = cv2.initUndistortRectifyMap(K1, D1, R1, P1, imsize, cv2.CV_32FC1)
rmapx,rmapy = cv2.initUndistortRectifyMap(K2,D2,R2,P2, imsize, cv2.CV_32FC1)



disp_name = []
unrect_left = []

i = 0
for filename in glob.glob(impath_l):
    if i == 10 :
        break
    i += 1
    real_name = os.path.split(filename)
    save = os.path.join(savepath_l,real_name[1])
    disp_name.append(real_name[1])
    unrect_im = cv2.imread(filename)
    unrect_left.append(unrect_im)
    rect_im = cv2.remap(unrect_im, lmapx,lmapy, cv2.INTER_LINEAR)
    Left_rectified.append(rect_im)
    cv2.imwrite(save, rect_im)

i = 0
for filename in glob.glob(impath_r):
    if i == 10 :
        break
    i += 1
    real_name = os.path.split(filename)
    save = os.path.join(savepath_r,real_name[1])
    unrect_im = cv2.imread(filename)
    rect_im = cv2.remap(unrect_im, rmapx,rmapy, cv2.INTER_LINEAR)
    Right_rectified.append(rect_im)
    cv2.imwrite(save, rect_im)

i = 0
def write_ply(fn, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')



rleft=[]
rright=[]
for filename in glob.glob(rrect):
    if i == 10 :
        break
    i += 1
    real_name = os.path.split(filename)
    im = cv2.imread(filename)
    rleft.append(im)

i=0
for filename in glob.glob(lrect):
    if i == 10 :
        break
    i += 1
    real_name = os.path.split(filename)
    im = cv2.imread(filename)
    rright.append(im)

window_size = 15


print(len(Left_rectified), len(Right_rectified), len(rleft), len(rright))

stereox = cv2.StereoSGBM_create(
    minDisparity=-1,
    numDisparities=8 * 16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=window_size,
    P1=8 * 3 * window_size,
    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 3 * window_size,
    disp12MaxDiff=16,
    uniquenessRatio=12,
    speckleWindowSize=50,
    speckleRange=32,
    preFilterCap=64,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
for i in range(len(Left_rectified)) :
    disparity = stereox.compute(Left_rectified[i], Right_rectified[i])
    disp2 = stereox.compute(rleft[i], rright[i])

    discolor = disparity.copy()

    scale_percent = 40  # percent of original size
    width = int(imsize[0] * scale_percent / 100)
    height = int(imsize[1] * scale_percent / 100)
    size = (width,height)
    au = cv2.resize(discolor, size, interpolation=cv2.INTER_AREA)
    av = cv2.resize(disp2, size, interpolation=cv2.INTER_AREA)
    a = cv2.hconcat((au,av))

    cv2.imshow("depth", a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



for i in range(len(unrect_left)) :
    scale_percent = 40  # percent of original size
    width = int(imsize[0] * scale_percent / 100)
    height = int(imsize[1] * scale_percent / 100)
    size = (width,height)
    au = cv2.resize(unrect_left[i], size, interpolation=cv2.INTER_AREA)
    av = cv2.resize(Left_rectified[i], size, interpolation=cv2.INTER_AREA)
    aw = cv2.resize(rleft[i], size, interpolation=cv2.INTER_AREA)
    a = cv2.hconcat((au, av))
    b = cv2.hconcat((a, aw))
    cv2.imshow("Comparison", b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



