import argparse

parser = argparse.ArgumentParser(description="Mono/Stereo Calibration", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-dpath', '--dpath', metavar="Image Directory", default="M:\\CODE\\Dataset\\Vaccinium\\Color",
                    help = "Directories in which the images are stored")
parser.add_argument('-par', '--par', metavar="Camera Parameters", default="M:\\CODE\\Dataset\\Calibration\\Vaccinium_stereo.matrices",
                    help = "Files which contain the calibrated camera parameters")
parser.add_argument('-s', '--s', metavar="Save Dir", default="M:\\CODE\\Dataset\\Vaccinium\\Rectify",
                    help = "Where to store rectified images")
parser.add_argument('-pc', '--pc', metavar="Point Cloud", default=False, type=bool)
parser.add_argument('-d', '--d', metavar="Depth Dir", default="M:\\CODE\\Dataset\\Vaccinium\\Depth",
                    help = "Disparity images for point cloud, if not available will generate disparity from LR image")



def main() :
    Left, Right, Q = rectify_stereo(args.dpath, args.par, args.s)

    if args.pc == True :
        Depth_Im = []
        if args.d :
            for filename in glob.glob(args.d):
                im = cv2.imread(filename)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                Depth_Im.append(im)
        else :
            for i in range(len(Left)) :
                disp = generate_disparity(Left,Right)
                Depth_Im.append(disp)
        generate_point_cloud(Left,Depth_Im,Q)





def retrieve_and_rectify_and_save(path,x,y,savex, side) :
    Rectified = []
    for filename in glob.glob(os.path.join(path,side,"*.png")):
        real_name = os.path.split(filename)
        save = os.path.join(savex, side)
        if not os.path.exists(save):
            os.makedirs(save)
        save1 = os.path.join(save, real_name[1])
        unrect_im = cv2.imread(filename)
        rect_im = cv2.remap(unrect_im, x, y, cv2.INTER_LINEAR)
        Rectified.append(rect_im)
        cv2.imwrite(save1, rect_im)
    return Rectified

def rectify_stereo(impath, param_path, save_path) :
    with open(param_path, "rb") as handle:
        cmtx = pickle.load(handle)

    K1 = cmtx['K_Left']
    D1 = cmtx['D_Left']
    K2 = cmtx['K_Right']
    D2 = cmtx['D_Right']
    imsize = (2048, 1536)
    R = cmtx['Rrect']
    T = cmtx['Trect']

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, imsize, R, T, flags=cv2.CALIB_ZERO_DISPARITY,
                                                      alpha=-1)

    lmapx, lmapy = cv2.initUndistortRectifyMap(K1, D1, R1, P1, imsize, cv2.CV_32FC1)
    rmapx, rmapy = cv2.initUndistortRectifyMap(K2, D2, R2, P2, imsize, cv2.CV_32FC1)

    Rectified_left = retrieve_and_rectify_and_save(impath, lmapx, lmapy, save_path, side = "Left")
    Rectified_right = retrieve_and_rectify_and_save(impath, rmapx, rmapy, save_path, side="Right")
    return Rectified_left, Rectified_right, Q

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

def generate_disparity(Left_Image, Right_Image) :
    window_size = 15
    disp = []
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

    for i in range(len(Left_Image)) :
        disparity = stereox.compute(Left_Image[i], Right_Image[i])
        disp.append(disparity)
    return disp

def generate_point_cloud(Rectified, Depth, Q) :
    for i in range(len(Rectified)) :
        points_3d = cv2.reprojectImageTo3D(Depth[i], Q)
        colors = cv2.cvtColor(Rectified[i], cv2.COLOR_BGR2RGB)

        # reflect on x axis
        reflect_matrix = np.identity(3)
        reflect_matrix[0] *= -1
        points = np.matmul(points_3d, reflect_matrix)

        # extract colors from image
        colors = cv2.cvtColor(Rectified[i], cv2.COLOR_BGR2RGB)

        # filter by min disparity
        mask = Depth[i] > Depth[i].min()
        out_points = points[mask]
        out_colors = colors[mask]

        # filter by dimension
        idx = np.fabs(out_points[:, 0]) < 28
        out_points = out_points[idx]
        out_colors = out_colors.reshape(-1, 3)
        out_colors = out_colors[idx]

        write_ply("Pointcloud_{0}.ply".format(i), out_points, out_colors)


if __name__ == '__main__':
    import cv2
    import glob
    import os
    import numpy as np
    import pickle

    args = parser.parse_args()

    main()



