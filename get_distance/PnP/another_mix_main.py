# -*- coding: utf-8 -*-
"""
一个老的测距程序 还有点云
"""
import sys
import cv2
import numpy as np
import stereoconfig
import pcl
import math
# import pcl.pcl_visualization
import open3d as o3d
def onmouse_pick_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        threeD = param
        print('\n像素坐标 x = %d, y = %d' % (x, y))
        # print("世界坐标是：", threeD[y][x][0], threeD[y][x][1], threeD[y][x][2], "mm")
        print("世界坐标xyz 是：", threeD[y][x][0]/ 1000.0, threeD[y][x][1]/ 1000.0, threeD[y][x][2]/ 1000.0, "m")

        distance = math.sqrt( threeD[y][x][0] **2 + threeD[y][x][1] **2 + threeD[y][x][2] **2 )
        distance = distance / 1000.0  # mm -> m
        print("距离是：", distance, "m")

WIN_NAME = 'Deep disp'
WIN_NAME_2 = 'Deep_real'
cv2.namedWindow(WIN_NAME,  cv2.WINDOW_AUTOSIZE)
cv2.namedWindow(WIN_NAME_2,  cv2.WINDOW_AUTOSIZE)
# 预处理
def preprocess(img1, img2):
    # 彩色图->灰度图
    if (img1.ndim == 3):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
    if (img2.ndim == 3):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 直方图均衡
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)

    return img1, img2


# 消除畸变
def undistortion(image, camera_matrix, dist_coeff):
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)

    return undistortion_image


# 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
# @param：config是一个类，存储着双目标定的参数:config = stereoconfig.stereoCamera()
def getRectifyTransform(height, width, config):
    # 读取内参和外参
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    # 计算校正变换
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                      (width, height), R, T, alpha=0)

    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q


# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)

    return rectifyed_img1, rectifyed_img2


# 立体校正检验----画线
def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    # 绘制等间距平行线
    line_interval = 50  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)

    return output


# 视差计算
def stereoMatchSGBM(left_image, right_image, down_scale=False):
    # SGBM匹配参数设置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 3
    paraml = {'minDisparity': 0,
              'numDisparities': 64,
              'blockSize': blockSize,
              'P1': 8 * img_channels * blockSize ** 2,
              'P2': 32 * img_channels * blockSize ** 2,
              'disp12MaxDiff': 10,
              'preFilterCap': 63,
              'uniquenessRatio': 15,
              'speckleWindowSize': 100,
              'speckleRange': 1,
              'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
              }

    # 构建SGBM对象
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)

    # 计算视差图
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)
    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]

        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right

    # 真实视差（因为SGBM算法得到的视差是×16的）
    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.

    return trueDisp_left, trueDisp_right


# 将h×w×3数组转换为N×3的数组
def hw3ToN3(points):
    height, width = points.shape[0:2]

    points_1 = points[:, :, 0].reshape(height * width, 1)
    points_2 = points[:, :, 1].reshape(height * width, 1)
    points_3 = points[:, :, 2].reshape(height * width, 1)

    points_ = np.hstack((points_1, points_2, points_3))

    return points_


# 深度、颜色转换为点云
def DepthColor2Cloud(points_3d, colors):
    rows, cols = points_3d.shape[0:2]
    size = rows * cols

    points_ = hw3ToN3(points_3d)
    colors_ = hw3ToN3(colors).astype(np.int64)

    # 颜色信息
    blue = colors_[:, 0].reshape(size, 1)
    green = colors_[:, 1].reshape(size, 1)
    red = colors_[:, 2].reshape(size, 1)

    rgb = np.left_shift(blue, 0) + np.left_shift(green, 8) + np.left_shift(red, 16)

    # 将坐标+颜色叠加为点云数组
    pointcloud = np.hstack((points_, rgb)).astype(np.float32)

    # 删掉一些不合适的点
    X = pointcloud[:, 0]
    Y = pointcloud[:, 1]
    Z = pointcloud[:, 2]

    # 下面参数是经验性取值，需要根据实际情况调整
    remove_idx1 = np.where(Z <= 0)
    remove_idx2 = np.where(Z > 15000)
    remove_idx3 = np.where(X > 10000)
    remove_idx4 = np.where(X < -10000)
    remove_idx5 = np.where(Y > 10000)
    remove_idx6 = np.where(Y < -10000)
    remove_idx = np.hstack(
        (remove_idx1[0], remove_idx2[0], remove_idx3[0], remove_idx4[0], remove_idx5[0], remove_idx6[0]))

    pointcloud_1 = np.delete(pointcloud, remove_idx, 0)

    return pointcloud_1


# 点云显示
def view_cloud(pointcloud):
    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_array(pointcloud)

    try:
        visual = pcl.pcl_visualization.CloudViewing()
        visual.ShowColorACloud(cloud)
        v = True
        while v:
            v = not (visual.WasStopped())
    except:
        pass


def getDepthMapWithQ(disparityMap: np.ndarray, Q: np.ndarray) -> np.ndarray:
    points_3d = cv2.reprojectImageTo3D(disparityMap, Q)
    depthMap = points_3d[:, :, 2]
    reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
    depthMap[reset_index] = 0

    return depthMap.astype(np.float32)


def getDepthMapWithConfig(disparityMap: np.ndarray, config: stereoconfig.stereoCamera) -> np.ndarray:
    fb = config.cam_matrix_left[0, 0] * (-config.T[0])
    doffs = config.doffs
    depthMap = np.divide(fb, disparityMap + doffs)
    reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
    depthMap[reset_index] = 0
    reset_index2 = np.where(disparityMap < 0.0)
    depthMap[reset_index2] = 0
    return depthMap.astype(np.float32)


if __name__ == '__main__':
    # 读取MiddleBurry数据集的图片
    cap = cv2.VideoCapture(0)
    cap.set(3, 2560)
    cap.set(4, 720)  # 打开并设置摄像头
    WIN_NAME = 'Deep disp'
    WIN_NAME_2 = 'Deep_real'
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WIN_NAME_2, cv2.WINDOW_AUTOSIZE)
    while True:
        ret, frame = cap.read()
        iml = frame[0:720, 0:1280]
        imr = frame[0:720, 1280:2560]  # 割开双目图像
        # imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # 将BGR格式转换成灰度图片
        # imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # iml = cv2.imread('/Users/kenton/Downloads/集群重点研发/蝠鲼数据集/jpg/PnP/Sampler/camL/left_0.jpg', 1)  # 左图
        # imr = cv2.imread('/Users/kenton/Downloads/集群重点研发/蝠鲼数据集/jpg/PnP/Sampler/camR/right_0.jpg', 1)  # 右图
        if (iml is None) or (imr is None):
            print("Error: Images are empty, please check your image's path!")
            sys.exit(0)
        height, width = iml.shape[0:2]

        # 读取相机内参和外参
        # 使用之前先将标定得到的内外参数填写到stereoconfig.py中的StereoCamera类中
        config = stereoconfig.stereoCamera()
        config.setMiddleBurryParams()
        # print(config.cam_matrix_left)

        # 立体校正
        map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)  # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
        iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
        # print(Q)

        # 绘制等间距平行线，检查立体校正的效果
        line = draw_line(iml_rectified, imr_rectified)
        # cv2.imwrite('/Users/kenton/Downloads/集群重点研发/蝠鲼数据集/jpg/PnP/Sampler/check.jpg', line)

        # 立体匹配
        iml_, imr_ = preprocess(iml, imr)  # 预处理，一般可以削弱光照不均的影响，不做也可以
        disp, _ = stereoMatchSGBM(iml, imr, False)  # 这里传入的是未经立体校正的图像，因为我们使用的middleburry图片已经是校正过的了
        # cv2.imwrite('/Users/kenton/Downloads/集群重点研发/蝠鲼数据集/jpg/PnP/Sampler/disaprity.png', disp * 4)

        # 计算深度图
        # depthMap = getDepthMapWithQ(disp, Q)
        depthMap = getDepthMapWithConfig(disp, config)
        minDepth = np.min(depthMap)
        maxDepth = np.max(depthMap)
        # print(minDepth, maxDepth)
        depthMapVis = (255.0 * (depthMap - minDepth)) / (maxDepth - minDepth)
        depthMapVis = depthMapVis.astype(np.uint8)
        # Q_get = stereoconfig.stereoCamera()
        threeD = cv2.reprojectImageTo3D(disp, Q, handleMissingValues=True)  # 计算三维坐标数据值
        threeD = threeD * 16
        # cv2.waitKey(0)
        cv2.setMouseCallback(WIN_NAME, onmouse_pick_points, threeD)
        cv2.imshow("left", iml)
        cv2.imshow(WIN_NAME, disp)
        cv2.imshow("DepthMap", depthMapVis)


        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    """
        # 使用open3d库绘制点云
        colorImage = o3d.geometry.Image(iml)
        depthImage = o3d.geometry.Image(depthMap)
        rgbdImage = o3d.geometry.RGBDImage().create_from_color_and_depth(colorImage, depthImage, depth_scale=1000.0,
                                                                         depth_trunc=np.inf)
        intrinsics = o3d.camera.PinholeCameraIntrinsic()
        # fx = Q[2, 3]
        # fy = Q[2, 3]
        # cx = Q[0, 3]
        # cy = Q[1, 3]
        fx = config.cam_matrix_left[0, 0]
        fy = fx
        cx = config.cam_matrix_left[0, 2]
        cy = config.cam_matrix_left[1, 2]
        print(fx, fy, cx, cy)
        intrinsics.set_intrinsics(width, height, fx=fx, fy=fy, cx=cx, cy=cy)
        extrinsics = np.array([[1., 0., 0., 0.],
                               [0., 1., 0., 0.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]])
        pointcloud = o3d.geometry.PointCloud().create_from_rgbd_image(rgbdImage, intrinsic=intrinsics, extrinsic=extrinsics)
        o3d.io.write_point_cloud("PointCloud.pcd", pointcloud=pointcloud)
        o3d.visualization.draw_geometries([pointcloud], width=720, height=480)
        sys.exit(0)
    
        # 计算像素点的3D坐标（左相机坐标系下）
        points_3d = cv2.reprojectImageTo3D(disp, Q)  # 参数中的Q就是由getRectifyTransform()函数得到的重投影矩阵
    
        # 构建点云--Point_XYZRGBA格式
        pointcloud = DepthColor2Cloud(points_3d, iml)
    
        # 显示点云
        # view_cloud(points_3d)
    """
    cap.release()
    cv2.destroyAllWindows()