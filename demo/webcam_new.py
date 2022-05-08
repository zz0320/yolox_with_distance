# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import cv2
import camera_config
# import get_distance.PnP.stereoconfig as stereoconfig
import cv2
import torch
import math
import numpy as np
from mmdet.apis import inference_detector, init_detector


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cpu', help='cpu/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args
def onmouse_pick_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        threeD = param
        print('\n像素坐标 x = %d, y = %d' % (x, y))
        # print("世界坐标是：", threeD[y][x][0], threeD[y][x][1], threeD[y][x][2], "mm")
        print("世界坐标xyz 是：", threeD[y][x][0]/ 1000.0, threeD[y][x][1]/ 1000.0, threeD[y][x][2]/ 1000.0, "m")

        distance = math.sqrt( threeD[y][x][0] **2 + threeD[y][x][1] **2 + threeD[y][x][2] **2 )
        distance = distance / 1000.0  # mm -> m
        print("距离是：", distance, "m")

def yolo_pick_points(x, y,  param):
    threeD = param
    # return ('\n像素坐标 x = %d, y = %d' % (x, y))
    # print("世界坐标是：", threeD[y][x][0], threeD[y][x][1], threeD[y][x][2], "mm")
    # print("世界坐标xyz 是：", threeD[y][x][0]/ 1000.0, threeD[y][x][1]/ 1000.0, threeD[y][x][2]/ 1000.0, "m")
    distance = math.sqrt( threeD[y][x][0] **2 + threeD[y][x][1] **2 + threeD[y][x][2] **2 )
    distance = distance / 1000.0  # mm -> m
    str = "距离是：z = %4f m" % (distance)
    return str

def getDepthMapWithQ(disparityMap: np.ndarray, Q: np.ndarray) -> np.ndarray:
  points_3d = cv2.reprojectImageTo3D(disparityMap, Q)
  depthMap = points_3d[:, :, 2]
  reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
  depthMap[reset_index] = 0

  return depthMap.astype(np.float32)

"""
def getDepthMapWithConfig(disparityMap: np.ndarray, config: stereoconfig.stereoCamera) -> np.ndarray:
  fb = config.cam_matrix_left[0, 0] * (-config.T[0])
  doffs = config.doffs
  depthMap = np.divide(fb, disparityMap + doffs)
  reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
  depthMap[reset_index] = 0
  reset_index2 = np.where(disparityMap < 0.0)
  depthMap[reset_index2] = 0
  return depthMap.astype(np.float32)
"""


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = init_detector(args.config, args.checkpoint, device=device)
    camera = cv2.VideoCapture(args.camera_id)
    camera.set(3, 2560)
    camera.set(4, 720)  # 打开并设置摄像头
    WIN_NAME = 'Deep disp'
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        ret_val, frame = camera.read()
        frame1 = frame[0:720, 0:1280]
        frame2 = frame[0:720, 1280:2560]  #割开双目图像
        # 这里选择右路right
        result = inference_detector(model, frame2)

        imageL = cv2.remap(frame1, camera_config.left_map1, camera_config.left_map2, cv2.INTER_LINEAR)
        imageR = cv2.remap(frame2, camera_config.right_map1, camera_config.right_map2, cv2.INTER_LINEAR)

        # 双目匹配SGBM
        blockSize = 7  # 分层
        img_channels = 3
        paraml = {'minDisparity': 0,
                  'numDisparities': 64,  # 64
                  'blockSize': blockSize,
                  'P1': 8 * img_channels * blockSize ** 2,
                  'P2': 32 * img_channels * blockSize ** 2,
                  'disp12MaxDiff': 1,
                  'preFilterCap': 5,
                  'uniquenessRatio': 100,
                  'speckleWindowSize': 800,
                  'speckleRange': 1,
                  'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
                  }
        # https://www.bilibili.com/s/video/BV1Sz4y1m7BW
        left_matcher = cv2.StereoSGBM_create(**paraml)
        paramr = paraml
        paramr['minDisparity'] = -paraml['numDisparities']
        right_matcher = cv2.StereoSGBM_create(**paramr)

        size = (imageL.shape[1], imageL.shape[0])
        disparity_left = left_matcher.compute(imageL, imageR)
        disparity_right = right_matcher.compute(imageR, imageL)

        # 真实视差（因为SGBM算法得到的视差是×16的）
        trueDisp_left = disparity_left.astype(np.float32) / 16.
        trueDisp_right = disparity_right.astype(np.float32) / 16.

        disp = cv2.normalize(trueDisp_left, trueDisp_left, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                             dtype=cv2.CV_8U)  # 归一化函数算法
        threeD = cv2.reprojectImageTo3D(trueDisp_left, camera_config.Q, handleMissingValues=True)  # 计算三维坐标数据值
        # threeD = threeD / 16
        # threeD[y][x] x:0~640; y:0~480;   !!!!!!!!!!
        cv2.setMouseCallback(WIN_NAME, onmouse_pick_points, threeD)

        # 展示检测结果
        model.show_result(
            frame2, result, score_thr=args.score_thr, wait_time=1, show=True)

        # 展示视场差图 和 测距结果
        bbox = np.vstack(result)
        for bbox_line in bbox:
            if bbox_line[-1] > args.score_thr:
                pre_yolo_param = bbox_line
                yolo_param = [(pre_yolo_param[0] + pre_yolo_param[2]) / 2, (pre_yolo_param[1] + pre_yolo_param[3]) / 2]
                res = yolo_pick_points(int(yolo_param[0]), int(yolo_param[1]), threeD)
                print(res)
            else:
                print("没有检测目标")
        # pre_yolo_param = [4.3777963e+02, 1.4396463e+02, 1.1712761e+03, 3.1977725e+02]  # yolo数据接口
        # yolo_param = [(pre_yolo_param[0] + pre_yolo_param[2]) / 2, (pre_yolo_param[1] + pre_yolo_param[3]) / 2]
        cv2.imshow(WIN_NAME, disp)

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break


if __name__ == '__main__':
    main()
