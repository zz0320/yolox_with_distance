"""
梅开二度 还是相机参数
"""
import numpy as np
import cv2

left_camera_matrix = np.array([[707.824075446486,1.07451655669836,601.201405138026],
                                         [0, 705.881696325990, 330.217417382197],
                                         [0., 0., 1.]])

# 左相机畸变系数:[k1, k2, p1, p2, k3]
left_distortion = np.array([[0.0893245604683052, -0.126754168307066, -0.00131656067296556, -0.000315713692799473, 0]])


# 右相机内参
right_camera_matrix = np.array([[711.585683187535, 0.901258032991587, 625.317084919039],
                                          [0, 709.262161181703, 354.966665955608],
                                            [0., 0., 1.]])
# 右相机畸变系数:[k1, k2, p1, p2, k3]
right_distortion = np.array([[0.102141197750782, -0.149625221344911, -0.00103972763397189, -0.000811782786473007,  0]])


# om = np.array([-0.00009, 0.02300, -0.00372])
# R = cv2.Rodrigues(om)[0]

# 旋转矩阵
R = np.array([[0.999991950633517,-0.000304302438593610, -0.00400075845303241],
                           [-0.000288013802500198, 0.999991670613285, -0.00407133296364548],
                           [0.00400196404571726, 0.00407014791833944, 0.999983708957151]])

# 平移向量
T = np.array([[-62.7132373122585], [-0.0418754846950936], [0.392862452856694]])

size = (1280, 720)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)

left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)