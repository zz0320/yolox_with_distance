"""
还是双目参数
"""
import numpy as np
import cv2

# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        # 左相机内参
        self.cam_matrix_left = np.array([[707.824075446486,1.07451655669836,601.201405138026],
                                         [0, 705.881696325990, 330.217417382197],
                                         [0., 0., 1.]])
        # 右相机内参
        self.cam_matrix_right = np.array([[711.585683187535, 0.901258032991587, 625.317084919039],
                                          [0, 709.262161181703, 354.966665955608],
                                            [0., 0., 1.]])

        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[0.0893245604683052, -0.126754168307066, -0.00131656067296556, -0.000315713692799473, 0]])
        self.distortion_r = np.array([[-0.149625221344911,-0.126754168307066, -0.00103972763397189, -0.000811782786473007,0]])

        # 旋转矩阵
        self.R = np.array([[0.999991950633517, 0.000288013802500198, 0.00400196404571726],
                           [-0.000304302438593610, 0.999991670613285, 0.00407014791833944],
                           [-0.00400075845303241, -0.00407133296364548, 0.999983708957151]])

        # 平移矩阵
        self.T = np.array([[-62.7132373122585], [-0.0418754846950936], [0.392862452856694]])

        self.size = (1280, 720)

        # 主点列坐标的差
        self.doffs = 0.0

        # 指示上述内外参是否为经过立体校正后的结果
        self.isRectified = False

        self.R1, self.R2, self.P1, self.P2, self.Q, self.validPixROI1, validPixROI2 = \
            cv2.stereoRectify(self.cam_matrix_left, self.distortion_l,
                            self.cam_matrix_right, self.distortion_r, self.size, self.R,
                            self.T)

    def setMiddleBurryParams(self):
        self.cam_matrix_left = np.array([[3997.684, 0, 225.0],
                                         [0., 3997.684, 187.5],
                                         [0., 0., 1.]])
        self.cam_matrix_right = np.array([[3997.684, 0, 225.0],
                                          [0., 3997.684, 187.5],
                                          [0., 0., 1.]])
        self.distortion_l = np.zeros(shape=(5, 1), dtype=np.float64)
        self.distortion_r = np.zeros(shape=(5, 1), dtype=np.float64)
        self.R = np.identity(3, dtype=np.float64)
        self.T = np.array([[-193.001], [0.0], [0.0]])
        self.doffs = 131.111
        self.isRectified = True

