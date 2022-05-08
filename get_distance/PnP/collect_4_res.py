#!/usr/bin/env python
"""
用来看看p3p四个解的程序
"""
import cv2
import numpy as np
import math
point3s=np.array(([0, 0, 0],[21, 0, 0],[21, 12, 0],[0, 12, 0]),dtype=np.double)
point3s_p3p=np.array(([0, 0, 0],[21, 0, 0],[21, 12, 0]),dtype=np.double)

point2s=np.array(([1014, 419],[1466, 477],[1466, 742],[1014, 742]),dtype=np.double)
point2s_p3p=np.array(([1014, 419],[1466, 477],[1466, 742]),dtype=np.double)

camera=np.array(([1.2675e+03, -1.0453, 1.1925e+03], [0, 1.2528e+03, 780.7102], [0, 0, 1]),dtype=np.double)
# camera=np.array(([1.2675e+03, 0, 0], [-1.0453, 1.2528e+03, 0], [1.1925e+03, 780.7102, 1]),dtype=np.double)
dist=np.array(([-0.0045, -0.0350, -8.8150e-04, -0.6492, 0.2280]),dtype=np.double)
# 先横向 再纵向
# https://blog.csdn.net/qq_43742590/article/details/104109103
# dist=dist.T
#dist=np.zeros((5,1))
# list = cv2.solvePnP(point3s, point2s, camera, dist) #计算雷达相机外参,r-旋转向量，t-平移向量 flags=cv2.SOLVEPNP_P3P
#R=cv2.Rodrigues(r)[0] #旋转向量转旋转矩阵
#camera_position=-np.matrix(R).T*np.matrix(t) #相机位置
# print(list[0])
# print("result1:",list[1])
# print("result3:",list[2])

# d3=np.array([[-3.14925, -1.54094, -1.06652]])
# d2,_=cv2.projectPoints(d3,r,t,camera,dist)#重投影验证
# print(r)
# print(t)
# print(d2)

result1 = cv2.solvePnP(point3s,point2s,camera,dist, flags=cv2.SOLVEPNP_UPNP) #计算雷达相机外参,r-旋转向量，t-平移向量
result2 = cv2.solvePnP(point3s,point2s,camera,dist, flags=cv2.SOLVEPNP_SQPNP) #计算雷达相机外参,r-旋转向量，t-平移向量

# list = cv2.solveP3P(point3s_p3p, point2s_p3p, camera, dist, flags=cv2.SOLVEPNP_P3P) #计算雷达相机外参,r-旋转向量，t-平移向量
#R=cv2.Rodrigues(r)[0] #旋转向量转旋转矩阵
#camera_position=-np.matrix(R).T*np.matrix(t) #相机位置
# print(result1[1],'\n','-------','\n', result1[2])
result3 = cv2.solvePnP(point3s_p3p,point2s_p3p,camera,dist, flags=cv2.SOLVEPNP_SQPNP) #计算雷达相机外参,r-旋转向量，t-平移向量
print(result3)
list = cv2.solveP3P(point3s_p3p, point2s_p3p, camera, dist, flags=cv2.SOLVEPNP_P3P) #计算雷达相机外参,r-旋转向量，t-平移向量
print(list)
"""print(list[1][0] * 180 / 3.14, list[2][0])
print("------------")
print(list[1][1] * 180 / 3.14,list[2][0])
print("------------")
print(list[1][0] * 180 / 3.14, list[2][1])
print("------------")
print(list[1][1] * 180 / 3.14, list[2][1])
print("------------")
"""