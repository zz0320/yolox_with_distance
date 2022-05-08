# -*- coding: utf-8 -*-
"""
用于双目相机拍照 保存到路径下
"""
import cv2
import time

AUTO = False  # 自动拍照，或手动按s键拍照
INTERVAL = 2  # 自动拍照间隔

cv2.namedWindow("left")
cv2.namedWindow("right")
camera = cv2.VideoCapture(0)

# 设置分辨率左右摄像机同一频率，同一设备ID；左右摄像机总分辨率2560x720；分割为两个1280x720
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

counter = 0
utc = time.time()
folder = "/Users/kenton/Downloads/集群重点研发/蝠鲼数据集/jpg/PnP/Sampler/"  # 拍照文件目录


def shot(pos, frame):
    global counter
    path = folder + pos + "_" + str(counter) + ".jpg"

    cv2.imwrite(path, frame)
    print("snapshot saved into: " + path)


while True:
    ret, frame = camera.read()
    print("ret:", ret)
    # 裁剪坐标为[y0:y1, x0:x1]    HEIGHT * WIDTH
    left_frame = frame[0:720, 0:1280]
    right_frame = frame[0:720, 1280:2560]

    cv2.imshow("left", left_frame)
    cv2.imshow("right", right_frame)

    now = time.time()
    if AUTO and now - utc >= INTERVAL:
        shot("left", left_frame)
        shot("right", right_frame)
        counter += 1
        utc = now

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        shot("left", left_frame)
        shot("right", right_frame)
        counter += 1
camera.release()
cv2.destroyWindow("left")
cv2.destroyWindow("right")