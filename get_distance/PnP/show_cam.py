"""
匹配效果 可视化调参
"""
from cv2 import cv2
import numpy as np
import camera_configs  # 摄像头的标定数据

cam1 = cv2.VideoCapture(0)  # 摄像头的ID不同设备上可能不同
# cam2 = cv2.VideoCapture(0)  # 摄像头的ID不同设备上可能不同
# cam1 = cv2.VideoCapture(1 + cv2.CAP_DSHOW)  # 摄像头的ID不同设备上可能不同
# cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 设置双目的宽度
# cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 设置双目的高度

# 创建用于显示深度的窗口和调节参数的bar
cv2.namedWindow("depth")
cv2.moveWindow("left", 0, 0)
cv2.moveWindow("right", 600, 0)

# 创建用于显示深度的窗口和调节参数的bar
# cv2.namedWindow("depth")
cv2.namedWindow("config", cv2.WINDOW_NORMAL)
cv2.moveWindow("left", 0, 0)
cv2.moveWindow("right", 600, 0)

cv2.createTrackbar("num", "config", 0, 60, lambda x: None)
cv2.createTrackbar("blockSize", "config", 30, 255, lambda x: None)
cv2.createTrackbar("SpeckleWindowSize", "config", 1, 10, lambda x: None)
cv2.createTrackbar("SpeckleRange", "config", 1, 255, lambda x: None)
cv2.createTrackbar("UniquenessRatio", "config", 1, 255, lambda x: None)
cv2.createTrackbar("TextureThreshold", "config", 1, 255, lambda x: None)
cv2.createTrackbar("UniquenessRatio", "config", 1, 255, lambda x: None)
cv2.createTrackbar("MinDisparity", "config", 0, 255, lambda x: None)
cv2.createTrackbar("PreFilterCap", "config", 1, 65, lambda x: None)  # 注意调节的时候这个值必须是奇数
cv2.createTrackbar("MaxDiff", "config", 1, 400, lambda x: None)


# 添加点击事件，打印当前点的距离
def callbackFunc(e, x, y, f, p):
    if e == cv2.EVENT_LBUTTONDOWN:
        print(threeD[y][x])
        if abs(threeD[y][x][2]) < 3000:
            print("当前距离:" + str(abs(threeD[y][x][2])))
        else:
            print("当前距离过大或请点击色块的位置")


cv2.setMouseCallback("depth", callbackFunc, None)

# 初始化计算FPS需要用到参数 注意千万不要用opencv自带fps的函数，那个函数得到的是摄像头最大的FPS
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

imageCount = 1

while True:
    t1 = cv2.getTickCount()
    ret1, frame = cam1.read()
    # ret1, frame2 = cam2.read()

    if not ret1:
        print("camera is not connected!")
        break

    # 这里的左右两个摄像头的图像是连在一起的，所以进行一下分割
    frame1 = frame[0:720, 0:1280]
    frame2 = frame[0:720, 1280:2560]

    ####### 深度图测量开始 #######
    # 立体匹配这里使用BM算法，

    # 根据标定数据对图片进行重构消除图片的畸变
    img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR,
                               cv2.BORDER_CONSTANT)
    img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR,
                               cv2.BORDER_CONSTANT)

    # 如有些版本 remap()的图是反的 这里对角翻转一下
    # img1_rectified = cv2.flip(img1_rectified, -1)
    # img2_rectified = cv2.flip(img2_rectified, -1)

    # 将图片置为灰度图，为StereoBM作准备，BM算法只能计算单通道的图片，即灰度图
    # 单通道就是黑白的，一个像素只有一个值如[123]，opencv默认的是BGR(注意不是RGB), 如[123,4,134]分别代表这个像素点的蓝绿红的值
    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

    out = np.hstack((img1_rectified, img2_rectified))
    for i in range(0, out.shape[0], 30):
        cv2.line(out, (0, i), (out.shape[1], i), (0, 255, 0), 1)
    cv2.imshow("epipolar lines", out)

    # 通过bar来获取到当前的参数
    # BM算法对参数非常敏感，一定要耐心调整适合自己摄像头的参数，前两个参数影响大 后面的参数也要调节
    num = cv2.getTrackbarPos("num", "config")
    SpeckleWindowSize = cv2.getTrackbarPos("SpeckleWindowSize", "config")
    SpeckleRange = cv2.getTrackbarPos("SpeckleRange", "config")
    blockSize = cv2.getTrackbarPos("blockSize", "config")
    UniquenessRatio = cv2.getTrackbarPos("UniquenessRatio", "config")
    TextureThreshold = cv2.getTrackbarPos("TextureThreshold", "config")
    MinDisparity = cv2.getTrackbarPos("MinDisparity", "config")
    PreFilterCap = cv2.getTrackbarPos("PreFilterCap", "config")
    MaxDiff = cv2.getTrackbarPos("MaxDiff", "config")
    if blockSize % 2 == 0:
        blockSize += 1
    if blockSize < 5:
        blockSize = 5

    # 根据BM算法生成深度图的矩阵，也可以使用SGBM，SGBM算法的速度比BM慢，但是比BM的精度高
    stereo = cv2.StereoBM_create(
        numDisparities=16 * num,
        blockSize=blockSize,
    )
    stereo.setROI1(camera_configs.validPixROI1)
    stereo.setROI2(camera_configs.validPixROI2)
    stereo.setPreFilterCap(PreFilterCap)
    stereo.setMinDisparity(MinDisparity)
    stereo.setTextureThreshold(TextureThreshold)
    stereo.setUniquenessRatio(UniquenessRatio)
    stereo.setSpeckleWindowSize(SpeckleWindowSize)
    stereo.setSpeckleRange(SpeckleRange)
    stereo.setDisp12MaxDiff(MaxDiff)

    # 对深度进行计算，获取深度矩阵
    disparity = stereo.compute(imgL, imgR)
    # 按照深度矩阵生产深度图
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 将深度图扩展至三维空间中，其z方向的值则为当前的距离
    threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., camera_configs.Q)
    # 将深度图转为伪色图，这一步对深度测量没有关系，只是好看而已
    fakeColorDepth = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

    cv2.putText(frame1, "FPS: {0:.2f}".format(frame_rate_calc), (30, 50), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # 按下S可以保存图片
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # 按下ESC退出程序
        break

    if interrupt & 0xFF == ord('s'):
        cv2.imwrite('images/left' + '.jpg', frame1)
        cv2.imwrite('images/right' + '.jpg', frame2)
        cv2.imwrite('images/img1_rectified' + '.jpg', img1_rectified)  # 畸变，注意观察正反
        cv2.imwrite('images/img2_rectified' + '.jpg', img2_rectified)
        cv2.imwrite('images/depth' + '.jpg', disp)
        cv2.imwrite('images/fakeColor' + '.jpg', fakeColorDepth)
        cv2.imwrite('mages/epipolar' + '.jpg', out)

    ####### 任务1：测距结束 #######

    # 显示
    # cv2.imshow("frame", frame) # 原始输出，用于检测左右
    # cv2.imshow("frame1", frame1)  # 左边原始输出
    # cv2.imshow("frame2", frame2)  # 右边原始输出
    # cv2.imshow("img1_rectified", img1_rectified)  # 左边矫正后输出
    # cv2.imshow("img2_rectified", img2_rectified)  # 右边边矫正后输出
    cv2.imshow("depth", disp)  # 输出深度图及调整的bar
    # cv2.imshow("fakeColor", fakeColorDepth)  # 输出深度图的伪色图，这个图没有用只是好看

    # 需要对深度图进行滤波将下面几行开启即可 开启后FPS会降低
    img_medianBlur = cv2.medianBlur(disp, 25)
    img_medianBlur_fakeColorDepth = cv2.applyColorMap(img_medianBlur, cv2.COLORMAP_JET)
    img_GaussianBlur = cv2.GaussianBlur(disp, (7, 7), 0)
    img_Blur = cv2.blur(disp, (5, 5))
    # cv2.imshow("img_GaussianBlur", img_GaussianBlur)  # 右边原始输出
    # cv2.imshow("img_medianBlur_fakeColorDepth", img_medianBlur_fakeColorDepth)  # 右边原始输出
    # cv2.imshow("img_Blur", img_Blur)  # 右边原始输出
    # cv2.imshow("img_medianBlur", img_medianBlur)  # 右边原始输出

    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1
    key = cv2.waitKey(1)

    if key == ord("q"):
        break
cam1.release()
cv2.destroyAllWindows()