"""
双目录像
"""
import cv2


def getCam():
    window_name = 'show image'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    video_cap_obj = cv2.VideoCapture(0)
    if video_cap_obj == None:
        print('video caoture error')
    if video_cap_obj.open(0) == False:
        print('open error')
    while True:
        retval, image = video_cap_obj.read()
        cv2.ShowImage(window_name, cv2.fromarray(image))
        if cv2.waitKey(10) == 27:
            break
    video_cap_obj.release()


if __name__ == '__main__':
    getCam()