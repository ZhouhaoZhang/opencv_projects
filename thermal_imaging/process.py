import cv2
import numpy as np
import time

def video_init(video):
    vc = cv2.VideoCapture(video)
    if vc.isOpened():
        op = True
    else:
        op = False
    return vc, op


def setmod():
    mode = int(input("请输入模式：1代表凸显低温，2代表凸显高温，3代表凸显区间"))
    return mode


def tem2bright(tempr):
    ref = cv2.imread("ref.png", 0)
    thlist = ref[int((60 - tempr) / 45 * 440)]
    sum = np.sum(thlist)
    th = sum / 20
    return th


def display_low(roi, th):
    # B，R通道低于阈值的变成黑色，高于阈值的不变
    ret, roib = cv2.threshold(roi, th, 255, cv2.THRESH_TOZERO)
    roir = roib
    roig = roi
    res = cv2.merge((roib, roig, roir))
    cv2.imshow("res", res)


def display_high(roi, th):
    # B，G通道高于阈值的变成黑色，低于阈值的不变
    ret, roib = cv2.threshold(roi, th, 255, cv2.THRESH_TOZERO_INV)
    roig = roib
    roir = roi
    res = cv2.merge((roib, roig, roir))
    cv2.imshow("res", res)


def display_range(roi, thl, thh):
    ret, roil = cv2.threshold(roi, thh, 255, cv2.THRESH_TOZERO)
    ret, roih = cv2.threshold(roi, thl, 255, cv2.THRESH_TOZERO_INV)
    roir = roih + roil
    roig = roir
    res = cv2.merge((roi, roig, roir))
    cv2.imshow("res", res)


"""
主程序
"""
# 视频初始化
vc, op = video_init("test.mp4")

# 设置模式
mod = setmod()
# 设定温度
if mod == 1 or mod == 2:
    tem = float(input("输入温度"))
    th = tem2bright(tem)
elif mod == 3:
    teml = float(input("输入下限温度"))
    thl = tem2bright(teml)
    temh = float(input("输入上限温度"))
    thh = tem2bright(temh)

# 循环读取每一帧
while op:
    ret, frame = vc.read()
    if frame is None:
        break
    roibgr = frame[50:670, 0:425]
    roi = cv2.cvtColor(roibgr, cv2.COLOR_BGR2GRAY)
    if mod == 1:
        display_low(roi, th)
    elif mod == 2:
        display_high(roi, th)
    elif mod == 3:
        display_range(roi, thl, thh)

    if cv2.waitKey(1) & 0xFF == 27:
        break
