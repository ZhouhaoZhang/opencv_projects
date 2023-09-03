import cv2
# import matplotlib.pyplot as plt
import numpy as np

from OpenCV_learning.generalFunctions import show

img = cv2.imread("54.jpg")
# cv2.imshow("res",img)
"""
1像素=k毫米
"""

k = 0.78125
# img = cv2.imread("images/2.jpg")
# print(img.shape)
# img_gray = cv2.imread("images/2.jpg", 0)
# img = img[0:190, 110:320]
# img_gray = img_gray[0:190, 110:320]
"""
提取饱和度通道
"""
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hh, s, v = cv2.split(img_hsv)
show("s", s)
key = s
"""
高斯滤波
"""
key_blur = cv2.GaussianBlur(key, (3, 3), 1)
show("s", s)
"""
二值化
"""
ret1, key_bin = cv2.threshold(key_blur, 127, 255, cv2.THRESH_BINARY)
show("bin", key_bin)
"""
开运算
"""
kernel = np.ones((30, 30), np.uint8)
key_bin_open = cv2.morphologyEx(key_bin, cv2.MORPH_OPEN, kernel)
show("bin_Open", key_bin_open)
"""
寻找边界，排序
"""
cnts, ret2 = cv2.findContours(key_bin_open, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cntsorted = sorted(cnts, key=lambda cnts: cv2.contourArea(cnts), reverse=True)
cnt = cntsorted[0]
"""
roi
"""
x, y, w, h = cv2.boundingRect(cnt)
bin_roi = key_blur[y:y + h, x:x + w]
show("roi", bin_roi)
"""
角点检测
"""
corners = cv2.goodFeaturesToTrack(bin_roi, 7, 0.03, 5, useHarrisDetector=False)
# print(corners)
"""

eps = 0 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, eps, True)
res = cv2.drawContours(img, [approx], -1, (0, 0, 255), 1)

"""

"""

for keypoint in corners:
    if keypoint is not None and len(keypoint) > 0:
        for xroi, yroi in keypoint:
            # cv2.circle(img, (int(xroi) + x - 20, int(yroi) + y - 20), 3, (0, 0, 255), thickness=3)
            cv2.circle(img, (x + w/2, y+h/2), 3, (0, 0, 255), thickness=3)
# print(x,y)



"""
x = float(x)
y = float(y)
cv2.circle(img, (int(x + w / 2), int(y + h / 2)), 3, (0, 0, 255), thickness=2)
# print(x, y)
x_fixed = int(320 + 0.93 * (x + w / 2 - 320))
y_fixed = int(240 + 0.93 * (y + h / 2 - 240))

cv2.circle(img, (x_fixed, y_fixed), 3, (0, 255, 0), thickness=2)
X = (x_fixed - 320) * k
Y = (y_fixed - 320) * k

show("res", img)
