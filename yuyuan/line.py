import cv2
from OpenCV_learning.generalFunctions import show
import numpy as np
import math

"""
对边线检测Y，角度
"""
"""
选取上半部分ROI，填充，角点检测，解算Y和角度
"""
"""
角度为负，则需要顺时针旋转
"""
k = 0.775
img = cv2.imread("64.jpg", 0)
roi = img[0:220, :]
top_size, bottom_size, right_size, left_size = (0, 260, 100, 100)
roi = cv2.copyMakeBorder(roi, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=255)
show("roi", roi)
ret1, bin = cv2.threshold(roi, 85, 255, cv2.THRESH_BINARY_INV)
show("bin", bin)
kernel = np.ones((15, 15), np.uint8)
bin_open = cv2.morphologyEx(bin, cv2.MORPH_OPEN, kernel)
show("open", bin_open)
cnts, ret2 = cv2.findContours(bin_open, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cntsorted = sorted(cnts, key=lambda cnts: cv2.contourArea(cnts), reverse=True)
cnt = cntsorted[0]
epsilon = 0.03 * cv2.arcLength(cnt, True)  # epsilon占周长的比例
approx = cv2.approxPolyDP(cnt, epsilon, True)
x, y, w, h = cv2.boundingRect(cnt)
[VX, VY, X, Y] = cv2.fitLine(cnt, cv2.DIST_HUBER, 0, 0.01, 0.01)
print(VX, VY, X, Y)
a = -(math.atan(VY / VX) / 6.28 * 360)

print(a)
distance = -(k * ((y + h / 2) - 240))
print(distance)
