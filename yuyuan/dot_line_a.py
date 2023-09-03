import cv2
from OpenCV_learning.generalFunctions import show
from matplotlib import pyplot as plt
import numpy as np
import math

k = 0.78125


def display(bedrew, result):
    b, g, r = cv2.split(bedrew)  # 颜色通道提取
    match_result_rgb = cv2.merge((r, g, b))  # 颜色通道合并
    plt.subplot(121), plt.imshow(result)
    plt.subplot(122), plt.imshow(match_result_rgb)
    plt.suptitle("cv2.TM_SQDIFF_NORMED")
    plt.show()


imgbgr = cv2.imread("47.jpg")
img = cv2.imread("47.jpg", 0)
template = cv2.imread("template5cm.png", 0)
roi = img[0:270, :]
res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
(X, Y) = min_loc
(y, x) = template.shape
match_result = cv2.rectangle(imgbgr, (X, Y), (X + x, Y + y), (0, 0, 255), 8)
display(match_result, res)
roi_line = img[0:172, X + x:X + x + 234]
show("roi", roi_line)
ret1, bin = cv2.threshold(roi_line, 85, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((15, 15), np.uint8)
bin_open = cv2.morphologyEx(bin, cv2.MORPH_OPEN, kernel)
show("open", bin_open)
cnts, ret2 = cv2.findContours(bin_open, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cntsorted = sorted(cnts, key=lambda cnts: cv2.contourArea(cnts), reverse=True)
cnt = cntsorted[0]
epsilon = 0.03 * cv2.arcLength(cnt, True)  # epsilon占周长的比例
approx = cv2.approxPolyDP(cnt, epsilon, True)
x, y, w, h = cv2.boundingRect(cnt)
[VX, VY, Xk, Yk] = cv2.fitLine(cnt, cv2.DIST_HUBER, 0, 0.01, 0.01)
print(VX, VY, Xk, Yk)
a = -(math.atan(VY / VX) / 6.28 * 360)

print(a, "度")


