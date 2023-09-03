import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
from OpenCV_learning.generalFunctions import show
img = cv2.imread('bill.png')
img_gray = cv2.imread('bill.png', 0)
# show('result', img)
# show('res', img_gray)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
# show('gauss', img_blur)
edges = cv2.Canny(img_blur, 50, 200)
plt.imshow(edges, 'gray'), plt.title('Canny')
plt.show()
# show('canny', edges)
kernel = np.ones((2, 2), np.uint8)
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
edges_close = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, rectKernel)
plt.imshow(edges_close, 'gray'), plt.title('Close')
plt.show()
edges_dilate = cv2.dilate(edges_close, kernel, iterations=3)
plt.imshow(edges_dilate, 'gray'), plt.title('Dilate')
plt.show()
# show('dilate', edges_dialate)

# show('close', edges_close)
# show('dilate', edges_dialate)

contours, hierarchy = cv2.findContours(edges_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=lambda cnts: cv2.arcLength(cnts, True), reverse=True)

img_copy = img.copy()
res = cv2.drawContours(img_copy, contours, 0, (0, 0, 255), 2)
# show('res', res)
img_copy = img.copy()
cnt = contours[0]
epsilon = 0.03 * cv2.arcLength(cnt, True)  # epsilon占周长的比例
approx = cv2.approxPolyDP(cnt, epsilon, True)
res2 = cv2.drawContours(img_copy, [approx], -1, (0, 0, 255), 5)
# print(approx)
show('res2', res2)
[[lt], [lb], [rb], [rt]] = approx
# print(lt, lb, rb, rt)
[ltx, lty] = lt
[lbx, lby] = lb
[rbx, rby] = rb
[rtx, rty] = rt
# print(ltx, lty, lbx, lby, rbx, rby, rtx, rty)
lt = (ltx, lty)
lb = (lbx, lby)
rb = (rbx, rby)
rt = (rtx, rty)
# print(lt, lb, rb, rt)
# 仿射变换
width = max(math.sqrt((rtx - ltx) ** 2 + (rty - lty) ** 2), math.sqrt((rbx - lbx) ** 2 + (rby - lby) ** 2))
height = max(math.sqrt((ltx - lbx) ** 2 + (lty - lby) ** 2), math.sqrt((rtx - rbx) ** 2 + (rty - rby) ** 2))
pts1 = np.float32([[ltx, lty], [rtx, rty], [lbx, lby], [rbx, rby]])
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
M = cv2.getPerspectiveTransform(pts1, pts2)
width = int(width)
height = int(height)
dst = cv2.warpPerspective(img, M, (width, height))
plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()
print(dst)
resu = cv2.threshold(dst, 120, 255, cv2.THRESH_BINARY)[1]
plt.imshow(resu), plt.title('Result')
plt.show()
cv2.imwrite('OCR.jpg', resu)
# ocr
# 创建reader对象

reader = easyocr.Reader(['ch_sim', 'en'])

# 读取图像

result = reader.readtext('OCR.jpg')
print(result)
for i in result:
    print(i[1])
