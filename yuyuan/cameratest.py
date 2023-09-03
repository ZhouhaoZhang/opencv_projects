import cv2
import matplotlib.pyplot as plt
from OpenCV_learning.generalFunctions import show

"""
1像素=k毫米

"""
k = 0.775
img = cv2.imread("images/2.jpg")
print(img.shape)
img_gray = cv2.imread("images/2.jpg", 0)
img = img[0:190, 110:320]
img_gray = img_gray[0:190, 110:320]
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

h, s, v = b, g, r = cv2.split(img_hsv)
key = s
key_blur = cv2.GaussianBlur(key, (3, 3), 1)


ret1, key_bin = cv2.threshold(key_blur, 60, 255, cv2.THRESH_BINARY)
key_canny = cv2.Canny(key_bin, 100, 175)

plt.subplot(321), plt.imshow(img)
plt.subplot(322), plt.imshow(key)
plt.subplot(323), plt.imshow(key_blur)
plt.subplot(324), plt.imshow(key_bin)
plt.subplot(325), plt.imshow(key_canny)
plt.subplot(326), plt.imshow(key)
plt.show()
