import cv2
from OpenCV_learning.generalFunctions import show
from matplotlib import pyplot as plt

k = 0.78125


def display(bedrew, result):
    b, g, r = cv2.split(bedrew)  # 颜色通道提取
    match_result_rgb = cv2.merge((r, g, b))  # 颜色通道合并
    plt.subplot(121), plt.imshow(result)
    plt.subplot(122), plt.imshow(match_result_rgb)
    plt.suptitle("cv2.TM_SQDIFF_NORMED")
    plt.show()


imgbgr = cv2.imread("42.jpg")
img = cv2.imread("42.jpg", 0)
template = cv2.imread("template5cm.png", 0)
roi = img[0:270, :]
res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
(X, Y) = min_loc
(y, x) = template.shape

match_result = cv2.rectangle(imgbgr, (X, Y), (X + x, Y + y), (0, 0, 255), 8)
display(match_result, res)
dot_center_x = X + x / 2
dot_center_y = Y + x / 2
move_x = (dot_center_x - 320) * k
move_y = (dot_center_y - 86) * k
print(move_x, move_y)
