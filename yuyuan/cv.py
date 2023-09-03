import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


def show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def display(bedrew, result):
    b, g, r = cv2.split(bedrew)  # 颜色通道提取
    match_result_rgb = cv2.merge((r, g, b))  # 颜色通道合并
    plt.subplot(121), plt.imshow(result)
    plt.subplot(122), plt.imshow(match_result_rgb)
    plt.suptitle("cv2.TM_SQDIFF_NORMED")
    plt.show()


def detect_line_a():
    """
    检测边界线，返回角度a逆时针为正
    """
    k = 0.775
    img = cv2.imread("64.jpg", 0)
    roi = img[0:220, :]
    top_size, bottom_size, right_size, left_size = (0, 260, 100, 100)
    roi = cv2.copyMakeBorder(roi, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=255)
    # show("roi", roi)
    ret1, bin = cv2.threshold(roi, 85, 255, cv2.THRESH_BINARY_INV)
    # show("bin", bin)
    kernel = np.ones((15, 15), np.uint8)
    bin_open = cv2.morphologyEx(bin, cv2.MORPH_OPEN, kernel)
    # show("open", bin_open)
    cnts, ret2 = cv2.findContours(bin_open, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cntsorted = sorted(cnts, key=lambda cnts: cv2.contourArea(cnts), reverse=True)
    cnt = cntsorted[0]
    # epsilon = 0.03 * cv2.arcLength(cnt, True)  # epsilon占周长的比例
    # approx = cv2.approxPolyDP(cnt, epsilon, True)
    # x, y, w, h = cv2.boundingRect(cnt)
    [VX, VY, X, Y] = cv2.fitLine(cnt, cv2.DIST_HUBER, 0, 0.01, 0.01)
    print(VX, VY, X, Y)
    a = -(math.atan(VY / VX) / 6.28 * 360)
    return a


def detect_line_y():
    """
    检测边界线，偏画面上为正，返回y
    """
    k = 0.775
    img = cv2.imread("64.jpg", 0)
    roi = img[0:220, :]
    top_size, bottom_size, right_size, left_size = (0, 260, 100, 100)
    roi = cv2.copyMakeBorder(roi, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=255)
    # show("roi", roi)
    ret1, bin = cv2.threshold(roi, 85, 255, cv2.THRESH_BINARY_INV)
    # show("bin", bin)
    kernel = np.ones((15, 15), np.uint8)
    bin_open = cv2.morphologyEx(bin, cv2.MORPH_OPEN, kernel)
    # show("open", bin_open)
    cnts, ret2 = cv2.findContours(bin_open, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cntsorted = sorted(cnts, key=lambda cnts: cv2.contourArea(cnts), reverse=True)
    cnt = cntsorted[0]
    epsilon = 0.03 * cv2.arcLength(cnt, True)  # epsilon占周长的比例
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    x, y, w, h = cv2.boundingRect(cnt)
    [VX, VY, X, Y] = cv2.fitLine(cnt, cv2.DIST_HUBER, 0, 0.01, 0.01)
    print(VX, VY, X, Y)
    distance = -(k * ((y + h / 2) - 100))
    return distance

def detect_block_xy():
    """
    检测块，返回x，y，偏右偏上为正
    """
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
    # show("s", s)
    key = s
    """
    高斯滤波
    """
    key_blur = cv2.GaussianBlur(key, (3, 3), 1)
    # show("s", s)
    """
    二值化
    """
    ret1, key_bin = cv2.threshold(key_blur, 127, 255, cv2.THRESH_BINARY)
    # show("bin", key_bin)
    """
    开运算
    """
    kernel = np.ones((30, 30), np.uint8)
    key_bin_open = cv2.morphologyEx(key_bin, cv2.MORPH_OPEN, kernel)
    # show("bin_Open", key_bin_open)
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
    # bin_roi = key_blur[y:y + h, x:x + w]
    # show("roi", bin_roi)
    """
    角点检测
    """
    # corners = cv2.goodFeaturesToTrack(bin_roi, 7, 0.03, 5, useHarrisDetector=False)
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
    Y = -((y_fixed - 240) * k)

    # show("res", img)
    return [X, Y]


def detect_dot_xy():
    """
    检测圆点，返回x，y
    """
    k = 0.78125
    imgbgr = cv2.imread("42.jpg")
    img = cv2.imread("42.jpg", 0)
    template = cv2.imread("template5cm.png", 0)
    res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    (X, Y) = min_loc
    (y, x) = template.shape

    match_result = cv2.rectangle(imgbgr, (X, Y), (X + x, Y + y), (0, 0, 255), 8)
    # display(match_result, res)
    dot_center_x = X + x / 2
    dot_center_y = Y + x / 2
    move_x = (dot_center_x - 320) * k
    move_y = -((dot_center_y - 100) * k)
    print(move_x, move_y)
    return [move_x, move_y]


def detect_dot_a():
    """
    检测圆点，返回角度
    """
    imgbgr = cv2.imread("47.jpg")
    img = cv2.imread("47.jpg", 0)
    template = cv2.imread("template5cm.png", 0)
    roi = img[0:270, :]
    res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    (X, Y) = min_loc
    (y, x) = template.shape
    match_result = cv2.rectangle(imgbgr, (X, Y), (X + x, Y + y), (0, 0, 255), 8)
    # display(match_result, res)
    roi_line = img[0:172, X + x:X + x + 234]
    # show("roi", roi_line)
    ret1, bin = cv2.threshold(roi_line, 85, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((15, 15), np.uint8)
    bin_open = cv2.morphologyEx(bin, cv2.MORPH_OPEN, kernel)
    # show("open", bin_open)
    cnts, ret2 = cv2.findContours(bin_open, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cntsorted = sorted(cnts, key=lambda cnts: cv2.contourArea(cnts), reverse=True)
    cnt = cntsorted[0]
    epsilon = 0.03 * cv2.arcLength(cnt, True)  # epsilon占周长的比例
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    x, y, w, h = cv2.boundingRect(cnt)
    [VX, VY, Xk, Yk] = cv2.fitLine(cnt, cv2.DIST_HUBER, 0, 0.01, 0.01)
    print(VX, VY, Xk, Yk)
    a = -(math.atan(VY / VX) / 6.28 * 360)
    return a


if __name__ == "__main__":
    print('test')
