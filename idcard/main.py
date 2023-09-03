import cv2
import matplotlib.pyplot as plt
import numpy as np


def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pltimg(image, way='gray'):
    if way == 'rgb':
        b, g, r = cv2.split(image)
        image = cv2.merge((r, g, b))
        plt.imshow(image), plt.title('image'), plt.show()
    else:
        plt.imshow(image, 'gray'), plt.title('image'), plt.show()


# 读入模版
template_gray = cv2.imread('template.png', 0)
template_bgr = cv2.imread('template.png')
# pltimg(template_gray)
# 二值化（反）
template_bin = cv2.threshold(template_gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
# pltimg(template_bin)
# 外轮廓
contours, hier = cv2.findContours(template_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
template_bgr_copy = template_bgr.copy()
draw = cv2.drawContours(template_bgr_copy, contours, -1, (0, 0, 255), 2)
pltimg(draw, 'rgb')
# 外接矩形
list_cont_xywh = []
template_bgr_copy = template_bgr.copy()
for i in range(len(contours)):
    cnt = contours[i]
    x, y, w, h = cv2.boundingRect(cnt)
    draw = cv2.rectangle(template_bgr_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
    list_cont_xywh.append([x, y, w, h])
pltimg(draw, 'rgb')
# 给list排序
while True:
    flag = 1
    for i in range(len(list_cont_xywh) - 1):
        if list_cont_xywh[i][0] > list_cont_xywh[i + 1][0]:
            flag = 0
            temp = list_cont_xywh[i]
            list_cont_xywh[i] = list_cont_xywh[i + 1]
            list_cont_xywh[i + 1] = temp
    if flag == 1:
        break
template_sep_list = []
for i in range(len(list_cont_xywh)):
    x1, x2, y1, y2 = list_cont_xywh[i][0], list_cont_xywh[i][0] + list_cont_xywh[i][2], list_cont_xywh[i][1], \
                     list_cont_xywh[i][1] + list_cont_xywh[i][3]
    template_sep_list.append(template_bin[y1:y2, x1:x2])
    # pltimg(template_sep_list[i])
for i in range(len(template_sep_list)):
    template_sep_list[i] = cv2.resize(template_sep_list[i], (55, 86))
    # pltimg(template_sep_list[i])
# 读入银行卡
idcard_bgr = cv2.imread('3.png')
idcard_bgr = cv2.resize(idcard_bgr, (583, 368))
idcard_gray = cv2.imread('3.png', 0)
idcard_gray = cv2.resize(idcard_gray, (583, 368))
idcard_bin = cv2.threshold(idcard_gray, 160, 255, cv2.THRESH_BINARY)[1]
# pltimg(idcard_bin)

kernel0 = np.ones((5, 5), np.uint8)
idcard_bin_tophat = cv2.morphologyEx(idcard_bin, cv2.MORPH_TOPHAT, kernel0)
# pltimg(idcard_bin_tophat)

kernel = np.ones((3, 20), np.uint8)
idcard_close = cv2.morphologyEx(idcard_bin_tophat, cv2.MORPH_CLOSE, kernel)
pltimg(idcard_close)
cnt_idcard, hier = cv2.findContours(idcard_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
idcard_copy = idcard_bgr.copy()
draw = cv2.drawContours(idcard_bgr, cnt_idcard, -1, (0, 0, 255), 2)
pltimg(draw, 'rgb')
idcard_list_cont_xywh = []
for i in range(len(cnt_idcard)):
    cnt = cnt_idcard[i]
    x, y, w, h = cv2.boundingRect(cnt)
    if 2.5 < w / h < 3.7 and w > 45 and h > 20:
        draw = cv2.rectangle(idcard_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
        idcard_list_cont_xywh.append([x, y, w, h])
while True:
    flag = 1
    for i in range(len(idcard_list_cont_xywh) - 1):
        if idcard_list_cont_xywh[i][0] > idcard_list_cont_xywh[i + 1][0]:
            flag = 0
            temp = idcard_list_cont_xywh[i]
            idcard_list_cont_xywh[i] = idcard_list_cont_xywh[i + 1]
            idcard_list_cont_xywh[i + 1] = temp
    if flag == 1:
        break
idcard_numberunit = []
for i in range(len(idcard_list_cont_xywh)):
    x1, x2, y1, y2 = idcard_list_cont_xywh[i][0], idcard_list_cont_xywh[i][0] + idcard_list_cont_xywh[i][2], \
                     idcard_list_cont_xywh[i][1], idcard_list_cont_xywh[i][1] + idcard_list_cont_xywh[i][3]
    idcard_numberunit.append(idcard_gray[y1 - 3:y2 + 3, x1 - 3:x2 + 3])
    pltimg(idcard_numberunit[i])

idcard_numbers = []
for i in range(len(idcard_numberunit)):

    contours_xywh = []
    unit_bin = cv2.threshold(idcard_numberunit[i], 150, 255, cv2.THRESH_BINARY)[1]
    pltimg(unit_bin)
    contours, hier = cv2.findContours(unit_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for j in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[j])
        contours_xywh.append([x, y, w, h])
    while True:
        flag = 0
        for j in range(len(contours_xywh) - 1):
            if contours_xywh[j][0] > contours_xywh[j + 1][0]:
                flag = 1
                temp = contours_xywh[j + 1]
                contours_xywh[j + 1] = contours_xywh[j]
                contours_xywh[j] = temp
        if flag == 0:
            break

    for j in range(len(contours_xywh)):
        if contours_xywh[j][2] > 10 and contours_xywh[j][3] > 20:
            x1, x2, y1, y2 = contours_xywh[j][0], contours_xywh[j][0] + contours_xywh[j][2], contours_xywh[j][1], \
                             contours_xywh[j][1] + contours_xywh[j][3]
            idcard_numbers.append(unit_bin[y1:y2, x1:x2])

# 比对
ans = []

for i in range(len(idcard_numbers)):
    idcard_numbers[i] = cv2.resize(idcard_numbers[i], (55, 86))
    score = []
    for j in range(len(template_sep_list)):
        res = cv2.matchTemplate(idcard_numbers[i], template_sep_list[j], cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        score.append(min_val)
    min_score = min(score)
    for k in range(len(score)):
        if min_score == score[k]:
            flag = k
            break
    ans.append(flag)
print(ans)
