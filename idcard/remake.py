import cv2
import matplotlib.pyplot as plt


def read(img, thresh=127, inv=False):
    origin = cv2.imread(img)
    gray = cv2.imread(img, 0)
    binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY)[1]
    return origin, gray, binary


def unify_size(img_list):
    outputlist = []
    xsize, ysize = img_list[0].shape
    for img in img_list:
        outputlist.append(cv2.resize(img, (ysize, xsize)))
    return outputlist


template_bgr, template_gray, template_bin = read('template.png', inv=True)
contours, hier = cv2.findContours(template_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rectangles = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    rectangles.append([x, y, w, h])
rectangles.sort(key=lambda rect: rect[0])
num_template = []
for [x, y, w, h] in rectangles:
    num_template.append(template_bin[y:y + h, x:x + w])
num_template = unify_size(num_template)
# 读入
idcard_bgr, idcard_gray, idcard_bin = read('1.png', 127)
# 构建三个将来会用到的卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
rectKernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 7))
basicKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# 改变大小
idcard_bgr, idcard_gray, idcard_bin = cv2.resize(idcard_bgr, (583, 368)), cv2.resize(idcard_gray,
                                                                                     (583, 368)), cv2.resize(idcard_bin,
                                                                                                             (583, 368))
# 顶帽操作
idcard_gray_tophat = cv2.morphologyEx(idcard_gray, cv2.MORPH_TOPHAT, rectKernel)
plt.imshow(idcard_gray_tophat)
plt.show()
# 梯度计算
gradX = cv2.convertScaleAbs(cv2.Sobel(idcard_gray_tophat, cv2.CV_64F, 1, 0, ksize=3))
gradY = cv2.convertScaleAbs(cv2.Sobel(idcard_gray_tophat, cv2.CV_64F, 0, 1, ksize=3))
grad = cv2.addWeighted(gradX, 0.5, gradY, 0.5, 0)
plt.imshow(grad)
plt.show()
# 闭操作
grad_close = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
plt.imshow(grad_close)
plt.show()
# 自适应阈值二值化
close_autobin = cv2.threshold(grad_close, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
plt.imshow(close_autobin)
plt.show()
# 闭操作
close2 = cv2.morphologyEx(close_autobin, cv2.MORPH_CLOSE, rectKernel2)
plt.imshow(close2)
plt.show()

# 找边界
contours, hier = cv2.findContours(close2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
boundingBoxes = []
# 按照宽高比和像素范围筛选数字组
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    ar = w / float(h)
    if 2.6 < ar < 3.8 and 70 < w < 115 and 15 < h < 45:
        boundingBoxes.append([x, y, w, h])
boundingBoxes.sort(key=lambda box: box[0])
group = []
# ROI
for [x, y, w, h] in boundingBoxes:
    group.append(idcard_gray[y - 5:y + h + 5, x - 5:x + w + 5])
    #plt.imshow(idcard_gray[y - 5:y + h + 5, x - 5:x + w + 5])
    #plt.show()
group_binary = []
# 对数字组进行形态学操作
for nominee in group:
    nominee_bin = cv2.threshold(nominee, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    nominee_close = cv2.morphologyEx(nominee_bin, cv2.MORPH_CLOSE, basicKernel)
    group_binary.append(nominee_close)

    #plt.imshow(cv2.threshold(nominee, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
    #plt.show()
num_nominee = []
# 对数字组找边界
for nominee in group_binary:
    contours, hier = cv2.findContours(nominee, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rectangles = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rectangles.append([x, y, w, h])
    rectangles.sort(key=lambda rect: rect[0])
    # ROI切割出单个数字
    for [x, y, w, h] in rectangles:
        if w > 8 and h > 20:
            num_nominee.append(nominee[y:y + h, x:x + w])
            plt.imshow(nominee[y:y + h, x:x + w])
            plt.show()
num_nominee = unify_size(num_nominee)
num_standard = []
(x, y) = num_nominee[0].shape
for i in range(len(num_template)):
    num_standard.append(cv2.resize(num_template[i], (y, x)))
ans = []
for nominee in num_nominee:
    score = []
    for standard in num_standard:
        res = cv2.matchTemplate(nominee, standard, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        score.append(min_val)
    ans.append(score.index(min(score)))
print(ans)
