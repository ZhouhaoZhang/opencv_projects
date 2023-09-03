import numpy as np
import cv2

# 经典的测试视频
cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# 形态学操作需要使用
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
# 创建混合高斯模型用于背景建模
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()  # 一帧一帧读取视频
    fgmask = fgbg.apply(frame)  # 将获取到的每一帧图像都应用到当前的背景提取当中，前景置为255，背景置为0
    # 形态学中开运算去掉噪音点
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # 寻找视频中的轮廓
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # 计算各轮廓的周长
        perimeter = cv2.arcLength(c, True)  # 计算各轮廓的周长
        if perimeter > 188:
            # 找到一个直矩阵(不会旋转)
            x, y, w, h = cv2.boundingRect(c)
            # 画出这个矩阵
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)
    k = cv2.waitKey(15) & 0xff
    if k == 27:  # 表示Esc退出
        break

cap.release()
cv2.destroyAllWindows()
