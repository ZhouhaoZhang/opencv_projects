import cv2
import numpy as np
import random


# 设置卡尔曼滤波器
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 100

# 初始化目标位置
last_measurement = current_measurement = np.array((2, 1), np.float32)
last_prediction = current_prediction = np.zeros((2, 1), np.float32)
X = np.linspace(100, 700, 60)
Y = np.linspace(100, 700, 60)
frame = np.zeros((800, 800, 3), np.uint8)

# 循环处理每一帧
for i in range(len(X)):
    # 假设您已经检测到目标并获取其位置为bbox
    #x = X[i]+random.gauss(0, 3)
    #y = Y[i]+random.gauss(0, 3)
    x = X[i]
    y = Y[i]
    current_measurement = np.array([np.float32(x), np.float32(y)])

    # 更新卡尔曼滤波器状态
    kalman.correct(current_measurement)
    current_prediction = kalman.predict()

    # 在图像上绘制目标位置预测 红色为真值，绿色为预测
    cv2.circle(frame, (int(current_prediction[0]), int(current_prediction[1])), 2, (75, 255, 75), -1)
    cv2.circle(frame, (int(x), int(y)), 2, (75, 75, 255), -1)

    # 更新上一次的测量值和预测值
    last_prediction = current_prediction.copy()
    last_measurement = current_measurement.copy()

    # 显示图像
    cv2.imshow('frame', frame)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break
