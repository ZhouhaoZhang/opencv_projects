# 开发时间: 2023/2/11 23:15
import cv2
import numpy as np


def myApprox(con,side_size):# con为预先得到的最大轮廓
    num = 0.001
    ep = num * cv2.arcLength(con, True)
    con = cv2.approxPolyDP(con, ep, True)
    while ( True ):
        if len(con) <= side_size:
            break
        else:
            num = num * 1.5
            ep = num * cv2.arcLength(con, True)
            con = cv2.approxPolyDP(con, ep, True)
            continue
    return con


# 相机参数
list1 = [1671.87, 0, 956.24, 0, 1671.71, 532.65, 0, 0, 1]
mtx = np.mat(list1).reshape(3, 3)
dist = np.mat([0, 0, 0, 0, 0])

list11 = [1671.87, 0, 956.24, 0, 0, 1671.71, 532.65, 0, 0, 0, 1, 0]
M1 = np.mat(list11).reshape(3, 4)

# 世界坐标
list2 = [[-268.7, 0.0, 0.0], [0.0, -268.7, 0.0], [268.7, 0.0, 0.0], [0.0, 268.7, 0.0]]
# objp = np.zeros((4, 3), np.float32)
objp = np.float32(list2)


ball_color = 'red'

color_dist = {'red': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
              'blue': {'Lower': np.array([100, 80, 46]), 'Upper': np.array([124, 255, 255])},
              'green': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
              }

cap = cv2.VideoCapture('./test.mp4')

flag = False
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        if frame is not None:
            gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)                     # 高斯模糊
            hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)                       # HSV图像
            erode_hsv = cv2.erode(hsv, None, iterations=8)
            cv2.namedWindow('result', cv2.WINDOW_KEEPRATIO)

            inRange_hsv = cv2.inRange(erode_hsv, color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])
            # 绘制
            cv2.imshow('result', inRange_hsv)
            cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

            c = max(cnts, key=cv2.contourArea)
            con = myApprox(c, side_size=4)

            cv2.drawContours(frame, con, -1, (0, 255, 255), 10)

            # 打印角点
            if flag:
                for j in range(0, 4, 1):
                    for i in range(0, 4, 1):
                        if(abs(con1[j][0][0]-con[i][0][0])<20 and abs(con1[j][0][1]-con[i][0][1])<20):
                            tmp = con[i][0][0]
                            con[i][0][0] = con[j][0][0]
                            con[j][0][0] = tmp
                            tmp = con[i][0][1]
                            con[i][0][1] = con[j][0][1]
                            con[j][0][1] = tmp
                            break
                cv2.putText(frame, "A", (con[0][0][0] - 20, con[0][0][1] - 20), cv2.FONT_HERSHEY_COMPLEX, 2.0,
                            (100, 200, 200), 5)
                cv2.putText(frame, "B", (con[1][0][0] - 50, con[1][0][1] + 10), cv2.FONT_HERSHEY_COMPLEX, 2.0,
                            (100, 200, 200), 5)
                cv2.putText(frame, "C", (con[2][0][0] - 20, con[2][0][1] + 60), cv2.FONT_HERSHEY_COMPLEX, 2.0,
                            (100, 200, 200), 5)
                cv2.putText(frame, "D", (con[3][0][0] + 10, con[3][0][1] + 10), cv2.FONT_HERSHEY_COMPLEX, 2.0,
                            (100, 200, 200), 5)
                # pnp
                list3 = [[con[0][0][0], con[0][0][1]], [con[1][0][0], con[1][0][1]], [con[2][0][0], con[2][0][1]],
                         [con[3][0][0], con[3][0][1]]]
                corners = np.zeros((4, 2), dtype=int)
                corners = np.float32(list3)
                _, R, T = cv2.solvePnP(objp, corners, mtx, dist)

                # 打印矢量
                text = str(R)
                text = text.replace(",", " ")
                text = text.replace('\n', '')
                cv2.putText(frame, "rotation vector", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (10, 10, 10), 3)
                cv2.putText(frame, text, (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1.0, (10, 10, 10), 3)
                text = str(T)
                text = text.replace(",", " ")
                text = text.replace('\n', '')
                cv2.putText(frame, "translation vector", (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1.0, (10, 10, 10), 3)
                cv2.putText(frame, text, (10, 160), cv2.FONT_HERSHEY_COMPLEX, 1.0, (10, 10, 10), 3)

                # 画坐标轴
                oj = np.float32([[0.0, 0.0, 0.0], [400.0, 0.0, 0.0], [0.0, 400.0, 0.0], [0.0, 0.0, 400.0]])

                # 旋转向量转化为旋转矩阵
                in_r = cv2.Rodrigues(R, jacobian=0)[0]

                # 输入平移向量
                in_weiyi = np.mat([T[0][0], T[1][0], T[2][0]])

                # 获得外参数矩阵
                # 列合并
                M2 = np.hstack((in_r, in_weiyi.T))
                # print(M2)
                yi = np.mat([0, 0, 0, 1])
                M2 = np.vstack((M2, yi))

                ########################相机矩阵###################################
                M = M1 * M2
                # 坐标转换
                corner_out = np.zeros((4, 2), np.float32)
                k = 0
                sigma = 0.12
                for l in oj:
                    l = np.append(l, 1)
                    l = np.mat(l)

                    out = (M1 * M2 * l.T) / ((M2 * l.T)[2, 0])
                    corner_out[k][0] = float(out[0, 0])
                    corner_out[k][1] = float(out[1, 0])
                    k += 1
                cv2.line(frame, (int(corner_out[0][0]), int(corner_out[0][1])),
                         (int(corner_out[1][0]), int(corner_out[1][1])), (200, 100, 50), 20)
                cv2.putText(frame, "x", (int(corner_out[1][0]) + 10, int(corner_out[1][1]) + 10),
                            cv2.FONT_HERSHEY_COMPLEX, 2.0,
                            (200, 100, 50), 10)

                cv2.line(frame, (int(corner_out[0][0]), int(corner_out[0][1])),
                         (int(corner_out[2][0]), int(corner_out[2][1])), (50, 40, 200), 20)
                cv2.putText(frame, "y", (int(corner_out[2][0]) + 10, int(corner_out[2][1]) + 10),
                            cv2.FONT_HERSHEY_COMPLEX, 2.0,
                            (50, 40, 200), 10)

                cv2.line(frame, (int(corner_out[0][0]), int(corner_out[0][1])),
                         (int(corner_out[3][0]), int(corner_out[3][1])), (100, 200, 50), 20)
                cv2.putText(frame, "z", (int(corner_out[3][0]) + 10, int(corner_out[3][1]) + 10),
                            cv2.FONT_HERSHEY_COMPLEX, 2.0,
                            (100, 200, 50), 10)


            cv2.waitKey(1)
            con1 = con
        else:
            print("无画面")

        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        cap.release()
        cv2.destroyAllWindows()
    if not flag:
        flag = True
