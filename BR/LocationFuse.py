from math import asin, cos, sin, atan, sqrt, tan, atan2, acos
from matplotlib import pyplot as plt
import random
import numpy as np
import cv2
from pytictoc import TicToc

pole0 = (2750, 2750, 200, 1000)
pole1 = (9250, 2750, 200, 1000)
pole2 = (2750, 6000, 200, 1000)
pole3 = (9250, 6000, 200, 1000)
pole4 = (2750, 9250, 200, 1000)
pole5 = (9250, 9250, 200, 1000)
pole6 = (4650, 4650, 400, 1200)
pole7 = (7350, 4650, 400, 1200)
pole8 = (4650, 7350, 400, 1200)
pole9 = (7350, 7350, 400, 1200)
pole10 = (6000, 6000, 400, 1900)
pole_normal = [pole0, pole1, pole2, pole3, pole4, pole5, pole6, pole7, pole8, pole9]
pole = [pole0, pole1, pole2, pole3, pole4, pole5, pole6, pole7, pole8, pole9, pole10]
pi = 3.1415926
eps = 1e-8


class LocationFuseNode:
    def __init__(self):

        """
        ros初始化，省略
        """
        """
        相机内外参
        """
        self.camera_height = 900
        self.camera_resolution = (640, 480)
        self.camera_fx = 430.86734615368283
        self.camera_fy = 430.87603115465436
        self.camera_cx = 325.215666294987
        self.camera_cy = 246.02912176677842

        # 水平
        self.camera_hv_r = atan((self.camera_resolution[0] - self.camera_cx) / self.camera_fx)
        self.camera_hv_l = -atan(self.camera_cx / self.camera_fx)
        # 竖直
        self.camera_vv_u = -atan(self.camera_cy / self.camera_fy)
        self.camera_vv_d = atan((self.camera_resolution[1] - self.camera_cy) / self.camera_fy)
        # 水平
        self.camera_fov_h = self.camera_hv_r / pi * 180 - self.camera_hv_l / pi * 180
        # 竖直
        self.camera_fov_v = self.camera_vv_d / pi * 180 - self.camera_vv_u / pi * 180

        self.image = np.zeros((self.camera_resolution[1], self.camera_resolution[0], 3), dtype='uint8')
        self.map = np.zeros((1200, 1200, 3), dtype='uint8')
        for p in pole_normal:
            cv2.circle(self.map, (int(p[0] / 10), int((12000 - p[1]) / 10)), 5, (0, 255, 255), 2)
        p = pole10
        cv2.circle(self.map, (int(p[0] / 10), int((12000 - p[1]) / 10)), 8, (0, 255, 255), 2)
        self.video = []

        self.fps = 17
        # 多少毫秒采样一次

        self.sample_rate = round(1000 / self.fps)  # 毫秒
        """
        定义一个李萨如运动（真值）
        """
        self.time = [i for i in range(6280 * 2)]
        self.sample_time = [i for i in range(6280 * 2) if i % self.sample_rate == 0]

        x_loc_real = [975 + 475 * sin(i / 1000) for i in range(6280 * 2)]
        # x_loc_real = [4000 + 3000 * sin(i / 1000) for i in range(6280 * 2)]
        y_loc_real = [6000 - 5000 * cos(i / 400) for i in range(6280 * 2)]

        theta_real = [30 * cos(i / 400) for i in range(6280 * 2)]
        self.pose_real = np.array([x_loc_real, y_loc_real, theta_real])

        """
        产生定位先验数据
        先认为视觉与定位严格耦合
        """
        self.pose_priori = None
        self.generate_pose_priori()

        """
        产生真实的柱子相对位姿(基于采样)
        """
        self.relative_pose_real = []
        self.generate_relative_pose_real()

        """
        产生vision话题
        对relative_pose_real 采样
        """
        self.vision_topic = []
        self.generate_vision_topic()
        """
        校正
        """
        self.corrected_pose = []
        self.correct_new()

    @staticmethod
    def get_cover_state(x11, y11, x12, y12, x21, y21, x22, y22):
        """
        先传的被判断的柱子，后传的前面柱子，返回一个四元数，分别代表左上，右上，左下，右下角点的遮挡情况，被遮挡为1
        """
        c1, c2, c3, c4 = False, False, False, False
        num = 0
        if x21 <= x11 <= x22 and y21 <= y11 <= y22:
            c1 = True
            num = num + 1
        if x21 <= x12 <= x22 and y21 <= y11 <= y22:
            c2 = True
            num = num + 1
        if x21 <= x11 <= x22 and y21 <= y12 <= y22:
            c3 = True
            num = num + 1
        if x21 <= x12 <= x22 and y21 <= y12 <= y22:
            c4 = True
            num = num + 1
        return c1, c2, c3, c4, num

    def correct(self):
        T = TicToc()
        corrected_pose_list = []  # 校正后的位姿列表
        pose_real_sample = self.pose_real[:, 0::self.sample_rate]  # 真正的位姿采样
        num_pole_frame_avg = 0
        num_pole_matched_avg = 0
        T.tic()
        for t in range(len(self.sample_time)):
            real_pose = (pose_real_sample[0][t], pose_real_sample[1][t], pose_real_sample[2][t])  # t时刻的位姿真值
            pose = self.pose_priori[:, t]  # t时刻的位姿先验
            vision_topic = self.vision_topic[t]  # t时刻的视觉话题 xywh
            num_pole_frame_avg = num_pole_frame_avg + len(vision_topic)
            # 对vision_topic进行预处理
            topic_useful = []  # 筛选出的优质的柱子角度
            for p in vision_topic:
                useful = True
                pl = p[0] - p[2] / 2
                pr = p[0] + p[2] / 2
                if pl <= 3 and pr >= self.camera_resolution[1] - 3:  # 贴左右边界 舍弃
                    continue
                for i in vision_topic:  # 检查混叠
                    if i is p:
                        continue
                    il = i[0] - i[2] / 2
                    ir = i[0] + i[2] / 2
                    if il <= pl <= ir or il <= pr <= ir or pl <= il < ir <= pr:  # 重叠
                        useful = False
                        break
                    elif 0 < pl - ir <= 5 or 0 < il - pr <= 5:  # 相距过近
                        useful = False
                        break
                if useful:
                    pix_xl = pl - self.camera_cx
                    pix_xr = pr - self.camera_cx
                    """
                    角度
                    """
                    # 左边沿偏移角 atan(pix_xl*dx /f)=atan(pix_xl/fx)
                    theta_l = atan(pix_xl / self.camera_fx)
                    # 右边沿偏移角
                    theta_r = atan(pix_xr / self.camera_fx)
                    # 角平分线
                    theta = (theta_l + theta_r) / 2
                    theta = theta / pi * 180
                    topic_useful.append(theta)
            # print(topic_useful)
            # 配对柱子
            x, y, theta = pose[0], pose[1], pose[2]  # 先验
            # print(x, y, theta)
            ref = []
            relative_pole_theta_priori = []  # 先验下的11根柱子角度
            for i in range(11):
                theta_pole = 180 * atan((pole[i][1] - y) / (pole[i][0] - x)) / pi
                if pole[i][0] - x <= 0:
                    if pole[i][1] - y <= 0:
                        theta_pole = -180 + theta_pole
                    else:
                        theta_pole = 180 + theta_pole

                theta_relative = theta - theta_pole
                if theta_relative > 180:
                    theta_relative = theta_relative - 360
                elif theta_relative <= -180:
                    theta_relative = 360 + theta_relative
                relative_pole_theta_priori.append(theta_relative)

            for p in topic_useful:
                error = []
                for theta_relative in relative_pole_theta_priori:
                    error.append(abs(theta_relative - p))
                if min(error) <= 2:
                    num = 0
                    for e in error:
                        if e <= 2:
                            num = num + 1
                    if num == 1:
                        index = error.index(min(error))
                        ref.append([index, p])  # 符合要求的柱子  编号，角度
            # print(ref)
            # 校正

            if len(ref) <= 2:
                corrected_pose_list.append(pose)
                """
                img = self.map.copy()
                cv2.circle(img, (int(pose[0] / 10), int((12000 - pose[1]) / 10)), 3, (0, 0, 255), 2)
                startx = int(pose[0] / 10)
                starty = int(1200 - pose[1] / 10)
                endx = int(pose[0] / 10 + 50 * cos(pose[2] / 180 * pi))
                endy = int(1200 - pose[1] / 10 - 50 * sin(pose[2] / 180 * pi))

                cv2.line(img, (startx, starty), (endx, endy), (0, 0, 255), 2)
                cv2.circle(img, (int(real_pose[0] / 10), int((12000 - real_pose[1]) / 10)), 3, (0, 255, 0), 3)

                cv2.imshow("res", img)
                cv2.imshow("frame", self.video[t])
                cv2.waitKey(0)
                """
            else:
                # 加校正
                num_pole_matched_avg = num_pole_matched_avg + len(ref)
                ref = sorted(ref, key=lambda d: d[1])
                p1 = pole[ref[0][0]]
                p2 = pole[ref[-1][0]]
                theta_diff = ref[-1][1] - ref[0][1]
                if ref[-1][1] - ref[1][1] > ref[-2][1] - ref[0][1]:
                    p3 = pole[ref[1][0]]
                    p4 = pole[ref[-1][0]]
                    theta_diff_2 = ref[-1][1] - ref[1][1]
                else:
                    p3 = pole[ref[-2][0]]
                    p4 = pole[ref[0][0]]
                    theta_diff_2 = ref[-2][1] - ref[0][1]

                o1, r1 = self.find_circle(p1, p2, theta_diff, [pose[0], pose[1]])
                o2, r2 = self.find_circle(p3, p4, theta_diff_2, [pose[0], pose[1]])
                intersection, rank = self.find_intersection([o1, r1, 0], [o2, r2, 0], x, y)
                if intersection is not None:
                    corrected_x, corrected_y = intersection
                    if abs(corrected_x - x) > 100 or abs(corrected_y - y) > 100:
                        corrected_pose_list.append(pose)
                    else:
                        corrected_pose = np.array([corrected_x, corrected_y, 0])
                        # 校正角度
                        p1 = ref[0]
                        p2 = ref[-1]
                        theta_p1 = 180 * atan((pole[p1[0]][1] - corrected_y) / (pole[p1[0]][0] - corrected_x)) / pi
                        if pole[p1[0]][0] - corrected_x <= 0:
                            if pole[p1[0]][1] - corrected_y <= 0:
                                theta_p1 = -180 + theta_p1
                            else:
                                theta_p1 = 180 + theta_p1

                        theta1 = theta_p1 + p1[1]
                        if theta1 > 180:
                            theta1 = theta1 - 360
                        elif theta1 <= -180:
                            theta1 = 360 + theta1

                        theta_p2 = 180 * atan((pole[p2[0]][1] - corrected_y) / (pole[p2[0]][0] - corrected_x)) / pi
                        if pole[p2[0]][0] - corrected_x <= 0:
                            if pole[p2[0]][1] - corrected_y <= 0:
                                theta_p2 = -180 + theta_p2
                            else:
                                theta_p2 = 180 + theta_p2

                        theta2 = theta_p2 + p2[1]
                        if theta2 > 180:
                            theta2 = theta2 - 360
                        elif theta2 <= -180:
                            theta2 = 360 + theta2
                        if theta1 * theta2 >= 0:
                            corrected_pose[2] = (theta1 + theta2) / 2
                        else:
                            if abs(theta1) < 90:
                                corrected_pose[2] = (theta1 + theta2) / 2
                            else:
                                corrected_pose[2] = (theta1 + theta2) / 2 + 180
                        corrected_pose_list.append(corrected_pose)
                        """
                o1 = (int(o1[0] / 10), int((12000 - o1[1]) / 10))
                o2 = (int(o2[0] / 10), int((12000 - o2[1]) / 10))
                img = self.map.copy()
                cv2.circle(img, o1, int(r1 / 10), (255, 255, 255), 2)
                cv2.circle(img, o2, int(r2 / 10), (255, 255, 255), 2)
                cv2.circle(img, (int(pose[0] / 10), int((12000 - pose[1]) / 10)), 5, (0, 0, 255), 2)
                cv2.circle(img, (int(real_pose[0] / 10), int((12000 - real_pose[1]) / 10)), 5, (0, 255, 0), 3)
                startx = int(pose[0] / 10)
                starty = int(1200 - pose[1] / 10)
                endx = int(pose[0] / 10 + 50 * cos(pose[2] / 180 * pi))
                endy = int(1200 - pose[1] / 10 - 50 * sin(pose[2] / 180 * pi))

                cv2.line(img, (startx, starty), (endx, endy), (0, 0, 255), 2)
                cv2.imshow("res", img)
                cv2.imshow("frame", self.video[t])
                cv2.waitKey(0)
                """
        pose_corrected = np.array(corrected_pose_list)
        pose_corrected = pose_corrected.T
        pose_real_sample = self.pose_real[:, 0::self.sample_rate]
        error_x_corrected = np.average(abs(pose_corrected[0] - pose_real_sample[0]))
        error_y_corrected = np.average(abs(pose_corrected[1] - pose_real_sample[1]))
        error_theta_corrected = np.average(abs(pose_corrected[2] - pose_real_sample[2]))

        num_pole_frame_avg = num_pole_frame_avg / len(self.sample_time)
        print("平均每帧yolo返回的柱子数量：", num_pole_frame_avg)
        statement_2 = '老校正 平均    x误差 : {0} 毫米, y误差 : {1} 毫米,角度误差 : {2} 度'
        statement_2 = statement_2.format(error_x_corrected, error_y_corrected, error_theta_corrected)
        print("老算法平均每帧成功匹配柱子数量：", num_pole_matched_avg / len(self.sample_time))
        T.toc()
        # self.plot()
        print(statement_2)

    def correct_new(self):
        corrected_pose_list = []  # 校正后的位姿列表
        num_pole_matched_avg = 0
        T = TicToc()
        T.tic()
        for t in range(len(self.sample_time)):
            pose_priori = self.pose_priori[:, t]  # t时刻的位姿先验
            x, y, theta = pose_priori[0], pose_priori[1], pose_priori[2]  # 先验

            relative_pole_theta_priori_selected = []  # 在先验视野范围内的
            for i in range(11):
                theta_pole = 180 * atan((pole[i][1] - y) / (pole[i][0] - x)) / pi
                if pole[i][0] - x <= 0:
                    if pole[i][1] - y <= 0:
                        theta_pole = -180 + theta_pole
                    else:
                        theta_pole = 180 + theta_pole
                theta_relative = theta - theta_pole
                if theta_relative > 180:
                    theta_relative = theta_relative - 360
                elif theta_relative <= -180:
                    theta_relative = 360 + theta_relative
                if abs(theta_relative) < self.camera_fov_h / 2 + 5:  # 5可以修改
                    relative_pole_theta_priori_selected.append([i, theta_relative])
            relative_pole_theta_priori_selected = sorted(relative_pole_theta_priori_selected, key=lambda d: d[1])

            vision_topic = self.vision_topic[t]  # t时刻的视觉话题 xywh

            """
            对vision_topic进行预处理
            """
            topic_processed = []
            # 每个元素为：[0柱子编号，1角度，2是否离散完好，3是否左右边界缺失，4是否重叠，5是否相距过近，6：往左数的第一个离散柱编号，7往右的，8评分]

            for p in vision_topic:
                topic_processed_element = [-1, 0, False, False, False, False, -1, -1, 0]  # 柱子编号-1代表匹配失败
                pl = p[0] - p[2] / 2
                pr = p[0] + p[2] / 2
                pix_xl = pl - self.camera_cx
                pix_xr = pr - self.camera_cx
                # 计算角度
                # 左边沿偏移角 atan(pix_xl*dx /f)=atan(pix_xl/fx)
                theta_l = atan(pix_xl / self.camera_fx)
                # 右边沿偏移角
                theta_r = atan(pix_xr / self.camera_fx)
                # 角平分线
                theta_p = (theta_l + theta_r) / 2
                theta_p = theta_p / pi * 180

                # 更新角度
                topic_processed_element[1] = theta_p

                # 检验左右边缺失
                if pl <= 3 and pr >= self.camera_resolution[1] - 3:
                    topic_processed_element[3] = True

                depart = True
                # 检查重叠和相距过近情况
                for i in vision_topic:
                    if i is p:
                        continue
                    il = i[0] - i[2] / 2
                    ir = i[0] + i[2] / 2
                    if il <= pl <= ir or il + 3 <= pr <= ir - 3 or pl <= il < ir <= pr:  # 重叠
                        topic_processed_element[4] = True
                    elif 0 < pl - ir <= 5 or 0 < il - pr <= 5:  # 相距过近
                        topic_processed_element[5] = True
                    elif 0 < pl - ir <= 30 or 0 < il - pr <= 30:
                        depart = False

                # 检查是否离散完好
                topic_processed_element[2] = (not topic_processed_element[3]) and (not topic_processed_element[4]) and (not topic_processed_element[5])
                topic_processed.append(topic_processed_element) and depart
            topic_processed = sorted(topic_processed, key=lambda d: d[1])

            # print(topic_processed)
            """
            匹配并打分
            """

            matched_outlier_num = 0
            # 匹配离散柱
            for p in topic_processed:
                error = []
                theta_p, is_outlier, is_broken, is_mixed, is_close, = p[1], p[2], p[3], p[4], p[5]
                if is_outlier:
                    for theta_relative in relative_pole_theta_priori_selected:
                        error.append(abs(theta_relative[1] - theta_p))
                    try:
                        if min(error) <= 5:
                            num = 0
                            for e in error:
                                if e <= 5:
                                    num = num + 1
                            if num == 1:
                                index = error.index(min(error))
                                p[0] = relative_pole_theta_priori_selected[index][0]
                                matched_outlier_num = matched_outlier_num + 1
                    except ValueError:
                        p[0] = -1
            if matched_outlier_num != 0:
                # 往左，往右的离散柱编号
                lp = -1
                rp = -1
                for p in topic_processed:
                    if p[2] and p[0] != -1:
                        lp = p[0]  # 更新左方离散柱
                    else:
                        p[6] = lp
                for p in reversed(topic_processed):
                    if p[2] and p[0] != -1:
                        rp = p[0]
                    else:
                        p[7] = rp
                # 根据左右的离散柱情况匹配非离散柱
                for p in topic_processed:
                    if p[0] == -1:
                        p[0] = self.match_confusing_pole(p, topic_processed, relative_pole_theta_priori_selected)

            # 检查重复匹配和失败的匹配
            for p in topic_processed:
                if p[8] == -1:
                    continue
                if p[0] == -1:
                    p[8] = -1
                    continue
                for i in topic_processed:
                    if i is p or i[8] == -1:
                        continue
                    else:
                        if i[0] == p[0]:
                            p[8] = -1
                            i[8] = -1
                            continue
            # 筛选得分不为-1的
            topic_final = []
            sum_r = 0
            for p in topic_processed:
                if p[8] != -1:
                    rank = 8 / 2 ** (p[3] + p[4] + p[5])
                    topic_final.append([p[0], p[1], rank])
                    sum_r = sum_r + rank
            num_pole_matched_avg = num_pole_matched_avg + len(topic_final)
            if len(topic_final) <= 3:
                corrected_pose_list.append(pose_priori)
            else:
                # 对匹配到的柱子全排列找圆
                circle_list = []  # 每个元素：圆心，半径，打分
                for p in topic_final:
                    for i in range(topic_final.index(p) + 1, len(topic_final)):
                        p1 = pole[p[0]]
                        p2 = pole[topic_final[i][0]]
                        theta_diff = topic_final[i][1] - p[1]
                        o, r = self.find_circle(p1, p2, theta_diff, [x, y])
                        rank_o = p[2] * topic_final[i][2]
                        circle_list.append([o, r, rank_o])
                # 对圆全排列找交点
                point_list = []  # 每个元素：交点，打分
                sum_rank = 0
                for c1 in circle_list:
                    for i in range(circle_list.index(c1) + 1, len(circle_list)):
                        c2 = circle_list[i]
                        intersection, rank = self.find_intersection(c1, c2, x, y)
                        if intersection is not None:
                            point_list.append([intersection, rank])
                            sum_rank = sum_rank + rank
                """
                概率版
                """

                step_x = np.array([i for i in range(-500, 501)])
                base_x = np.zeros(1001)
                base_y = np.zeros(1001)
                for p in point_list:
                    x_mu = p[0][0] - x
                    y_mu = p[0][1] - y
                    x_sigma = 50 / (p[1] / 128)
                    y_sigma = x_sigma
                    base_x = base_x + (p[1] / x_sigma) * 2.71828 ** (-(step_x - x_mu) ** 2 / (2 * x_sigma ** 2))
                    base_y = base_y + (p[1] / y_sigma) * 2.71828 ** (-(step_x - y_mu) ** 2 / (2 * y_sigma ** 2))

                corrected_x = np.unravel_index(np.argmax(base_x), base_x.shape)[0] - 500 + x
                corrected_y = np.unravel_index(np.argmax(base_y), base_y.shape)[0] - 500 + y
                """
                corrected_x = 0
                corrected_y = 0
                for p in point_list:
                    corrected_x = corrected_x + p[0][0] * p[1] / sum_rank
                    corrected_y = corrected_y + p[0][1] * p[1] / sum_rank
                """
                if abs(corrected_x - x) > 100 or abs(corrected_y - y) > 100:
                    corrected_pose_list.append(pose_priori)
                else:
                    corrected_pose = np.array([corrected_x, corrected_y, 0])
                    # 校正角度
                    theta_corrected = 0
                    for p1 in topic_final:
                        theta_p1 = 180 * atan((pole[p1[0]][1] - corrected_y) / (pole[p1[0]][0] - corrected_x)) / pi
                        if pole[p1[0]][0] - corrected_x <= 0:
                            if pole[p1[0]][1] - corrected_y <= 0:
                                theta_p1 = -180 + theta_p1
                            else:
                                theta_p1 = 180 + theta_p1
                        theta1 = theta_p1 + p1[1]
                        if theta1 > 180:
                            theta1 = theta1 - 360
                        elif theta1 <= -180:
                            theta1 = 360 + theta1
                        theta_corrected = theta_corrected + theta1 * p1[2] / sum_r
                    if abs(theta_corrected - theta) > 5:
                        corrected_pose[2] = theta
                    else:
                        corrected_pose[2] = theta_corrected
                    corrected_pose_list.append(corrected_pose)

        pose_corrected = np.array(corrected_pose_list)
        pose_corrected = pose_corrected.T
        pose_real_sample = self.pose_real[:, 0::self.sample_rate]
        error_x_corrected = np.average(abs(pose_corrected[0] - pose_real_sample[0]))
        error_y_corrected = np.average(abs(pose_corrected[1] - pose_real_sample[1]))
        error_theta_corrected = np.average(abs(pose_corrected[2] - pose_real_sample[2]))

        print("新算法平均每帧成功匹配柱子数量：", num_pole_matched_avg / len(self.sample_time))
        T.toc()
        statement_2 = '新校正 平均    x误差 : {0} 毫米, y误差 : {1} 毫米,角度误差 : {2} 度'
        statement_2 = statement_2.format(error_x_corrected, error_y_corrected, error_theta_corrected)

        # self.plot()
        print(statement_2)

    def find_intersection(self, circle_1, circle_2, x, y):
        """
        圆1，圆2，位置先验
        """

        def sgn(num):
            if num < -eps:
                return -1
            elif num > eps:
                return 1
            else:
                return 0

        # 计算两个圆的交点
        def cross_points_of_two_circles(x1, y1, ra1, x2, y2, ra2):
            dx, dy = x2 - x1, y2 - y1
            dis2 = dx ** 2 + dy ** 2

            # 两个圆相离或包含
            if dis2 > (ra1 + ra2) ** 2 or dis2 < (ra1 - ra2) ** 2:
                return []
            # 计算两个圆心的连线与x轴的角度t
            t = atan2(dy, dx)
            # 计算两个圆心的连线与圆心与交点之间的夹角a
            a = acos((ra1 * ra1 - ra2 * ra2 + dis2) / (2 * ra1 * sqrt(dis2)))

            # 计算交点
            x3, y3 = x1 + ra1 * cos(t + a), y1 + ra1 * sin(t + a)
            x4, y4 = x1 + ra1 * cos(t - a), y1 + ra1 * sin(t - a)

            if sgn(a) == 0:  # 两个圆相切，返回1个点
                return [x3, y3]
            else:  # 两个圆相交，返回2个点
                return [[x3, y3], [x4, y4]]

        (o1_x, o1_y), r1, rank1 = circle_1
        (o2_x, o2_y), r2, rank2 = circle_2
        intersection = cross_points_of_two_circles(o1_x, o1_y, r1, o2_x, o2_y, r2)

        if len(intersection) == 2:
            if self.distance(intersection[0], [x, y]) < self.distance(intersection[1], [x, y]):
                return intersection[0], rank1 + rank2
            return intersection[1], rank1 + rank2
        if len(intersection) == 0:
            return None, None
        if len(intersection) == 1:
            return intersection, rank1 + rank2

    @staticmethod
    def match_confusing_pole(pole_to_match, topic, poles_priori):
        outlier_l = pole_to_match[6]
        outlier_r = pole_to_match[7]
        if outlier_l == -1:  # 根据右离散柱匹配
            outlier_r_index_topic = -1
            outlier_r_index_priori = -1
            for p in reversed(topic):
                if p[0] == outlier_r:
                    outlier_r_index_topic = topic.index(p)
                    break
            for p in reversed(poles_priori):
                if p[0] == outlier_r:
                    outlier_r_index_priori = poles_priori.index(p)
                    break
            error = []
            bias_theta_to_ref = pole_to_match[1] - topic[outlier_r_index_topic][1]
            for p in poles_priori:
                error.append(abs(bias_theta_to_ref - (p[1] - poles_priori[outlier_r_index_priori][1])))
            try:
                return poles_priori[error.index(min(error))][0]
            except ValueError:
                return -1
        if outlier_r == -1:  # 根据左离散柱匹配
            outlier_l_index_topic = -1
            outlier_l_index_priori = -1
            for p in topic:
                if p[0] == outlier_l:
                    outlier_l_index_topic = topic.index(p)
                    break
            for p in poles_priori:
                if p[0] == outlier_l:
                    outlier_l_index_priori = poles_priori.index(p)
                    break
            error = []
            bias_theta_to_ref = pole_to_match[1] - topic[outlier_l_index_topic][1]
            for p in poles_priori:
                error.append(abs(bias_theta_to_ref - (p[1] - poles_priori[outlier_l_index_priori][1])))
            try:
                return poles_priori[error.index(min(error))][0]
            except ValueError:
                return -1
        # 被夹在离散柱之间，按照比例匹配
        between_priori = []  # 每个元素：[柱子编号，比例]
        outlier_r_index_topic = -1
        outlier_r_index_priori = -1
        outlier_l_index_topic = -1
        outlier_l_index_priori = -1
        for p in reversed(topic):
            if p[0] == outlier_r:
                outlier_r_index_topic = topic.index(p)
                break
        for p in reversed(poles_priori):
            if p[0] == outlier_r:
                outlier_r_index_priori = poles_priori.index(p)
                break
        for p in topic:
            if p[0] == outlier_l:
                outlier_l_index_topic = topic.index(p)
                break
        for p in poles_priori:
            if p[0] == outlier_l:
                outlier_l_index_priori = poles_priori.index(p)
                break

        def ratio(mid, left, right):
            return (mid - left) / (right - left)

        ratio_to_match = ratio(pole_to_match[1], topic[outlier_l_index_topic][1], topic[outlier_r_index_topic][1])
        for i in range(outlier_l_index_priori + 1, outlier_r_index_priori):
            r = ratio(poles_priori[i][1], poles_priori[outlier_l_index_priori][1], poles_priori[outlier_r_index_priori][1])
            between_priori.append([poles_priori[i][0], r])

        ratio_error = []
        for i in between_priori:
            ratio_error.append(abs(ratio_to_match - i[1]))
        try:
            return between_priori[ratio_error.index(min(ratio_error))][0]
        except ValueError:
            return -1

    def find_circle(self, p1, p2, diff, position):
        # 两个柱子连线中点
        diff = diff / 180 * pi
        center_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        # 弦长
        a = self.distance(p1, p2)
        # 半径
        r = a / 2 / sin(diff)
        # 圆心到弦
        o2a = a / 2 / tan(diff)
        # 中垂线斜率
        try:
            k = (p1[0] - p2[0]) / (p2[1] - p1[1])
            # 求两个候选圆心坐标
            o1 = (center_point[0] + o2a * (1 / sqrt(k ** 2 + 1)), center_point[1] + o2a * (k / sqrt(k ** 2 + 1)))
            o2 = (center_point[0] - o2a * (1 / sqrt(k ** 2 + 1)), center_point[1] - o2a * (k / sqrt(k ** 2 + 1)))
        except ZeroDivisionError:
            o1 = (center_point[0], center_point[1] + o2a)
            o2 = (center_point[0], center_point[1] - o2a)
        if self.distance(o1, position) > self.distance(o2, position):
            o = o2
        else:
            o = o1
        return o, r

    @staticmethod
    def distance(p1, p2):
        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def generate_vision_topic(self):

        message_list = []  # 随时间变化的整个message

        for t in self.sample_time:
            i = self.relative_pose_real[t]

            one_message = []  # 若干可见one_pole
            """
            筛选，装入message队列
            """
            # 按照距离排序

            i_sorted = sorted(i, key=lambda d: d[1])

            for p in i_sorted:
                if p[2] != 10:
                    theta_l = (p[0] / 180) * pi - asin(50 / p[1])
                    theta_r = (p[0] / 180) * pi + asin(50 / p[1])
                else:
                    theta_l = (p[0] / 180) * pi - asin(75 / p[1])
                    theta_r = (p[0] / 180) * pi + asin(75 / p[1])
                if theta_r < self.camera_hv_l or theta_l > self.camera_hv_r:
                    continue
                theta_u = atan((self.camera_height - pole[int(p[2])][3]) / (p[1] - 50))
                theta_d = atan((self.camera_height - pole[int(p[2])][2]) / (p[1] - 50))
                # 角度到像素点
                # 左上
                x1 = self.camera_cx + self.camera_fx * tan(theta_l)
                y1 = self.camera_cy + self.camera_fy * tan(theta_u)

                x2 = self.camera_cx + self.camera_fy * tan(theta_r)
                y2 = self.camera_cx + self.camera_fy * tan(theta_d)

                if x1 < 0:
                    x1 = 0
                if x2 > self.camera_resolution[0]:
                    x2 = self.camera_resolution[0]
                if y1 < 0:
                    y1 = 0
                if y2 > self.camera_resolution[1]:
                    y2 = self.camera_resolution[1]

                # 检查遮挡
                covered = False
                if len(one_message) == 0:
                    xywh = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
                    one_message.append(xywh)
                else:
                    for each_pole in one_message:
                        [x, y, w, h] = each_pole
                        each_pole = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
                        c1, c2, c3, c4, num = self.get_cover_state(x1, y1, x2, y2, each_pole[0], each_pole[1],
                                                                   each_pole[2], each_pole[3])

                        if num == 4:
                            covered = True
                            break
                        if num == 2:
                            if c1 and c2:
                                y1 = each_pole[3]
                            if c1 and c3:
                                x1 = each_pole[2]
                            if c3 and c4:
                                y2 = each_pole[1]
                            if c2 and c4:
                                x2 = each_pole[0]
                    if not covered:
                        if 0 < x1 < x2 < self.camera_resolution[0] and 0 < y1 < y2 < self.camera_resolution[1]:
                            bias_x = random.gauss(0, 3)
                            bias_y = random.gauss(0, 3)
                            x1 = x1 + bias_x
                            y1 = y1 + bias_y
                            x2 = x2 - bias_x
                            y2 = y2 - bias_y
                        if (x2 - x1) * (y2 - y1) > 300:
                            xywh = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
                            one_message.append(xywh)
            message_list.append(one_message)

        for i in message_list:
            for p in i:
                cv2.rectangle(self.image, (int(p[0] - p[2] / 2), int(p[1] - p[3] / 2)),
                              (int(p[0] + p[2] / 2), int(p[1] + p[3] / 2)), (0, 255, 255), 2)

            self.video.append(self.image)
            self.image = np.zeros((self.camera_resolution[1], self.camera_resolution[0], 3), dtype='uint8')
        """
        for i in self.video:
            cv2.imshow("res", i)
            cv2.waitKey(0)
        """
        self.vision_topic = message_list

    def generate_relative_pose_real(self):
        for i in self.time:
            x = self.pose_real[0][i]
            y = self.pose_real[1][i]
            theta = self.pose_real[2][i]
            self.get_relative_pose_real(x, y, theta)

        self.relative_pose_real = np.array(self.relative_pose_real)

    def generate_pose_priori(self):
        x_loc_priori = []
        y_loc_priori = []
        theta_priori = []

        for i in self.sample_time:
            sample_x = round(self.pose_real[0][i] + random.gauss(0, 50))
            sample_y = round(self.pose_real[1][i] + random.gauss(0, 50))
            sample_theta = round(self.pose_real[2][i] + random.gauss(0, 1.5), 3)
            x_loc_priori.append(sample_x)
            y_loc_priori.append(sample_y)
            theta_priori.append(sample_theta)

        self.pose_priori = np.array([x_loc_priori, y_loc_priori, theta_priori])
        pose_real_sample = self.pose_real[:, 0::self.sample_rate]
        """
        计算先验和真值的误差
        """

        error_x_priori = np.average(abs(self.pose_priori[0] - pose_real_sample[0]))
        error_y_priori = np.average(abs(self.pose_priori[1] - pose_real_sample[1]))
        error_theta_priori = np.average(abs(self.pose_priori[2] - pose_real_sample[2]))

        statement_1 = '不校正 平均    x误差 : {0} 毫米, y误差 : {1} 毫米,角度误差 : {2} 度'
        statement_1 = statement_1.format(error_x_priori, error_y_priori, error_theta_priori)

        # self.plot()
        print(statement_1)

    def get_relative_pose_real(self, x, y, theta):
        relative_pose_set = []

        for i in range(11):
            pose = []
            theta_pole = 180 * atan((pole[i][1] - y) / (pole[i][0] - x)) / pi
            if pole[i][0] - x < 0:
                if pole[i][1] - y < 0:
                    theta_pole = -180 + theta_pole
                else:
                    theta_pole = 180 + theta_pole

            theta_relative = theta - theta_pole
            if theta_relative > 180:
                theta_relative = theta_relative - 360
            elif theta_relative <= -180:
                theta_relative = 360 + theta_relative

            pose.append(theta_relative)
            distance = sqrt((pole[i][0] - x) ** 2 + (pole[i][1] - y) ** 2)
            pose.append(distance)
            pose.append(i)
            relative_pose_set.append(pose)

        self.relative_pose_real.append(relative_pose_set)

    def plot_trace_real(self):

        plt.plot(self.pose_real[:][0], self.pose_real[:][1], linewidth=1)
        plt.xlim(0, 12000)
        plt.ylim(0, 12000)
        plt.title('trace_real')

    def plot_theta_real(self):

        plt.plot(self.time, self.pose_real[:][2], linewidth=1)
        plt.title('theta_real')

    def plot_trace_priori(self):

        plt.scatter(self.pose_priori[:][0], self.pose_priori[:][1], s=0.1)
        plt.xlim(0, 12000)
        plt.ylim(0, 12000)
        plt.title('trace_prori')

    def plot_theta_priori(self):

        plt.scatter(self.sample_time, self.pose_priori[:][2], s=0.1)
        plt.title('theta_priori')

    def plot_pole_relative_theta_real(self):

        plt.plot(self.time, self.relative_pose_real[:, :, 0])

        plt.title('relative_theta_real')

    def plot_pole_relative_distance_real(self):
        plt.plot(self.time, self.relative_pose_real[:, :, 1])
        plt.title('relative_distance_real')

    def plot(self):
        plt.figure(figsize=(15, 15), dpi=300)
        plt.subplot(3, 2, 1)
        self.plot_trace_real()
        plt.subplot(3, 2, 3)
        self.plot_theta_real()
        plt.subplot(3, 2, 2)
        self.plot_trace_priori()
        plt.subplot(3, 2, 4)
        self.plot_theta_priori()
        plt.subplot(3, 2, 5)
        self.plot_pole_relative_theta_real()
        plt.subplot(3, 2, 6)
        self.plot_pole_relative_distance_real()
        plt.show()


loc = LocationFuseNode()