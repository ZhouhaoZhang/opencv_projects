import numpy as np
import cv2
import matplotlib.pyplot as plt

# 左目内参，3x3
camera_mtx_left = np.matrix([[1478.1207795887253, 0.0, 1066.4054725787548], [0.0, 1477.7926728669713, 561.1241011045615], [0.0, 0.0, 1.0]],
                            dtype="double")
camera_mtx_left_I = camera_mtx_left.I
lcx = float(camera_mtx_left[0, 2])
lfx = float(camera_mtx_left[0, 0])
# 右目内参 3x3
camera_mtx_right = np.matrix([[1478.7952036311003, 0.0, 1061.4781110020526], [0.0, 1480.415316375148, 562.571757392987], [0.0, 0.0, 1.0]],
                             dtype="double")
camera_mtx_right_I = camera_mtx_right.I
rcx = float(camera_mtx_right[0, 2])
rfx = float(camera_mtx_right[0, 0])
# 左目在右目的描述（齐次） 4x4
mtx_linr = np.matrix([[0.999999371905613, 0.0002650624894432946, 0.0010890042497998867, -0.12053591498829215],
                      [-0.00026119944106836845, 0.9999936790441695, -0.00354593380627485, -0.00011260493080025077],
                      [-0.0010899372602942163, 0.0035456471317923616, 0.9999931201879269, 0.000992993296107907], [0, 0, 0, 1]], dtype="double")
B = 0.12053
ROI_W = 600
ROI_H = 600


class Analyzer:
    def __init__(self):
        self.vc = cv2.VideoCapture('./videos/1.avi')
        if self.vc.isOpened():
            self.op = True
        else:
            self.op = False

        self.detect_result = [[0] * 4] * 2

        self.roi = [[0] * 4] * 2

        self.area = [None] * 2

        self.xywh = [[0, 0, 0, 0], [0, 0, 0, 0]]

        self.posi_x = []
        self.posi_y = []
        self.posi_z = []
        self.posi_ana_time = []

        self.orintation_rolling = []
        self.orintation_pitch = []
        self.orintation_ana_time = []

        self.frame = None
        self.showimg = None

        self.z_c_old = 4

        self.fig = plt.figure()
        # 创建3d绘图区域
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        self.frame_count = 0

        # 卡尔曼滤波器

        # roi滤波器
        kalman_filter_l = cv2.KalmanFilter(4, 2)
        kalman_filter_r = cv2.KalmanFilter(4, 2)
        self.kalman_filter_pix = [kalman_filter_l, kalman_filter_r]
        for fi in self.kalman_filter_pix:
            fi.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
            fi.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
            fi.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1

        # position滤波器
        self.kalman_filter_posi = cv2.KalmanFilter(6, 3)
        self.kalman_filter_posi.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], np.float32)
        self.kalman_filter_posi.transitionMatrix = np.array(
            [[1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], np.float32)
        self.kalman_filter_posi.processNoiseCov = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], np.float32) * 1

        self.current_measurement_pix = [np.array((2, 1), np.float32)] * 2
        self.current_measurement_posi = np.array((3, 1), np.float32)
        self.current_prediction_pix = [np.array((2, 1), np.float32)] * 2
        self.current_prediction_posi = np.array((3, 1), np.float32)

        self.last_measurement_pix = [np.array((2, 1), np.float32)] * 2
        self.last_measurement_posi = np.array((3, 1), np.float32)
        self.last_prediction_pix = [np.array((2, 1), np.float32)] * 2
        self.last_prediction_posi = np.array((3, 1), np.float32)

        self.analyzer_started = False
        self.solvable = True

    @staticmethod
    def pic2cam(camid, posi_pic, z_c):
        """
        由一个假设的z_c和像素坐标，求相机坐标
        :param camid: 相机编号 0: left 1: right
        :param posi_pic: 目标的像素坐标 3x1
        :param z_c: 猜测的z_c
        :return: 目标的相机坐标 3x1
        """
        if camid == 0:
            cam_mtx_i = np.matrix(camera_mtx_left_I)
        else:
            cam_mtx_i = np.matrix(camera_mtx_right_I)
        return z_c * cam_mtx_i * posi_pic

    @staticmethod
    def cam2pic(camid, posi_cam):
        """
        由相机坐标求像素坐标
        :param camid: 相机编号 0: left 1: right
        :param posi_cam: 目标的相机坐标  3x1
        :return: 目标的像素坐标 3x1
        """
        if camid == 0:
            cam_mtx = camera_mtx_left
        else:
            cam_mtx = camera_mtx_right

        # 根据相机坐标给出目标在像素坐标下的坐标 3x1
        return float(1 / posi_cam[2][0]) * cam_mtx * posi_cam

    def reprojection_error(self, z_c, lx, ly, rx, ry):
        """
        由猜测的z_c(到左目) 求目标在右目的重投影误差
        :param z_c: 猜测的z_c 到左目
        :param lx: 目标在左目的像素坐标x
        :param ly: 目标在左目的像素坐标y
        :param rx: 目标在右目的像素坐标x
        :param ry: 目标在右目的像素坐标y
        :return: 重投影误差
        """
        # 根据所给的z_c求目标在右目的重投影误差
        # 1.求目标在左目的相机坐标
        posi_left_pic = np.matrix([lx, ly, 1], dtype="double")
        posi_left_pic = posi_left_pic.reshape((3, 1))
        posi_left_cam = self.pic2cam(0, posi_left_pic, z_c)  # 3x1
        # 2.求posi_right_cam
        posi_left_cam = np.concatenate((posi_left_cam, [[1]]), axis=0)
        posi_right_cam = mtx_linr * posi_left_cam
        posi_right_cam = posi_right_cam[:3, :]
        # 3.求重投影点
        posi_right_pic_reprojection = self.cam2pic(1, posi_right_cam)

        error = (posi_right_pic_reprojection[1] - ry) ** 2 + (posi_right_pic_reprojection[0] - rx) ** 2

        error = error[0][0]
        return float(error)

    def position_solver(self, z_c, lx, ly, rx, ry):
        """
        通过优化右目重投影误差来确定目标在左目的相机坐标
        :param z_c: 开始迭代的z_c
        :param lx: 目标的左目像素坐标x
        :param ly: 目标的左目像素坐标y
        :param rx: 目标的右目像素坐标x
        :param ry: 目标的右目像素坐标y
        :return: 目标在左目的相机坐标 非齐次 3x1
        """
        eps = 0.001
        while True:
            err_l = self.reprojection_error(z_c - eps, lx, ly, rx, ry)
            err_r = self.reprojection_error(z_c + eps, lx, ly, rx, ry)
            k = err_r - err_l
            if abs(k * 0.001) <= 0.00001:
                break
            z_c = z_c - k * 0.001

        posi_left_pic = np.matrix([lx, ly, 1], dtype="double")
        posi_left_pic = posi_left_pic.reshape((3, 1))
        posi_left_cam = self.pic2cam(0, posi_left_pic, z_c)  # 3x1
        return [float(posi_left_cam[0][0]), float(posi_left_cam[1][0]), z_c]

    def position_solver_2(self, lx, ly, rx, ry):
        ldx = lx - lcx
        rdx = rx - rcx
        z_c = B / (ldx / lfx - rdx / rfx)
        posi_left_pic = np.matrix([lx, ly, 1], dtype="double")
        posi_left_pic = posi_left_pic.reshape((3, 1))
        posi_left_cam = self.pic2cam(0, posi_left_pic, z_c)  # 3x1
        return [float(posi_left_cam[0][0]), float(posi_left_cam[1][0]), z_c]

    def orintation_solver(self):
        area = self.frame[int(self.xywh[0][1] - self.xywh[0][3] / 2 - 20):int(self.xywh[0][1] + self.xywh[0][3] / 2 + 20), int(
            self.xywh[0][0] - self.xywh[0][2] / 2 - 20):int(
            self.xywh[0][0] + self.xywh[0][2] / 2 + 20)]

        b, g, r = cv2.split(area)
        area = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)

        lower_threshold = np.array([100, 20, 20])
        upper_threshold = np.array([145, 255, 200])

        mask = cv2.inRange(area, lower_threshold, upper_threshold)

        cv2.imshow("h", mask)

        cv2.waitKey(0)
        print(area)

    @staticmethod
    def update_roi(x, y):
        x = float(x)
        y = float(y)
        x_ = x - ROI_W / 2
        y_ = y - ROI_H / 2
        if x_ <= 0:
            x_ = 0
        if y_ <= 0:
            y_ = 0

        return [int(x_), int(y_), int(ROI_W), int(ROI_H)]

    def main(self):
        self.frame_count = 0
        while self.op:
            ret, self.frame = self.vc.read()
            if self.frame is None:
                break

            self.frame_count = self.frame_count + 1
            print(self.frame_count)
            cv2.imshow("frame", self.frame)
            key = cv2.waitKey(0)
            if key != ord('s'):
                continue
            if key == ord('s'):
                print("务必先框左边，再框右边。")
                # 框选ROI区域
                for i in range(2):
                    self.roi[i] = cv2.selectROI('frame', self.frame, showCrosshair=True, fromCenter=False)
                    self.xywh[i][0] = int(self.roi[i][0] + self.roi[i][2] / 2)
                    self.xywh[i][1] = int(self.roi[i][1] + self.roi[i][3] / 2)
                print(self.xywh)
                posi = self.position_solver_2(self.xywh[0][0], self.xywh[0][1], self.xywh[1][0] - 1920, self.xywh[1][1])

                # print(self.xywh[0][0], self.xywh[0][1], self.xywh[1][0]-1920, self.xywh[1][1])

                self.posi_x.append(posi[0])
                self.posi_y.append(posi[1])
                self.posi_z.append(posi[2])

                self.z_c_old = posi[2]
                print("++++++++++++++++++++++++++++++++++++++++++")
                print("frame:", self.frame_count)

                print("posi:", posi)
                print("++++++++++++++++++++++++++++++++++++++++++")


if __name__ == "__main__":
    an = Analyzer()
    an.main()
    an.ax.plot3D(an.posi_x, an.posi_y, an.posi_z, linewidth=1.5)
    an.ax.set_title('Trace')
    plt.show()
