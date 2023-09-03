import numpy as np
import cv2
import os
from detector import Yolo_detect

resolution = (1280, 720)
# 左目内参，3x3
camera_mtx_left = np.matrix([[703.379983, 0.0, 706.716016], [0.0, 704.567618, 366.667722], [0.0, 0.0, 1.0]],
                            dtype="double")
camera_mtx_left_I = camera_mtx_left.I
lcx = float(camera_mtx_left[0, 2])
lfx = float(camera_mtx_left[0, 0])
# 右目内参 3x3
camera_mtx_right = np.matrix([[702.041773, 0.0, 699.569314], [0.0, 703.415283, 363.008182], [0.0, 0.0, 1.0]],
                             dtype="double")
camera_mtx_right_I = camera_mtx_right.I
rcx = float(camera_mtx_right[0, 2])
rfx = float(camera_mtx_right[0, 0])
# 左目在右目的描述（齐次） 4x4
mtx_linr = np.matrix([[0.999999371905613, 0.0002650624894432946, 0.0010890042497998867, -0.12053591498829215],
                      [-0.00026119944106836845, 0.9999936790441695, -0.00354593380627485, -0.00011260493080025077],
                      [-0.0010899372602942163, 0.0035456471317923616, 0.9999931201879269, 0.000992993296107907], [0, 0, 0, 1]], dtype="double")

B = 0.12053  # 两个摄像头的眼距


class Analyzer:
    def __init__(self):
        self.vc = cv2.VideoCapture('test.avi')
        self.out = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                   (1280 * 2, 720))

        if self.vc.isOpened():
            self.op = True
        else:
            self.op = False
        self.ptPath = os.path.join(os.path.dirname(__file__), "yolov5s.pt")  # 训练集路径
        self.yoloModel = Yolo_detect(self.ptPath, device="gpu")  # 加载yolov5模型

        self.frame = None
        self.frame_l = None
        self.frame_r = None
        self.showimg = None

        self.detect_result_l_matched = []
        self.detect_result_r_matched = []

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
        优点是考虑了两个摄像头的装配误差，不共轴的误差，精度高
        缺点是计算量大，（因为目前的参数没调好，所以有时候会计算出错）
        而且需要事先估计一个距离z_c作为迭代求解的初始值。这个值可以用上一帧求出来的结果

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
        """
        通过构造三角形来求解，优点是计算量小，缺点是忽略了一些装配上的误差，精度可能会有所降低，但应该也够用了
        """

        ldx = lx - lcx
        rdx = rx - rcx
        z_c = B / (ldx / lfx - rdx / rfx)
        posi_left_pic = np.matrix([lx, ly, 1], dtype="double")
        posi_left_pic = posi_left_pic.reshape((3, 1))
        posi_left_cam = self.pic2cam(0, posi_left_pic, z_c)  # 3x1
        return [float(posi_left_cam[0][0]), float(posi_left_cam[1][0]), z_c]

    def detect_and_match(self):
        """
        检测画面中的目标，筛选，匹配
        """
        self.frame_l = self.frame[:, :int(self.frame.shape[1] / 2)]
        self.frame_r = self.frame[:, int(self.frame.shape[1] / 2):]
        detect_result_l = self.yoloModel.detect(self.frame_l)
        detect_result_r = self.yoloModel.detect(self.frame_r)

        self.detect_result_l_matched = []
        self.detect_result_r_matched = []

        for tl in detect_result_l:
            pt1 = (int(tl[0] - tl[2] / 2), int(tl[1] - tl[3] / 2))
            pt2 = (int(tl[0] + tl[2] / 2), int(tl[1] + tl[3] / 2))
            self.showimg = cv2.rectangle(self.showimg, pt1, pt2, (80, 80, 255), 3)

        for tr in detect_result_r:
            pt1 = (int(tr[0] - tr[2] / 2 + self.showimg.shape[1] / 2), int(tr[1] - tr[3] / 2))
            pt2 = (int(tr[0] + tr[2] / 2 + self.showimg.shape[1] / 2), int(tr[1] + tr[3] / 2))
            self.showimg = cv2.rectangle(self.showimg, pt1, pt2, (80, 80, 255), 3)

        for tl in detect_result_l:
            similarity = []
            for tr in detect_result_r:
                res = cv2.matchTemplate(cv2.resize(tl, (100, 100)), cv2.resize(tr, (100, 100)), cv2.TM_SQDIFF)
                similarity.append(int(res[0]))
            index = similarity.index(min(similarity))

            if min(similarity) < 1e8:
                self.detect_result_l_matched.append(tl)
                self.detect_result_r_matched.append(detect_result_r[index])

        for tl in self.detect_result_l_matched:
            pt1 = (int(tl[0] - tl[2] / 2), int(tl[1] - tl[3] / 2))
            pt2 = (int(tl[0] + tl[2] / 2), int(tl[1] + tl[3] / 2))
            self.showimg = cv2.rectangle(self.showimg, pt1, pt2, (80, 255, 80), 3)

        for tr in self.detect_result_r_matched:
            pt1 = (int(tr[0] - tr[2] / 2 + self.showimg.shape[1] / 2), int(tr[1] - tr[3] / 2))
            pt2 = (int(tr[0] + tr[2] / 2 + self.showimg.shape[1] / 2), int(tr[1] + tr[3] / 2))
            self.showimg = cv2.rectangle(self.showimg, pt1, pt2, (80, 255, 80), 3)

    def main(self):
        while self.op:
            ret, self.frame = self.vc.read()
            if self.frame is None:
                break
            self.showimg = self.frame.copy()

            self.detect_and_match()
            for i in range(len(self.detect_result_l_matched)):
                x1, y1, w1, h1, conf1, class1 = self.detect_result_l_matched[i]
                x2, y2, w2, h2, conf2, class2 = self.detect_result_r_matched[i]
                X, Y, Z = self.position_solver_2(x1, y1, x2, y2)
                X, Y, Z = round(X, 2), round(Y, 2), round(Z, 2)
                self.showimg = cv2.putText(self.showimg, "x:" + str(X), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (5, 5, 150), 3)
                self.showimg = cv2.putText(self.showimg, "y:" + str(Y), (int(x1), int(y1) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (5, 150, 5), 3)
                self.showimg = cv2.putText(self.showimg, "z:" + str(Z), (int(x1), int(y1) + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 5, 5), 3)

            cv2.imshow("result", self.showimg)
            cv2.waitKey(1)
            self.out.write(self.showimg)


if __name__ == "__main__":
    an = Analyzer()
    an.main()
