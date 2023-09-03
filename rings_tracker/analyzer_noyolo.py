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
mtx_rinl = mtx_linr.I
B = 0.12053
OPENCV_OBJECT_TRACKERS = {
    'boosting': cv2.legacy.TrackerBoosting_create,
    'csrt': cv2.legacy.TrackerCSRT_create,
    'kcf': cv2.legacy.TrackerKCF_create,
    'mil': cv2.legacy.TrackerMIL_create,
    'tld': cv2.legacy.TrackerTLD_create,
    'medianflow': cv2.legacy.TrackerMedianFlow_create,
    'mosse': cv2.legacy.TrackerMOSSE_create
}
# 初始化追踪器集合
trackers = cv2.legacy.MultiTracker_create()
fig = plt.figure()
# 创建3d绘图区域
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

pose_x = []
pose_y = []
pose_z = []


class Analyzer:
    def __init__(self):
        self.frame = None
        self.vc = cv2.VideoCapture('./zed2.avi')
        if self.vc.isOpened():
            self.op = True
        else:
            self.op = False

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
            cam_mtx_I = np.matrix(camera_mtx_left_I)
        else:
            cam_mtx_I = np.matrix(camera_mtx_right_I)
        return z_c * cam_mtx_I * posi_pic

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
        cnt = 0

        while True:
            err_l = self.reprojection_error(z_c - eps, lx, ly, rx, ry)
            err_r = self.reprojection_error(z_c + eps, lx, ly, rx, ry)
            k = err_r - err_l
            cnt += 1
            step = k * 0.0001

            if abs(step) <= 0.001:
                break

            z_c -= step

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

    def main(self):
        count = 0
        z_c_old = 0.5
        while self.op:
            ret, frame = self.vc.read()
            if frame is None:
                break
            count = count + 1
            key = cv2.waitKey(30)
            if key == ord('s'):
                # 框选ROI区域
                roi = cv2.selectROI('frame', frame, showCrosshair=True, fromCenter=False)
                #         print(roi)
                # 创建一个实际的目标追踪器
                trackerl = OPENCV_OBJECT_TRACKERS['csrt']()
                trackers.add(trackerl, frame, roi)

                roi = cv2.selectROI('frame', frame, showCrosshair=True, fromCenter=False)
                trackerr = OPENCV_OBJECT_TRACKERS['csrt']()
                trackers.add(trackerr, frame, roi)
                continue
            # 更新追踪器，追踪目标
            success, boxes = trackers.update(frame)
            # 绘制追踪到的矩形区域
            boxes = sorted(boxes, key=lambda bo: bo[0], reverse=False)
            for box in boxes:
                # box是个浮点型ndarray, 画图需要整型
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 8)

            showimg = frame.copy()
            if boxes:
                posi = self.position_solver(z_c_old, boxes[0][0], boxes[0][1], boxes[1][0] - 1920, boxes[1][1])
                showimg = cv2.putText(showimg, "x:" + str(posi[0]), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (5, 5, 150), 3)
                showimg = cv2.putText(showimg, "y:" + str(posi[1]), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (5, 150, 5), 3)
                showimg = cv2.putText(showimg, "z(depth):" + str(posi[2]), (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (150, 5, 5), 3)
                pose_x.append(posi[0])
                pose_y.append(posi[1])
                pose_z.append(posi[2])
                z_c_old = posi[2]
            cv2.imshow('frame', showimg)
            cv2.waitKey(1)

        return


if __name__ == "__main__":
    an = Analyzer()
    an.main()
    ax.plot3D(pose_x, pose_y, pose_z, linewidth=1.5)
    ax.set_title('Trace')
    plt.show()
