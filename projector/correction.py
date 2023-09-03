import cv2
import numpy as np
import apriltag
import matplotlib.pyplot as plt
from math import asin
from scipy.optimize import minimize

"""
规定坐标系:

图像坐标系，光机坐标系，相机坐标系：传统

世界坐标系：原点和光机坐标系原点重合。光机坐标系按照重力方向反着转回去，z轴，x轴平行于地面，y轴沿重力向下
墙面坐标系：墙上的2D坐标系，y轴沿重力向下

下标：p代表光机，c代表相机，w代表世界，wall代表墙
"""


class KeystoneCorrection:
    def __init__(self, camera_intrinsic_mtx, camera_distortion,
                 projector_intrinsic_mtx, projector_reso,
                 r_cinp, t_cinp, gravity_vector,
                 img_camera, img_projector,
                 ifbiggest):
        # 内参和畸变
        self.Kc = camera_intrinsic_mtx
        self.dist_c = camera_distortion
        self.Kp = projector_intrinsic_mtx
        self.resolution_p = projector_reso
        self.aspect_ratio = self.resolution_p[0] / self.resolution_p[1]

        # 外参，重力
        self.R_cinp = r_cinp  # 相机在光机中的描述
        self.t_cinp = t_cinp  # 相机在光机中的描述
        self.gravity_vector_p = np.array(gravity_vector)  # 光机坐标系下的重力向量
        # 画面和照片
        self.img_p = img_projector
        self.img_c = img_camera
        # 矫正模式,True代表最大，False代表最清晰
        self.ifbiggest = ifbiggest

        self.plot_wall = None
        self.offsets = None  # 四个角点的像素offsets
        self.rpy = None  # 光机在世界中的欧拉角

        self.corrected_corners_p = None

        self.__correct()
        self.__get_rpy()

    @staticmethod
    def __get_keypoints(img) -> np.ndarray:
        def draw_kps():
            image_bgr = cv2.drawChessboardCorners(img, pattern_size, corners, True)
            for r in results:
                if r.tag_id != 26 and r.tag_id != 27 and r.tag_id != 28 and r.tag_id != 29:
                    continue
                for j in r.corners:
                    image_bgr = cv2.circle(image_bgr, np.array(j, dtype="int"), 3, (0, 255, 0), 3)
                image_bgr = cv2.circle(image_bgr, np.array(r.center, dtype="int"), 3, (0, 0, 255), 3)

            plt.imshow(image_bgr)
            plt.show()
            return

        """
        获得图像上的特征点
        返回N*2的array
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 寻找棋盘格角点
        pattern_size = (18, 9)
        retval, corners = cv2.findChessboardCorners(img_gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH)
        kps = np.squeeze(corners, axis=1)

        # 再找apriltag
        # 创建AprilTag检测器
        options = apriltag.DetectorOptions(families='tag16h5')  # 选择AprilTag类型
        detector = apriltag.Detector(options)
        # 进行AprilTag检测
        results = detector.detect(img_gray)
        results = sorted(results, key=lambda x: x.tag_id)
        for i in results:
            if i.tag_id != 26 and i.tag_id != 27 and i.tag_id != 28 and i.tag_id != 29:
                continue
            kps = np.vstack((kps, i.corners))
            kps = np.vstack((kps, np.expand_dims(i.center, axis=0)))

        draw_kps()

        return kps

    @staticmethod
    def __get_plane(kps_xyz) -> list:
        """
        拟合平面
        返回平面参数
        """

        def plot3D() -> None:
            # 创建一个新的三维图形
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')  # 使用Axes3D类
            # 提取坐标分量
            x = kps_xyz[:, 0]
            y = kps_xyz[:, 1]
            z = kps_xyz[:, 2]
            # 绘制三维散点图
            ax.scatter(x, y, z, c='b', marker='o')  # c为颜色，marker为标记类型

            # 绘制拟合的平面
            xx, yy = np.meshgrid(np.linspace(min(x), max(x), 10), np.linspace(min(y), max(y), 10))
            zz = X[0] * xx + X[1] * yy + X[2]
            ax.plot_surface(xx, yy, zz, color='r', alpha=0.3)

            # 设置坐标轴标签
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            # 显示图形
            plt.show()

        # kps_xyz 的每一行表示一个点的 (x, y, z) 坐标
        # 构建系数矩阵 A 和结果向量 b
        A = np.hstack((kps_xyz[:, :2], np.ones((kps_xyz.shape[0], 1))))
        b = kps_xyz[:, 2]

        # 使用最小二乘法公式计算系数向量 x = (a, b, c)
        X = np.linalg.lstsq(A, b, rcond=None)[0]

        # 提取拟合的系数
        a, b, c = X

        # 输出拟合的平面方程系数
        print(f"拟合的平面方程：{a:.4f}x + {b:.4f}y + {c:.4f} = z")

        plot3D()
        return X

    @staticmethod
    def __get_H(kps_from, kps_to) -> np.ndarray:
        """
        寻找单应性矩阵
        返回3*3的array
        """
        H = cv2.findHomography(kps_from, kps_to)[0]
        print("单应性矩阵：\n")
        print(H)
        return H

    def __get_inscribe_rect(self, corners_wall) -> np.ndarray:
        """
        得到内接矩形
        返回4*2的array
        """

        def plot_wall() -> None:
            plt.scatter(corners_wall[:, 0], -corners_wall[:, 1])
            plt.scatter(inscribe_rect_corners[:, 0], -inscribe_rect_corners[:, 1])
            plt.text(0, 0, "y axis is flipped")
            plt.show()

        def slope(point1, point2) -> float:
            try:
                vec = point2 - point1
                return vec[1] / vec[0]
            except ValueError:
                return float('inf')

        def distance(point1, point2) -> float:
            return np.linalg.norm(point2 - point1)

        def point_in_line_by_x(point1, point2, xx) -> np.ndarray:
            # 给定两个点和x，找到两点连线上横坐标为x的点
            y__ = point1[1] + (xx - point1[0]) * (point2[1] - point1[1]) / (point2[0] - point1[0])
            return np.array([xx, y__])

        def point_in_line_by_y(point1, point2, yy) -> np.ndarray:
            # 给定两个点和y，找到两点连线上横坐标为y的点
            x__ = point1[0] + (yy - point1[1]) * (point2[0] - point1[0]) / (point2[1] - point1[1])
            return np.array([x__, yy])

        def point_in_line_by_aspect_ratio(point1, point2, launch_point, aspect_ratio) -> np.ndarray:

            k1 = slope(point2, point1)
            k2 = -1 / aspect_ratio
            b1 = point1[1] - k1 * point1[0]
            b2 = launch_point[1] - k2 * launch_point[0]
            x__ = (b2 - b1) / (k1 - k2)
            y__ = (k1 * b2 - k2 * b1) / (k1 - k2)
            return np.array([x__, y__])

        # 这个函数将所有的寻找内接矩形的任务分成了四种等效的情况。
        # 如果该任意四边形并不直接符合四种情况之一，则可以通过水平和竖直翻转来使其符合
        # 在返回值时，只需要将最终的结果翻转回去即可
        inscribe_rect_corners = np.zeros((4, 2))
        flip_vertical = True  # 先颠倒一下y轴，符合正常数学坐标系，A左上角，B在右上，C在右下，D在左下
        corners_wall[:, 1] = -corners_wall[:, 1]
        flip_horizontal = False
        if self.ifbiggest:
            if distance(corners_wall[1], corners_wall[2]) > distance(corners_wall[0], corners_wall[3]):
                # 水平翻转
                flip_horizontal = True
                corners_wall[:, 0] = -corners_wall[:, 0]
                corners_wall = np.array([corners_wall[1], corners_wall[0], corners_wall[3], corners_wall[2]])

        count = 0
        while True:
            # 得到上，下，左的斜率
            A, B, C, D = np.array(corners_wall)
            k = [slope(A, B), slope(D, A), slope(D, C), ]
            # 找左边的内接
            # 判断上左右边的斜率

            if k[0] > 0 and k[1] > 0 and k[2] > 0:
                # "PP 对角线"情形，对应000，111（需上下翻转）
                # 左上角为PP点，找右下对角线
                inscribe_rect_corners[0] = A
                inscribe_rect_corners[2] = point_in_line_by_aspect_ratio(D, C, A, self.aspect_ratio)
                inscribe_rect_corners[1] = np.array([inscribe_rect_corners[2][0], inscribe_rect_corners[0][1]])
                inscribe_rect_corners[3] = np.array([inscribe_rect_corners[0][0], inscribe_rect_corners[2][1]])
                break
            elif k[0] > 0 > k[2] and k[1] > 0:
                # "PP L"情形，对应001，011（需上下翻转）
                # 左上角为PP点，找竖直向下线
                inscribe_rect_corners[0] = A
                inscribe_rect_corners[3] = point_in_line_by_x(D, C, A[0])
                h = abs(inscribe_rect_corners[0][1] - inscribe_rect_corners[3][1])
                inscribe_rect_corners[1] = np.array([A[0] + self.aspect_ratio * h, A[1]])
                inscribe_rect_corners[2] = np.array([inscribe_rect_corners[1][0], inscribe_rect_corners[3][1]])
                break
            elif k[0] > 0 > k[1] and k[2] > 0:
                # 三点接触情形1，对应010，101（需上下翻转）
                # 左边找点，水平向右与底边交，竖直向上与上边交
                x = ((self.aspect_ratio * (k[1] - k[0]) * A[0]) - ((1 / k[2] - 1 / k[1]) * (A[1] - D[1] - k[1] * A[0]))) / (
                        (1 / k[2] - 1 / k[1]) * k[1] - self.aspect_ratio * (k[0] - k[1]))

                y = point_in_line_by_x(A, D, x)[1]

                w = (1 / k[2] - 1 / k[1]) * (y - D[1])
                h = (k[0] - k[1]) * (x - A[0])
                inscribe_rect_corners[0] = np.array([x, y + h])
                inscribe_rect_corners[1] = np.array([x + w, y + h])
                inscribe_rect_corners[2] = np.array([x + w, y])
                inscribe_rect_corners[3] = np.array([x, y])

                break
            elif k[0] < 0 < k[1] and k[2] > 0:
                # 三点接触情形2，对应100，110（需上下翻转）
                # 左边找点，水平向右与右边交，右下对角线与底边交
                # TODO：写这种情况
                break

            # 跑到这里说明不符合上面四种情况，需要上下翻转，回到之前y轴颠倒的情形
            flip_vertical = False
            corners_wall[:, 1] = -corners_wall[:, 1]
            corners_wall = np.array([corners_wall[3], corners_wall[2], corners_wall[1], corners_wall[0]])
            count += 1
            # TODO: 写一个应急返回值，一般不会发生，以防万一
            # 如果count==2，并且跑到了这里，说明程序出问题了，没找到，需要返回一个应急的结果
            pass

        if flip_vertical:
            # 将y翻转回去
            corners_wall[:, 1] = -corners_wall[:, 1]
            inscribe_rect_corners[:, 1] = -inscribe_rect_corners[:, 1]
        else:
            inscribe_rect_corners = np.array([inscribe_rect_corners[3], inscribe_rect_corners[2], inscribe_rect_corners[1], inscribe_rect_corners[0]])

        if flip_horizontal:
            # 将x翻转回去
            corners_wall[:, 0] = -corners_wall[:, 0]
            inscribe_rect_corners[:, 0] = -inscribe_rect_corners[:, 0]
            inscribe_rect_corners = np.array([inscribe_rect_corners[1], inscribe_rect_corners[0], inscribe_rect_corners[3], inscribe_rect_corners[2]])

        plot_wall()
        return inscribe_rect_corners

    @staticmethod
    def __get_corrected_corners_p(inside_rect_wall, H_wall2p) -> np.ndarray:
        """
        得到矫正后的图像上的四个角点
        返回 4*2的array
        """

        inside_rect_wall_ = np.hstack((inside_rect_wall, np.ones((4, 1))))
        corrected_corners_p_ = H_wall2p @ inside_rect_wall_.T
        # TODO：检查一下结果是不是在显示范围内，否则需要缩小
        return (corrected_corners_p_[:2, :] / corrected_corners_p_[2, :]).T

    # TODO: 补充获得光机欧拉角的代码
    def __get_rpy(self) -> list:
        pass

    def __get_kps_xyz_in_proj(self, kps_p, kps_c) -> np.ndarray:
        """
        返回关键点在光机坐标系下的三维坐标
        返回N*3的array
        """

        def plot3D() -> None:
            # 创建一个新的三维图形
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')  # 使用Axes3D类
            # 提取坐标分量
            x = kps_xyz_p[:, 0]
            y = kps_xyz_p[:, 1]
            z = kps_xyz_p[:, 2]
            # 绘制三维散点图
            ax.scatter(x, y, z, c='b', marker='o')  # c为颜色，marker为标记类型

            # 设置坐标轴标签
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            # 显示图形
            plt.show()

        # 求【R t
        #    0 1】 的逆，得到光机在相机中的描述，也就是相机的外参
        Rt = np.hstack((self.R_cinp, self.t_cinp))  # R和t左右拼接
        Rt01 = np.vstack((Rt, np.array([0, 0, 0, 1])))  # 下方再拼接一个 0 0 0 1
        Rt_I = np.matrix(Rt01).I  # 求逆
        # 相机的投影矩阵，相机内参x相机外参
        projection_matrix_c = self.Kc @ Rt_I[:3, :]
        # 光机的投影矩阵，即光机内参（增广）
        projection_matrix_p = np.hstack((self.Kp, np.zeros((3, 1))))
        # 进行三角测量，注意这里接受的array是2*n的
        kps_xyz = cv2.triangulatePoints(projection_matrix_p,
                                        projection_matrix_c,
                                        kps_p.T,
                                        kps_c.T
                                        ).T
        # 将齐次坐标转换为三维坐标
        kps_xyz_p = kps_xyz[:, :3] / kps_xyz[:, 3:]
        # 返回三维特征点的坐标（光机坐标系下）
        plot3D()
        return kps_xyz_p

    def __get_corners_wall(self, H_wall2p) -> np.ndarray:
        """
        得到墙上的四个角点坐标
        返回4*2array
        """

        def plot_wall() -> None:
            plt.scatter(self.plot_wall[:-4, 0], -self.plot_wall[:-4, 1])
            plt.scatter(self.plot_wall[-4:, 0], -self.plot_wall[-4:, 1])

        corners_p_ = np.array([[0, 0, 1],
                               [self.resolution_p[0], 0, 1],
                               [self.resolution_p[0], self.resolution_p[1], 1],
                               [0, self.resolution_p[1], 1]])
        corners_wall_ = np.matrix(H_wall2p).I @ corners_p_.T
        corners_wall = (corners_wall_[:2, :] / corners_wall_[2, :]).T
        print("四个角点的墙上坐标：")
        print(corners_wall)

        self.plot_wall = np.array(np.vstack((self.plot_wall, corners_wall)))
        plot_wall()

        return np.array(corners_wall)

    def __proj2world(self, kps_xyz_in_p) -> np.ndarray:
        """
        将光机坐标系下的特征点三维坐标转换到世界坐标系下
        返回N*3的array
        """
        G_norm = np.linalg.norm(self.gravity_vector_p)
        G_w = np.array([0, G_norm, 0])
        G_cross = np.cross(self.gravity_vector_p, G_w)
        theta = asin(np.linalg.norm(G_cross) / (G_norm ** 2))
        rotation_vector_p2w = G_cross * (theta / np.linalg.norm(G_cross))
        R_p2w = cv2.Rodrigues(rotation_vector_p2w)[0]
        kps_xyz_w = (R_p2w @ kps_xyz_in_p.T).T
        return kps_xyz_w

    def __world2wall(self, kps_xyz_in_w, plane_coefs) -> np.ndarray:

        """
        将世界坐标系中的三维特征点转换到墙面上的二维点
        返回N*2的array
        """
        # 获取平面系数 ax + by + d = z
        abc = np.array([plane_coefs[0], plane_coefs[1], -1])

        n_norm = np.linalg.norm(abc)
        n_wall = np.array([0, 0, -n_norm])
        n_cross = np.cross(abc, n_wall)
        theta = asin(np.linalg.norm(n_cross) / (n_norm ** 2))
        rotation_vector_w2wall = n_cross * (theta / np.linalg.norm(n_cross))
        R_w2wall = cv2.Rodrigues(rotation_vector_w2wall)[0]

        kps_xy_wall = (R_w2wall @ kps_xyz_in_w.T).T
        kps_xy_wall = kps_xy_wall[:, :2]
        # 重整坐标，保证左上角为0，0
        kps_xy_wall -= np.min(kps_xy_wall, axis=0)

        self.plot_wall = kps_xy_wall

        return kps_xy_wall

    def __correct(self) -> None:
        """
        矫正环节，计算offsets
        """
        # 得到画面和图像中的特征点，一一对应，N*2的array
        kps_p = self.__get_keypoints(self.img_p)
        kps_c = self.__get_keypoints(self.img_c)

        """
        # 在最后一列添加值为 1 的列，增广为齐次坐标
        ones_column = np.ones((kps_p.shape[0], 1))
        corners_p_ = np.hstack((kps_p, ones_column))
        corners_c_ = np.hstack((kps_c, ones_column))
        # 归一化平面坐标
        x_p_ = np.matrix(self.Kp).I @ corners_p_.T
        x_c_ = np.matrix(self.Kc).I @ corners_c_.T
        h = cv2.findHomography(x_c_[:2, :].T, x_p_[:2, :].T)[0]
        # 分解单应性矩阵，恢复旋转和平移
        decomposed_H = cv2.decomposeHomographyMat(h, np.eye(3))

        rotation_matrix = decomposed_H[1]
        translation_vector = decomposed_H[2]

        print("Rotation Matrix:")
        print(rotation_matrix)
        rv = cv2.Rodrigues(rotation_matrix)[0]
        print("rv=")
        print(rv)

        print("\nTranslation Vector:")
        print(translation_vector)
        """
        # 得到光机坐标系下的特征点三维坐标 N*3的array
        kps_xyz_p = self.__get_kps_xyz_in_proj(kps_p, kps_c)
        # 得到世界坐标系下的特征点三维坐标 N*3的array
        kps_xyz_w = self.__proj2world(kps_xyz_p)
        # 拟合平面，得到平面方程的参数
        plane_coefs = self.__get_plane(kps_xyz_w)
        kps_xy_wall = self.__world2wall(kps_xyz_w, plane_coefs)
        # 获得墙面到投影画面的单应性矩阵
        H_wall2p = self.__get_H(kps_xy_wall, kps_p)
        # 由单应性矩阵反推四个墙上角点
        corners_wall = self.__get_corners_wall(H_wall2p)
        # 获得内接矩形
        inside_rect_wall = self.__get_inscribe_rect(corners_wall)
        # 获得画面中的矫正后的角点
        self.corrected_corners_p = np.array(self.__get_corrected_corners_p(inside_rect_wall, H_wall2p), dtype="int")
        # 算offsets

        print(self.corrected_corners_p)

        self.offsets = self.corrected_corners_p - np.array([[0, 0],
                                                            [self.resolution_p[0], 0],
                                                            [self.resolution_p[0], self.resolution_p[1]],
                                                            [0, self.resolution_p[1]]])

        # self.offsets = corrected_corners_p

        return


if __name__ == "__main__":
    # 内参，畸变
    camera_mtx = np.array([[598.46979743, 0., 318.86661906],
                           [0., 598.30945874, 205.43676911],
                           [0., 0., 1., ]])
    projector_mtx = np.array([[2304., 0., 960.], [0., 2309., 1080.], [0., 0., 1.]])
    camera_dist = np.array([2.15362815e-01, - 1.10434423e+00, 4.45313876e-04, 4.08613255e-03,
                            8.52395384e-01])
    projector_resolution = (1920, 1080)
    # 相机在投影仪坐标系下的描述
    """
    R_c2p = np.array((
        [[0.99976794, -0.01013114, 0.01901136],
         [0.0127791, 0.98938046, -0.14478607],
         [-0.01734262, 0.14499541, 0.98928033]]))
    t_c2p = np.array((
        [[-0.94404568],
         [-0.07937728],
         [0.3201203]])) 
    """
    R_c2p = np.array(
        [[0.99968821, - 0.00781298, 0.02371567],
         [0.01143069, 0.9876158, - 0.15647482],
         [-0.02219944, 0.15669712, 0.98739718]],
    )
    t_c2p = np.array((
        [[-0.99487706],
         [-0.01556169],
         [0.09988724]]))
    ratio = -6.75 / t_c2p[0][0]
    t_c2p = t_c2p * ratio

    # 重力向量
    gravity = (0.0001, 9.8, 0.0001)  # 按照规定的坐标系方向的重力加速度
    # 画面和照片
    img_p = cv2.imread("up2.jpg")
    img_c = cv2.imread("uc12.jpg")
    img_c = cv2.undistort(img_c, camera_mtx, camera_dist)
    # 实例化一个矫正器
    correction = KeystoneCorrection(camera_mtx, camera_dist, projector_mtx, projector_resolution, R_c2p, t_c2p, gravity, img_c, img_p, False)
    # 跟矫正器要offsets，欧拉角
    offsets = correction.offsets
    p = correction.corrected_corners_p
    offsets = np.array(offsets, dtype="int")
    A = offsets[3]
    B = offsets[2]
    C = offsets[1]
    D = offsets[0]
    res = "adb shell setprop persist.vendor.hwc.keystone " + str(0 + abs(A[0])) + "," + str(0 + abs(A[1])) + "," + str(
        1920 - abs(B[0])) + "," + str(0 + abs(B[1])) + "," + str(1920 - abs(C[0])) + "," + str(
        1080 - abs(C[1])) + "," + str(0 + abs(D[0])) + "," + str(1080 - abs(D[1]))

    print(str(res))
    img_p = cv2.line(img_p, (int(p[0][0]), int(p[0][1])), (int(p[1][0]), int(p[1][1])), (0, 0, 255), 5)
    img_p = cv2.line(img_p, (int(p[1][0]), int(p[1][1])), (int(p[2][0]), int(p[2][1])), (0, 0, 255), 5)
    img_p = cv2.line(img_p, (int(p[2][0]), int(p[2][1])), (int(p[3][0]), int(p[3][1])), (0, 0, 255), 5)
    img_p = cv2.line(img_p, (int(p[3][0]), int(p[3][1])), (int(p[0][0]), int(p[0][1])), (0, 0, 255), 5)
    plt.imshow(img_p)
    plt.show()
