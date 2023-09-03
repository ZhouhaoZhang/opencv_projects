import numpy as np
import cv2
import glob

# 指定包含图像文件的文件夹路径
image_folder = "./calibimgs"

# 使用glob获取文件夹中所有图像文件的路径
images = glob.glob(image_folder + "/*.jpg")  # 根据实际情况修改文件扩展名
# 设置棋盘格的行列数
rows = 8
cols = 11

# 棋盘格每个格子的尺寸（单位：mm）
square_size = 30.0

# 存储棋盘格的角点坐标
obj_points = []
img_points = []

# 准备棋盘格角点的理论坐标
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

for image_name in images:
    img = cv2.imread(image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

    if ret:
        obj_points.append(objp)
        img_points.append(corners)

        # 在图像上绘制角点
        cv2.drawChessboardCorners(img, (cols, rows), corners, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(0)  # 显示图像一段时间，单位ms

cv2.destroyAllWindows()

# 相机标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

print("相机矩阵:")
print(mtx)
print("\n畸变系数:")
print(dist)
