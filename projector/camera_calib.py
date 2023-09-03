import argparse
import os
from glob import glob
import numpy as np
import cv2


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext


def calibrate(args):
    check_path(args.out_path)
    out_chess_path = os.path.join(args.out_path, "output")
    check_path(out_chess_path)
    # 棋盘格图像文件
    # img_names = sorted(os.listdir(args.chess_path))
    file_names = glob(os.path.join(args.chess_path, "*.jpg"))
    # 生成棋盘格世界坐标
    pattern_points = np.zeros((np.prod(args.pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(args.pattern_size).T.reshape(-1, 2)
    pattern_points *= args.square_size

    obj_points = []
    img_points = []
    h, w = cv2.imread(file_names[0], 0).shape[:2]

    # 多线程获取检测棋盘格角点坐标
    def processImage(fn):
        print('processing %s... ' % fn)
        img = cv2.imread(fn, 0)
        if img is None:
            print("Failed to load", fn)
            return None

        assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
        found, corners = cv2.findChessboardCorners(img, args.pattern_size)
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis, args.pattern_size, corners, found)
        _path, name, _ext = splitfn(fn)
        outfile = os.path.join(out_chess_path, name + '_chess.png')
        cv2.imwrite(outfile, vis)

        if not found:
            print('chessboard not found')
            return None
        print('           %s... OK' % fn)
        return corners.reshape(-1, 2), pattern_points

    threads_num = args.threads
    if threads_num <= 1:
        chessboards = [processImage(fn) for fn in file_names]
    else:
        print("Run with %d threads..." % threads_num)
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool(threads_num)
        chessboards = pool.map(processImage, file_names)

    chessboards = [x for x in chessboards if x is not None]
    for (corners, pattern_points) in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)

    # 计算相机畸变
    rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    np.savez("calibrate_parm.npz", camera_matrix=camera_matrix, dist_coefs=dist_coefs)
    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())

    # 图像矫正
    for fn in file_names:
        _path, name, _ext = splitfn(fn)
        img_found = os.path.join(out_chess_path, name + '_chess.png')
        outfile = os.path.join(out_chess_path, name + '_undistorted.png')
        outfile_crop = os.path.join(out_chess_path, name + '_undistorted_crop.png')
        img = cv2.imread(img_found)
        if img is None:
            continue

        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
        dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
        x, y, w, h = roi
        dst_crop = dst[y:y + h, x:x + w]
        cv2.imwrite(outfile, dst)
        cv2.imwrite(outfile_crop, dst_crop)
        print('Undistorted image written to: %s' % outfile)

    if args.real_path:
        out_real_path = os.path.join(args.out_path, "output_real")
        check_path(out_real_path)
        for file in os.listdir(args.real_path):
            fn = os.path.join(args.real_path, file)
            _path, name, _ext = splitfn(fn)
            outfile = os.path.join(out_real_path, name + '_undistorted.png')
            outfile_crop = os.path.join(out_real_path, name + '_undistorted_crop.png')
            real_img = cv2.imread(fn)
            if real_img is None:
                continue
            h, w = real_img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
            dst = cv2.undistort(real_img, camera_matrix, dist_coefs, None, newcameramtx)
            x, y, w, h = roi
            dst_crop = dst[y:y + h, x:x + w]
            cv2.imwrite(outfile, dst)
            cv2.imwrite(outfile_crop, dst_crop)
            print('Undistorted real image written to: %s' % outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chess_path', type=str, default="calibimgs", help="棋盘格图像文件夹")
    parser.add_argument('--real_path', type=str, default="calibimgs", help="实拍图像文件夹")
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--square_size', type=float, default=3.0)
    parser.add_argument('--pattern_size', type=int, nargs=2, default=[11, 8], help="棋盘格尺寸")
    parser.add_argument('--out_path', type=str, default="run")
    opt = parser.parse_args()
    print(opt)
    calibrate(opt)

