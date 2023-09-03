import numpy as np
import cv2


feature_params = dict(maxCorners=100,  # 最多角点数量
                      qualityLevel=0.1,  # 角点品质
                      minDistance=30)  # 角点之间的最小距离
# lucas kanade参数
lk_params = dict(winSize=(20, 20),  # 搜索窗口大小
                 maxLevel=5)  # 最大金字塔层数
# 随机颜色条
color = np.random.randint(0, 255, (100, 3))
# 拿到第一帧图像
cap = cv2.VideoCapture(0)
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
# 创建一个mask
mask = np.zeros_like(old_frame)
while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # 需要传入前一帧和当前图像以及前一帧检测到的角点
    """
    前一帧，当前帧，待匹配特征点
    """
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # st=1表示
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # 绘制轨迹
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)

    cv2.imshow('frame', img)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

    # 更新
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()