import cv2

# MultiTracker_create以及一些其他的目标追踪算法在opencv4.5以后换了地方.
# cv2.legacy.MultiTracker_create

# 定义OpenCV中的七种目标追踪算法
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
# 读取视频
cap = cv2.VideoCapture(0)

while True:
    flag, frame = cap.read()
    if frame is None:
        break
    # 更新追踪器，追踪目标
    success, boxes = trackers.update(frame)
    # 绘制追踪到的矩形区域
    for box in boxes:
        # box是个浮点型ndarray, 画图需要整型
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 8)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(30)
    if key == ord('s'):
        # 框选ROI区域
        roi = cv2.selectROI('frame', frame, showCrosshair=True, fromCenter=False)
        #         print(roi)
        # 创建一个实际的目标追踪器
        tracker = OPENCV_OBJECT_TRACKERS['csrt']()
        trackers.add(tracker, frame, roi)
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
