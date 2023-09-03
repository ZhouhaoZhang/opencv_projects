import cv2

count = 0
cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    k = cv2.waitKey(32)
    if k & 0xff == 27:  # 表示Esc退出
        break
    elif k == ord('s'):
        cv2.imwrite(str(count) + '.jpg', frame)
        count+=1
    else:
        continue

cap.release()
