import cv2
import numpy as np


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


kernel = np.ones((5, 5), np.uint8)
kernel2 = np.ones((50, 50), np.uint8)
img1 = cv2.imread('laser.jpg', 0)
ret1, img1 = cv2.threshold(img1, 100, 255, cv2.THRESH_BINARY)

# 旋转，平移


height, width = img1.shape[:2]
# get the center coordinates of the image to create the 2D rotation matrix
center = (width / 2, height / 2)

# using cv2.getRotationMatrix2D() to get the rotation matrix
rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=(0.4114 / 3.14) * 180, scale=1)
img1 = cv2.warpAffine(src=img1, M=rotate_matrix, dsize=(width, height))

translation_matrix = np.array([
    [1, 0, 47.9],
    [0, 1, 188.4]
], dtype=np.float32)
img1 = cv2.warpAffine(src=img1, M=translation_matrix, dsize=(width, height))
cv_show('res', img1)

# img1 = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel2)

# img1 = cv2.resize(img1, (436, 314))

img2 = cv2.imread('world.jpg', 0)

img2 = cv2.dilate(img2, kernel, iterations=1)

# img2 = cv2.resize(img2, (128, 216))
sift = cv2.SIFT_create()
kp1, dst1 = sift.detectAndCompute(img1, None)
kp2, dst2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(crossCheck=True)
'''
# 暴力匹配
matches = bf.match(dst1, dst2)
matches = sorted(matches, key=lambda x: x.distance)
img1 = cv2.imread('laser.jpg')
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:100], None, flags=2)
cv_show('forced', img3)

'''

# knn


bf = cv2.BFMatcher()
matches = bf.knnMatch(dst1, dst2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])
img1 = cv2.imread('laser.jpg')
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
cv_show('knn', img3)
