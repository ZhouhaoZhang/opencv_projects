import cv2
from matplotlib import pyplot as plt
img_bgr = cv2.imread('ball.jpg')
img = cv2.imread('ball.jpg', 0)
tem = cv2.imread('ball_t.jpg', 0)
res = cv2.matchTemplate(img, tem, cv2.TM_SQDIFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
(x, y) = min_loc
(X, Y) = tem.shape
(u, v) = (x + X / 2, y + Y / 2)
print(u, v)

FX = 1823.4364
FY = 1827.9003
CX = 631.4012
CY = 294.1035
W = 4

U = ((u - CX) * W) / FX
V = ((v - CY) * W) / FY
print(U, V)

match_result = cv2.rectangle(img_bgr, (x, y), (X + x, Y + y), (0, 0, 255), 8)
b, g, r = cv2.split(match_result)  # 颜色通道提取
match_result_rgb = cv2.merge((r, g, b))  # 颜色通道合并


plt.subplot(121), plt.imshow(match_result_rgb)
plt.subplot(122), plt.imshow(res), plt.title("SQDIFF_NORMED")
plt.show()
