import cv2
import math

resolution = (width, height) = (1280, 720)
center_frame = (center_frame_x, center_frame_y) = (width / 2, height / 2)
PI = 3.1415926
hori = PI / 3
camera_height = 0.565
dis_to_sensor = center_frame_x / math.tan(hori / 2)
ball_r_s = 0.3
ball_r_b = 0.5
img = cv2.imread("two.png")
img = cv2.resize(img, (1280, 720))
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_h, img_s, img_v = cv2.split(img_hsv)


def show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def distance(dot_1, dot_2):
    return math.sqrt((dot_1[0] - dot_2[0]) ** 2 + (dot_1[1] - dot_2[1]) ** 2)


def calculate_dis(contour, ball_r, name):
    ellipse = cv2.fitEllipse(contour)
    cv2.ellipse(img, ellipse, (0, 255, 0), 2)
    ((center_x, center_y), (short_ax, long_ax), ang) = ellipse
    ang_long = (ang - 90) / 180 * PI
    dot1 = (center_x - long_ax / 2 * math.cos(ang_long), center_y - long_ax / 2 * math.sin(ang_long))
    dot2 = (center_x + long_ax / 2 * math.cos(ang_long), center_y + long_ax / 2 * math.sin(ang_long))

    if distance(dot1, center_frame) > distance(dot2, center_frame):
        far = (far_x, far_y) = (dot1[0], dot1[1])
        near = (near_x, near_y) = (dot2[0], dot2[1])
    else:
        near = (near_x, near_y) = (dot1[0], dot1[1])
        far = (far_x, far_y) = (dot2[0], dot2[1])
    cv2.circle(img, (int(center_frame_x), int(center_frame_y)), 3, (0, 0, 255), 3)
    cv2.circle(img, (int(near_x), int(near_y)), 5, (0, 0, 255), 3)
    cv2.circle(img, (int(far_x), int(far_y)), 5, (255, 0, 0), 3)

    theta_near = ((center_x - center_frame_x) * (near_x - center_frame_x)) / (math.fabs(
        (center_x - center_frame_x) * (near_x - center_frame_x))) * math.atan(
        distance(near, center_frame) / dis_to_sensor)
    theta_far = math.atan(distance(far, center_frame) / dis_to_sensor)

    theta_center_ball = (theta_far + theta_near) / 2

    frame_center_dis_to_center_ball = dis_to_sensor * math.tan(theta_center_ball)

    center_ball = (center_ball_x, center_ball_y) = (center_frame_x + (center_x - center_frame_x) / math.fabs(
        center_x - center_frame_x) * frame_center_dis_to_center_ball * math.cos(ang_long),
                                                    center_frame_y + (center_y - center_frame_y) / math.fabs(
                                                        center_y - center_frame_y) * frame_center_dis_to_center_ball * math.fabs(
                                                        math.sin(
                                                            ang_long)))
    cv2.circle(img, (int(center_ball_x), int(center_ball_y)), 5, (255, 255, 0), 5)

    v_pix = center_ball_y - center_frame_y
    u_pix = center_ball_x - center_frame_x
    V = camera_height - ball_r

    U = u_pix * V / v_pix
    W = dis_to_sensor * V / v_pix
    print(name, ':')
    print('(U, V, W) =', (U, V, W))


ret1, img_bin = cv2.threshold(img_h, 90, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cntsorted = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
calculate_dis(cntsorted[0], ball_r_b, 'big')
calculate_dis(cntsorted[1], ball_r_s, 'small')
show('res', img)
