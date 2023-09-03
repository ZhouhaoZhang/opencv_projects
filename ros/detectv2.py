#!/home/retr0/anaconda3/bin/python
from lib2to3.pgen2.token import VBAR
from tkinter import Image
import cv2
import rospy
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
import numpy as np
"""
相机内参
焦距(毫米)
fx = f/dx
fy = f/dy
cx
cy
"""
f = 3.5
fx = 966.9573
fy = 963.5666
cx = 1003.4721
cy = 508.0226
resolution = (1920, 1080)

pi = 3.14


class process():
    def __init__(self):
        self.node = rospy.init_node('picture_save_node', anonymous=True)
        self.image_sub = rospy.Subscriber("/image_raw/compressed", CompressedImage, callback=self.CompressedImage_Callback)
        self.rate = rospy.Rate(600)
        self.cvbridge = CvBridge()
        self.cvImage = None
        self.num = 0



    def CompressedImage_Callback(self, data):
        self.cvImage = self.cvbridge.compressed_imgmsg_to_cv2(data, "bgr8") # 将sensor_msgs.Image类型转换为opencv类型

    def main(self):
        '''
        主函数
        '''
        while not rospy.is_shutdown():
            self.rate.sleep()
            if self.cvImage is None or len(self.cvImage) == 0:
                rospy.loginfo('No image')
            else:
                k = cv2.waitKey(1)
                img = self.cvImage
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(img_hsv)
                ret1, v_bin = cv2.threshold(v, 110, 255, cv2.THRESH_BINARY_INV)
                # 开运算
                kernel1 = np.ones((3, 3), np.uint8)  # 产生卷积核
                v_bin = cv2.morphologyEx(v_bin, cv2.MORPH_OPEN, kernel1)
                kernel2 = np.ones((3, 3), np.uint8)

                v_bin = cv2.morphologyEx(v_bin,cv2.MORPH_CLOSE, kernel2)
                # show(v_bin)
                # 找外部边界
                contours, hi = cv2.findContours(
                    v_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                cntsorted = sorted(
                    contours, key=lambda c: cv2.contourArea(c), reverse=True)

                cnt_interest = cntsorted[0:3]
                # print(cnt_interest)
                area_proportion = []
                circle_center_list = []
                #cv2.drawContours(img, cnt_interest, -1, (0, 0, 255), 3)
                for cnt in cnt_interest:
                    (center_x, center_y), r = cv2.minEnclosingCircle(cnt)
                    circle_center_list.append([center_x, center_y])
                    # cv2.circle(img, (int(center_x), int(center_y)), int(r), (0, 255, 0), 1)
                    area_proportion.append(cv2.contourArea(cnt) / (pi * r ** 2))

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img, str(cv2.contourArea(cnt) / (pi * r ** 2)), (int(center_x), int(center_y - r)), font, 0.5,
                                (200, 255, 155), 2, cv2.LINE_AA)
                    # print(ellipse_center)
                area_proportion = np.array(area_proportion)
                #print(area_proportion)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(area_proportion)
                max_loc = max_loc[1]
                #print(circle_center_list)
                center_target = (ctx, cty) = (int(circle_center_list[max_loc][0]), int(circle_center_list[max_loc][1]))
                cv2.circle(img, center_target, 5, (0, 0, 255), 6)
                cv2.line(img, (ctx - 30, cty + 30),(ctx + 30, cty - 30), (255, 255, 0), 2)
                cv2.line(img, (ctx - 30, cty - 30),(ctx + 30, cty + 30), (255, 255, 0), 2)
                # show(img)
                cv2.namedWindow('img',0)
                cv2.resizeWindow('img',1280,720)
                cv2.imshow('img',img)
                # print(img.shape)

if __name__ == '__main__':
    p = process() 
    p.main()
    rospy.spin()
