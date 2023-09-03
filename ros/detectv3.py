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
        self.image_sub = rospy.Subscriber("/image_raw/compressed", CompressedImage, callback=self.Callback)
        self.rate = rospy.Rate(600)
        self.cvbridge = CvBridge()
        self.cvImage = None
        self.num = 0
    def Callback(self, data):
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
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                ret1, v_bin = cv2.threshold(v, 80, 255, cv2.THRESH_BINARY_INV)
                # 开运算
                kernel = np.ones((3, 3), np.uint8)  # 产生卷积核
                v_bin = cv2.morphologyEx(v_bin, cv2.MORPH_OPEN, kernel)
                v_bin = cv2.morphologyEx(v_bin,cv2.MORPH_CLOSE, kernel)
                # show(v_bin)
                # 找外部边界
                contours, hierarchy = cv2.findContours(v_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                num_contours = np.shape(hierarchy)[1]
                """ 
            for i in range(num_contours):
                    print(i)
                    print(hierarchy[0][i])
                """
                cnt_depth = []
                for hie in hierarchy[0]:
                    num = 0
                    hie_cpy = hie
                    while hie_cpy[3] != -1:
                        num += 1
                        hie_cpy = hierarchy[0][hie_cpy[3]]
                    cnt_depth.append(num)
                # print(cnt_depth)
                cnt_depth = np.array(cnt_depth)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cnt_depth)
                # show(res1)
                max_loc = max_loc[1]
                #print(circle_center_list)
                
                (center_x, center_y), r = cv2.minEnclosingCircle(contours[max_loc])
                
                
                
                center_target = (ctx, cty) = (int(center_x),int(center_y))
                cv2.circle(img, center_target, 5, (0, 0, 255), 6)
                cv2.line(img, (ctx - 30, cty + 30),(ctx + 30, cty - 30), (255, 255, 0), 2)
                cv2.line(img, (ctx - 30, cty - 30),(ctx + 30, cty + 30), (255, 255, 0), 2)
                # show(img)
                cv2.namedWindow('img',0)
                cv2.resizeWindow('img',1280,720)
                cv2.imshow('img',img)
                # print(img.shape)
                
                
                
                
                
                cv2.namedWindow('img',0)
                cv2.resizeWindow('img',1280,720)
                cv2.imshow('img',img)
                # print(img.shape)

if __name__ == '__main__':
    p = process() 
    p.main()
    rospy.spin()
