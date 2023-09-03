"""
录制双面视频脚本
"""
import numpy as np
import pyzed.sl as sl
import cv2
import datetime


class ZedCamera:
    def __init__(self):
        self.zed = sl.Camera()
        self.runtime_param = sl.RuntimeParameters()
        self.image_size, self.image_zed = self.camera_init()
        self.seq = 0
        self.image = None

    def camera_init(self):
        # Create a ZED camera object
        input_type = sl.InputType()  # Set configuration parameters
        init = sl.InitParameters(input_t=input_type)  # Init Parameters
        init.camera_resolution = sl.RESOLUTION.VGA  # set resolution
        init.camera_fps = 100
        init.coordinate_units = sl.UNIT.MILLIMETER
        self.zed.open(init)  # Open the camera

        image_size = self.zed.get_camera_information().camera_resolution  # retrieve half-resolution images
        image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
        return image_size, image_zed

    def cap(self):
        self.zed.grab()
        image_sl_left = sl.Mat()  # left_img
        self.zed.retrieve_image(image_sl_left, sl.VIEW.LEFT)
        image_cv_left = image_sl_left.get_data()
        image_sl_right = sl.Mat()  # right_img
        self.zed.retrieve_image(image_sl_right, sl.VIEW.RIGHT)
        image_cv_right = image_sl_right.get_data()

        ################# BGRA 转成 BGR #################
        image_cv_left = cv2.cvtColor(image_cv_left, 1)
        image_cv_right = cv2.cvtColor(image_cv_right, 1)
        ################# BGRA 转成 BGR #################

        img_sbs = np.concatenate((image_cv_left, image_cv_right), axis=1)

        img_show = img_sbs.copy()
        cv2.resize(img_show, (1200,600))

        cv2.imshow("ZED-img", img_show)
        self.image = img_sbs
        self.seq += 1
        return self.seq, img_sbs


if __name__ == "__main__":
    zed = ZedCamera()
    current_time = datetime.datetime.now()
    out = cv2.VideoWriter('./videos/' + str(current_time) + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 100,
                          (zed.image_size.width * 2, zed.image_size.height))
    while True:
        if zed.zed.grab(zed.runtime_param) == sl.ERROR_CODE.SUCCESS:
            seq, image = zed.cap()
            out.write(image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    out.release()  # 资源释放
    cv2.destroyAllWindows()
