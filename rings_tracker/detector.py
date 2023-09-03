#!/bin/python3
# -*- coding utf-8 -*-

"""
重写yolov5,改为单张检测，输入opencvbgr图像
调用detect运行一次单张检测，返回目标xywh信息
"""
import torch
import numpy as np

from models.experimental import attempt_load
from utils.general import (
    non_max_suppression, scale_coords, xyxy2xywh
)
from utils.datasets import letterbox


class Yolo_detect:
    def __init__(self, modelPath, imgSize=640, device=""):
        self.device, self.devicename = self.selectDevice(device)  # 选择cpu或gpu
        self.model = attempt_load(modelPath, map_location=self.device)  # 加载训练集模型
        self.names = (
            self.model.module.names
            if hasattr(self.model, "module")
            else self.model.names
        )  # 获取类别
        self.imgSize = imgSize  # 图片大小

    def selectDevice(self, device):  # 选择使用cpu或gpu(cuda)
        cpu_request = device.lower() == "cpu"  # 判断用户是否制定用cpu
        if (not cpu_request) and torch.cuda.is_available():  # 如果指定gpu且gpu可用
            return torch.device("cuda:0"), "gpu"
        else:  # 如果指定cpu或gpu不可用
            return torch.device("cpu"), "cpu"

    def imgConvert(self, cv_img):  # 对opencv图像进行预处理
        img = letterbox(cv_img, new_shape=self.imgSize)[0]  # 转换长方形至正方形
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)  # 将array转为内存连续存储的数组，提高运行速度
        img = torch.from_numpy(img).to(self.device).float()  # opencv转torch
        img /= 255.0  # 归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect(self, cv_img):  # 主函数
        img = self.imgConvert(cv_img)  # 图片格式转换
        det = self.model(img, augment=False)[0]  # 模型检测
        det = non_max_suppression(det, 0.25, 0.45, classes=None, agnostic=False)[
            0
        ]  # 非最大值抑制NMS

        if det is not None and len(det):
            if self.devicename == "cpu":
                det = det.numpy()  # cpu tensor 转 numpy
            elif self.devicename == "gpu":
                det = det.cpu().numpy()  # gpu tensor 转 numpy
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], cv_img.shape).round()
            det[:4] = xyxy2xywh(det[:4])  # 左上右下转为中心宽高

        return det
