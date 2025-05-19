from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__': 
    model = YOLO("E:/deep_learn/spore/mixed/ultralytics-main/ultralytics/cfg/models/v8/v8_SPD_GAM.yaml")   # 修改模型结构，不进行预训练
    # model = YOLO("E:/deep_learn/spore/mixed/ultralytics-main/yolov8s.pt")   # 用原有权重进行预训练
    model.train(data='E:/deep_learn/spore/mixed/ultralytics-main/ultralytics/cfg/datasets/mixed.yaml', epochs = 100, batch = 2, lr0 = 0.01, close_mosaic=0)
    result = model.val()
