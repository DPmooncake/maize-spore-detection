import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'E:/deep_learn/spore/mixed/ultralytics-main/runs/detect/train_SPD_PSA_e/weights/best.pt')
    model.val(data='E:/deep_learn/spore/mixed/ultralytics-main/ultralytics/cfg/datasets/mixed.yaml',
              split='val',
              batch=2,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )