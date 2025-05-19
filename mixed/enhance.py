import os
import cv2
import numpy as np
from tqdm import tqdm
from imgaug import augmenters as iaa
import random


def extract_objects(image_dir, label_dir, class_id=1, output_dir="cropped_objects"):
    """
    提取指定类别（class_id）的目标区域
    :param class_id: 少数类别的ID（0或2，根据你的标签顺序）
    """
    os.makedirs(output_dir, exist_ok=True)

    for label_file in tqdm(os.listdir(label_dir)):
        if not label_file.endswith(".txt"):
            continue

        # 读取图像和标注
        img_path = os.path.join(image_dir, label_file.replace(".txt", ".jpg"))
        image = cv2.imread(img_path)
        if image is None:
            continue

        with open(os.path.join(label_dir, label_file), 'r') as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if int(parts[0]) != class_id:
                continue

            # YOLO格式转绝对坐标
            x_center = float(parts[1]) * image.shape[1]
            y_center = float(parts[2]) * image.shape[0]
            width = float(parts[3]) * image.shape[1]
            height = float(parts[4]) * image.shape[0]

            # 裁剪目标区域
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)

            cropped = image[y_min:y_max, x_min:x_max]

            # 保存裁剪后的小目标区域
            if cropped.size > 0:
                output_path = os.path.join(output_dir, f"{class_id}_{label_file}_{idx}.jpg")
                cv2.imwrite(output_path, cropped)


def generate_synthetic_images(backgrounds_dir, object_imgs_dir, output_dir="synthetic", num_images=1000):
    """
    将增强后的小目标拼接到随机背景上，生成新图像（含YOLO格式标注）
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    # 加载所有背景图（可以使用原始数据集图像作为背景）
    bg_images = [cv2.imread(os.path.join(backgrounds_dir, f)) for f in os.listdir(backgrounds_dir)]
    bg_images = [img for img in bg_images if img is not None]

    # 加载少类目标的裁剪图
    object_images = [cv2.imread(os.path.join(object_imgs_dir, f)) for f in os.listdir(object_imgs_dir)]
    object_images = [img for img in object_images if img is not None]

    # 增强策略
    augmenter = iaa.Sequential([
        iaa.Affine(rotate=(-5, 5), scale=(0.8, 1.2)),
        iaa.MultiplyBrightness((0.8, 1.2)),
        iaa.GaussianBlur(sigma=(0, 0.5))
    ])

    for i in tqdm(range(num_images)):
        # 随机选择一个背景
        bg = random.choice(bg_images).copy()
        h, w = bg.shape[:2]

        # 随机插入5-10个小目标并生成YOLO标注
        txt_lines = []

        for _ in range(random.randint(10, 11)):
            obj = random.choice(object_images)
            obj_aug = augmenter.augment_image(obj)

            # 随机位置和缩放
            max_scale = min(w // obj_aug.shape[1], h // obj_aug.shape[0])
            # print(max_scale)
            scale = random.uniform(0.6, 0.8) * max_scale * 0.1
            obj_aug = cv2.resize(obj_aug, (int(obj_aug.shape[1] * scale), int(obj_aug.shape[0] * scale)))

            # 在背景上粘贴目标
            x = random.randint(0, w - obj_aug.shape[1])
            y = random.randint(0, h - obj_aug.shape[0])
            bg[y:y + obj_aug.shape[0], x:x + obj_aug.shape[1]] = obj_aug

            # 生成YOLO格式标注（类别0为玉米大斑病，可替换成你的类别ID）
            x_center = (x + obj_aug.shape[1] / 2) / w
            y_center = (y + obj_aug.shape[0] / 2) / h
            width = obj_aug.shape[1] / w
            height = obj_aug.shape[0] / h
            txt_lines.append(f"2 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # 保存合成图像和标注
        cv2.imwrite(f"{output_dir}/images/synthetic2_{i}.jpg", bg)
        with open(f"{output_dir}/labels/synthetic2_{i}.txt", 'w') as f:
            f.writelines(txt_lines)

# ----------------- 提取少数类目标 -----------------
# 提取玉米大斑病（假设类别ID为1）
extract_objects(image_dir="E:/deep_learn/spore/VOCdevkit/images/train",
                label_dir="E:/deep_learn/spore/VOCdevkit/labels/train",
                class_id=2,
                output_dir="cropped_class2")

# # 提取玉米锈病（假设类别ID为2）
# extract_objects(image_dir="your/images",
#                 label_dir="your/labels",
#                 class_id=2,
#                 output_dir="cropped_class2")

# ----------------- 生成合成图像 -------------------
# 为两个少类生成各1000张新样本
generate_synthetic_images(backgrounds_dir="E:/spore/ddbfb/JPEGImages",  # 使用原始图像作为背景池
                         object_imgs_dir="cropped_class2",
                         output_dir="synthetic_class2",
                         num_images=1000)

# generate_synthetic_images(backgrounds_dir="your/images",
#                          object_imgs_dir="cropped_class2",
#                          output_dir="synthetic_class2",
#                          num_images=1000)
