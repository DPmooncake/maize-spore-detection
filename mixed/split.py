import os
import shutil
import random
from sklearn.model_selection import train_test_split

# 配置路径 (⚠️修改为你的实际路径)
root_path = r"E:\deep_learn\spore\mixed\VOCdevkit"
source_img_dir = os.path.join(root_path, "VOC2007", "JPEGImages")  # 源图片路径
source_label_dir = os.path.join(root_path, "VOC2007", "YOLOLabels")  # 源标签路径


# 创建目标文件夹结构
def create_dirs():
    dirs = [
        os.path.join(root_path, "images", "train"),
        os.path.join(root_path, "images", "val"),
        os.path.join(root_path, "images", "test"),
        os.path.join(root_path, "labels", "train"),
        os.path.join(root_path, "labels", "val"),
        os.path.join(root_path, "labels", "test")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")


# 获取文件列表并检查对应关系
def get_file_list():
    # 获取所有图片文件名（不带扩展名）
    all_files = [f.split(".")[0] for f in os.listdir(source_img_dir) if f.endswith(".jpg")]

    # 验证标签是否存在
    valid_files = []
    for f in all_files:
        label_path = os.path.join(source_label_dir, f + ".txt")
        if os.path.exists(label_path):
            valid_files.append(f)
        else:
            print(f"⚠️ 缺少标签文件: {f}.txt")

    print(f"\n有效文件总数: {len(valid_files)}")
    print(f"缺失标签文件数: {len(all_files) - len(valid_files)}\n")
    return valid_files


# 分割并复制文件
def split_and_copy(file_list):
    # 第一次拆分：训练集80%，临时集20%
    train_files, temp_files = train_test_split(file_list, test_size=0.2, random_state=42)

    # 第二次拆分：验证集50%，测试集50%
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    print(f"训练集数量: {len(train_files)}")
    print(f"验证集数量: {len(val_files)}")
    print(f"测试集数量: {len(test_files)}")

    # 自定义复制函数
    def copy_files(files, split_type):
        for name in files:
            # 复制图片
            src_img = os.path.join(source_img_dir, name + ".jpg")
            dst_img = os.path.join(root_path, "images", split_type, name + ".jpg")
            shutil.copyfile(src_img, dst_img)

            # 复制标签
            src_label = os.path.join(source_label_dir, name + ".txt")
            dst_label = os.path.join(root_path, "labels", split_type, name + ".txt")
            shutil.copyfile(src_label, dst_label)

    # 执行复制操作
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")
    print("\n文件复制完成！")


if __name__ == "__main__":
    create_dirs()
    valid_files = get_file_list()
    split_and_copy(valid_files)


