import os
import torch
from ultralytics import YOLO
from pathlib import Path
from tkinter import Tk, filedialog, messagebox
from PIL import Image


def predict(model_path, source, save_dir):
    # 加载 YOLOv8 训练好的模型
    model = YOLO(model_path)

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 判断 source 是单张图片还是文件夹
    source_path = Path(source)
    images = [source_path] if source_path.is_file() else list(source_path.glob("*.jpg")) + list(
        source_path.glob("*.png"))

    if not images:
        messagebox.showerror("错误", "未找到可用的图片！")
        return

    for img_path in images:
        results = model(img_path)  # 运行模型预测

        for result in results:
            # 获取保存路径
            save_path = Path(save_dir) / f"{img_path.stem}_pred.jpg"

            # 保存预测结果
            result = results[0]

            # 直接保存可视化后的图像
            Image.fromarray(result.plot()).save(save_path)
            print(f"结果已保存至: {save_path}")

    messagebox.showinfo("完成", f"预测完成！结果保存在: {save_dir}")


if __name__ == "__main__":
    # 隐藏 Tkinter 主窗口
    root = Tk()
    root.withdraw()

    # 选择模型文件
    model_path = filedialog.askopenfilename(title="选择 YOLOv8 模型文件", filetypes=[("PyTorch Model", "*.pt")])
    if not model_path:
        messagebox.showerror("错误", "未选择模型文件！")
        exit()

    # 选择图片或文件夹
    source = filedialog.askopenfilename(title="选择要检测的图片", filetypes=[("Images", "*.jpg;*.png")])
    if not source:
        source = filedialog.askdirectory(title="或者选择一个包含图片的文件夹")
        if not source:
            messagebox.showerror("错误", "未选择图片或文件夹！")
            exit()

    # 选择结果保存文件夹
    save_dir = filedialog.askdirectory(title="选择保存结果的文件夹")
    if not save_dir:
        messagebox.showerror("错误", "未选择保存目录！")
        exit()

    # 运行检测
    predict(model_path, source, save_dir)
