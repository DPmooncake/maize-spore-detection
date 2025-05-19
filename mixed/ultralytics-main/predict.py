import os
import cv2
import argparse
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path

from PIL import Image


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv8预测脚本')
    parser.add_argument('--model', type=str, default='sporev8s.pt',
                        help='模型权重路径，默认为当前目录下的best.pt')
    parser.add_argument('--source', type=str,
                        default=r'E:\deep_learn\spore\mixed\ultralytics-main\ultralytics\assets',
                        help='输入路径，可以是单张图片或包含图片的文件夹')
    parser.add_argument('--output', type=str, default=r'E:\deep_learn\spore\mixed\ultralytics-main\predict\yolov8s',
                        help='输出目录路径，默认为predict_results')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值（0-1），默认0.25')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='NMS的IOU阈值（0-1），默认0.7')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='推理尺寸（像素），默认640')
    parser.add_argument('--device', type=str, default='0',
                        help='推理设备，如cpu 或 0（GPU），默认使用GPU')
    parser.add_argument('--show', action='store_true',
                        help='是否实时显示预测结果窗口')
    parser.add_argument('--save-txt', action='store_true',
                        help='是否保存检测框标签文件')
    return parser.parse_args()


def prepare_output_dirs(output_root):
    """创建标准化输出目录结构"""
    (output_root / 'images').mkdir(parents=True, exist_ok=True)
    (output_root / 'labels').mkdir(parents=True, exist_ok=True)
    return output_root / 'images', output_root / 'labels'


def process_image(model, img_path, output_img_dir, output_label_dir, args):
    """处理单张图片的预测流程"""
    # 推理预测
    results = model.predict(
        source=str(img_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False  # 关闭详细日志
    )

    # 解析首个结果（因单张输入）
    result = results[0]

    # 保存带标注的图像
    img_result = result.plot()
    img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
    Image.fromarray(img_result).save(str(output_img_dir / img_path.name))

    # 可选：保存标签文件
    if args.save_txt:
        label_path = output_label_dir / f'{img_path.stem}.txt'
        with open(label_path, 'w') as f:
            for box in result.boxes:
                # 转换坐标为YOLO格式（归一化坐标）
                x_center = (box.xyxyn[0][0].item() + box.xyxyn[0][2].item()) / 2
                y_center = (box.xyxyn[0][1].item() + box.xyxyn[0][3].item()) / 2
                width = box.xyxyn[0][2].item() - box.xyxyn[0][0].item()
                height = box.xyxyn[0][3].item() - box.xyxyn[0][1].item()

                line = (f"{int(box.cls.item())} "
                        f"{x_center:.6f} {y_center:.6f} "
                        f"{width:.6f} {height:.6f} "
                        f"{box.conf.item():.6f}\n")
                f.write(line)

    # 可选：显示图像窗口
    if args.show:
        cv2.imshow('Detection', result.plot())
        cv2.waitKey(100)  # 持续显示100ms


def main(args):
    # 初始化模型
    model = YOLO(args.model)
    print(f"已加载模型：{args.model}")

    # 准备输入文件列表
    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"输入路径不存在：{args.source}")

    if source_path.is_file():
        img_paths = [source_path]
    else:
        img_paths = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png'))

    print(f"发现 {len(img_paths)} 张待预测图像")

    # 准备输出目录
    output_root = Path(args.output)
    output_img_dir, output_label_dir = prepare_output_dirs(output_root)
    print(f"结果将保存至：{output_root}")

    # 进度条与处理
    for img_path in tqdm(img_paths, desc="预测进度"):
        process_image(model, img_path, output_img_dir, output_label_dir, args)

    # 关闭显示窗口
    if args.show:
        cv2.destroyAllWindows()

    print("\n预测完成！输出结果结构：")
    print(f"标注图像：{len(list(output_img_dir.glob('*')))} 张")
    print(f"标签文件：{len(list(output_label_dir.glob('*')))} 个" if args.save_txt else "")


if __name__ == "__main__":
    args = parse_args()
    main(args)
