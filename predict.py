import argparse
import io
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import IMAGE_SIZE, NUM_CLASSES, OUTPUT_DIR
from src.dataset import read_image
from src.model import UNet
from src.visualize import make_visualization


def preprocess(image_bgr):
    resized = cv2.resize(image_bgr, (IMAGE_SIZE, IMAGE_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="预测苹果叶像素类别并估计病斑面积")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path, help="用户自行准备的六类 U-Net checkpoint")
    parser.add_argument("--no-tta", action="store_true", help="关闭水平翻转测试时增强（默认开启）")
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"找不到模型权重：{args.weights}。请通过 --weights 指定自行准备的权重文件")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(num_classes=NUM_CLASSES).to(device)
    # 与训练端一致：通过内存缓冲读取，兼容该 Windows PyTorch 环境。
    checkpoint = torch.load(io.BytesIO(args.weights.read_bytes()), map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    original = read_image(args.image, cv2.IMREAD_COLOR)
    with torch.no_grad():
        batch = preprocess(original).to(device)
        logits = model(batch)
        if not args.no_tta:
            flipped_logits = model(torch.flip(batch, dims=[3]))
            logits = (logits + torch.flip(flipped_logits, dims=[3])) / 2
        prediction = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    prediction = cv2.resize(prediction, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay, color_mask, report = make_visualization(original, prediction)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    cv2.imencode(".png", overlay)[1].tofile(str(OUTPUT_DIR / f"{stem}_result.png"))
    cv2.imencode(".png", color_mask)[1].tofile(str(OUTPUT_DIR / f"{stem}_mask.png"))
    print("预测完成")
    for key, value in report.items():
        print(f"{key}: {value}")
    print(f"结果图：{OUTPUT_DIR / f'{stem}_result.png'}")


if __name__ == "__main__":
    main()
