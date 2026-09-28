"""Browser upload and inference API for a user-supplied ATLDSD U-Net checkpoint."""
from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import pickle
import threading
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from src.config import CLASS_COLORS, CLASS_NAMES, DISEASE_CLASSES, IMAGE_SIZE, NUM_CLASSES
from src.model import UNet
from src.visualize import colorize_mask

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PROJECT_ROOT / "checkpoints" / "best_unet.pth"
WEIGHTS_PATH = Path(os.environ.get("ATLDSD_WEIGHTS", str(DEFAULT_WEIGHTS))).expanduser()
EXPECTED_WEIGHTS_SHA256 = os.environ.get("ATLDSD_EXPECTED_SHA256", "").strip().lower()
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(max(1, int(os.environ.get("ATLDSD_CPU_THREADS", "4"))))
logger = logging.getLogger("uvicorn.error")
_model: UNet | None = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()
_checkpoint_sha256: str | None = None


def get_model() -> UNet:
    """Load once, verify the selected checkpoint, and share it across requests."""
    global _model, _checkpoint_sha256
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        if not WEIGHTS_PATH.is_file():
            raise FileNotFoundError(f"模型权重不存在：{WEIGHTS_PATH}")
        if len(EXPECTED_WEIGHTS_SHA256) != 64 or any(
            character not in "0123456789abcdef" for character in EXPECTED_WEIGHTS_SHA256
        ):
            raise ValueError("请设置 ATLDSD_EXPECTED_SHA256 为所选权重的实际 SHA256。")
        contents = WEIGHTS_PATH.read_bytes()
        actual_sha = hashlib.sha256(contents).hexdigest()
        if actual_sha != EXPECTED_WEIGHTS_SHA256:
            raise ValueError(f"权重 SHA256 不匹配：{actual_sha}；期望 {EXPECTED_WEIGHTS_SHA256}")
        checkpoint = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
        model = UNet(num_classes=NUM_CLASSES)
        model.load_state_dict(checkpoint["model"])
        model.to(DEVICE).eval()
        _checkpoint_sha256 = actual_sha
        _model = model
        logger.info("ATLDSD 模型已加载，device=%s，weights=%s", DEVICE, WEIGHTS_PATH)
    return _model


@asynccontextmanager
async def lifespan(application: FastAPI):
    try:
        await run_in_threadpool(get_model)
    except (OSError, KeyError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        # Keep the page reachable so the UI can explain a missing checkpoint.
        logger.error("模型初始化失败，请检查 ATLDSD_WEIGHTS：%s", exc)
    yield


app = FastAPI(title="叶析 · 苹果叶病害分割", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "demo" / "static"), name="static")


def image_to_data_url(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("无法编码结果图片")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def decode_image(contents: bytes) -> np.ndarray:
    """Check dimensions before pixel decoding; apply the camera's EXIF rotation."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(contents)) as image:
                if image.format not in ALLOWED_FORMATS:
                    raise HTTPException(415, "请上传 JPG、PNG、WEBP 或 BMP 图片。")
                width, height = image.size
                if width * height > MAX_IMAGE_PIXELS or max(width, height) > 8000:
                    raise HTTPException(413, "图片尺寸过大：最大 2,000 万像素，单边不超过 8,000 像素。")
                if getattr(image, "is_animated", False):
                    raise HTTPException(415, "请使用静态图片，暂不支持动画图片。")
                oriented = ImageOps.exif_transpose(image)
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                rgb = np.asarray(Image.alpha_composite(background, rgba).convert("RGB"))
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(413, "图片尺寸过大，请缩小后再上传。") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(400, "无法读取这张图片，请检查文件是否损坏。") from exc


def preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(image_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0)


def predict_image(image_bgr: np.ndarray) -> dict:
    model = get_model()
    batch = preprocess(image_bgr).to(DEVICE)
    with _inference_lock, torch.inference_mode():
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        logits = model(batch)
        flipped_logits = model(torch.flip(batch, dims=[3]))
        logits = (logits + torch.flip(flipped_logits, dims=[3])) / 2
        small_mask = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        inference_ms = round((time.perf_counter() - started) * 1000, 1)
    height, width = image_bgr.shape[:2]
    mask = cv2.resize(small_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    color_mask = colorize_mask(mask)
    overlay = image_bgr.copy()
    foreground = mask > 0
    blended = cv2.addWeighted(image_bgr, 0.62, color_mask, 0.38, 0)
    overlay[foreground] = blended[foreground]
    counts = np.bincount(mask.ravel(), minlength=NUM_CLASSES)
    leaf_pixels = int(counts[1:].sum())
    disease_pixels = int(counts[list(DISEASE_CLASSES)].sum())
    dominant_id = max(DISEASE_CLASSES, key=lambda class_id: counts[class_id])
    if not leaf_pixels:
        status, label = "no_leaf", "未识别到叶片"
    elif not disease_pixels:
        status, label = "no_lesion", "未检出病斑"
    else:
        status, label = "lesion_detected", CLASS_NAMES[dominant_id]
    areas = [
        {
            "class_id": class_id,
            "name": CLASS_NAMES[class_id],
            "color": "#" + "".join(f"{v:02x}" for v in reversed(CLASS_COLORS[class_id])),
            "pixels": int(counts[class_id]),
            "percent_of_predicted_leaf": round(float(counts[class_id]) / leaf_pixels * 100, 3) if leaf_pixels else 0.0,
        }
        for class_id in DISEASE_CLASSES
    ]
    return {
        "overlay": image_to_data_url(overlay),
        "mask": image_to_data_url(color_mask),
        "label_mask": image_to_data_url(mask),
        "summary": {
            "status": status,
            "most_predicted_disease": label,
            "total_disease_percent": round(disease_pixels / leaf_pixels * 100, 2) if leaf_pixels else 0.0,
            "dominant_disease_percent": round(float(counts[dominant_id]) / leaf_pixels * 100, 2) if disease_pixels else 0.0,
            "predicted_leaf_pixels": leaf_pixels,
            "disease_pixels": disease_pixels,
            "image_width": width,
            "image_height": height,
            "inference_ms": inference_ms,
        },
        "disease_areas": areas,
        "model": {"name": "ATLDSD U-Net", "input_size": IMAGE_SIZE, "tta": "horizontal_flip", "device": str(DEVICE), "weights_sha256": _checkpoint_sha256},
        "notice": "病斑比例是预测病害像素占预测叶片像素的面积比例，并非分类置信度。",
    }


@app.get("/")
def home():
    return FileResponse(PROJECT_ROOT / "demo" / "static" / "index.html")


@app.get("/api/health")
def health():
    try:
        get_model()
        status, message = "ready", "模型已就绪"
    except (OSError, KeyError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError):
        status, message = "unavailable", "模型未就绪，请按启动说明检查权重和后端终端提示。"
    return {
        "status": status,
        "message": message,
        "device": str(DEVICE),
        "model_name": "ATLDSD U-Net",
        "input_size": IMAGE_SIZE,
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "classes": [
            {"id": class_id, "name": name, "color": "#" + "".join(f"{v:02x}" for v in reversed(CLASS_COLORS[class_id]))}
            for class_id, name in CLASS_NAMES.items()
        ],
    }


@app.post("/api/predict")
async def predict_upload(file: UploadFile = File(...)):
    started = time.perf_counter()
    try:
        if Path(file.filename or "").suffix.lower() not in ALLOWED_SUFFIXES:
            raise HTTPException(415, "请上传 JPG、PNG、WEBP 或 BMP 图片。")
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    if not contents:
        raise HTTPException(400, "上传文件为空，请重新选择图片。")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "图片不能超过 8 MB，请压缩后再上传。")
    image = await run_in_threadpool(decode_image, contents)
    try:
        result = await run_in_threadpool(predict_image, image)
    except (OSError, KeyError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        logger.exception("模型推理失败")
        raise HTTPException(503, "模型暂时无法推理，请检查权重配置或查看后端终端提示。") from exc
    result["request_id"] = uuid4().hex
    result["processing_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result
