from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Set ATLDSD_ROOT to use an authorized local dataset outside the source tree.
# The default is a portable project-relative path, not a developer's disk path.
DATASET_ROOT = Path(
    os.environ.get("ATLDSD_ROOT", str(PROJECT_ROOT / "data" / "ATLDSD"))
).expanduser()
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

IMAGE_SIZE = 256
NUM_CLASSES = 6

# 标签像素值与数据集目录的对应关系。
CLASS_NAMES = {
    0: "背景",
    1: "健康叶片",
    2: "锈病",
    3: "交链孢叶斑病",
    4: "灰斑病",
    5: "褐斑病",
}

DISEASE_CLASSES = (2, 3, 4, 5)

# 用于叠加预测掩膜的 BGR 颜色。
CLASS_COLORS = {
    0: (0, 0, 0),
    1: (0, 180, 0),
    2: (0, 165, 255),
    3: (255, 0, 255),
    4: (200, 200, 200),
    5: (42, 42, 165),
}
