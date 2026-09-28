import cv2
import numpy as np

from .config import CLASS_COLORS, CLASS_NAMES, DISEASE_CLASSES


def disease_report(mask: np.ndarray):
    """Report predicted pixel areas; these ratios are not disease severity grades."""
    leaf_area = int(np.count_nonzero(mask > 0))
    disease_areas = {class_id: int(np.count_nonzero(mask == class_id)) for class_id in DISEASE_CLASSES}
    disease_id = max(disease_areas, key=disease_areas.get)
    disease_area = disease_areas[disease_id]
    total_disease_area = sum(disease_areas.values())
    if leaf_area == 0:
        disease_id, disease_name, status = 0, "未识别到叶片", "no_leaf"
    elif total_disease_area == 0:
        disease_id, disease_name, status = 1, "未检出病斑", "no_lesion"
    else:
        disease_name, status = CLASS_NAMES[disease_id], "lesion_detected"
    return {
        "status": status,
        "disease_id": disease_id,
        "disease_name": disease_name,
        "leaf_area": leaf_area,
        "disease_area": disease_area,
        "dominant_disease_percent": 100 * disease_area / leaf_area if leaf_area else 0.0,
        "total_disease_area": total_disease_area,
        "total_disease_percent": 100 * total_disease_area / leaf_area if leaf_area else 0.0,
    }


def colorize_mask(mask: np.ndarray):
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, bgr in CLASS_COLORS.items():
        color[mask == class_id] = bgr
    return color


def make_visualization(original_bgr: np.ndarray, mask: np.ndarray):
    color_mask = colorize_mask(mask)
    overlay = original_bgr.copy()
    foreground = mask > 0
    blended = cv2.addWeighted(original_bgr, 0.62, color_mask, 0.38, 0)
    overlay[foreground] = blended[foreground]
    return overlay, color_mask, disease_report(mask)
