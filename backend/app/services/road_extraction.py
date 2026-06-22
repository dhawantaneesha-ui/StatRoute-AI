from pathlib import Path

import cv2
import numpy as np


def extract_road_mask(image_path: Path, output_path: Path) -> dict:
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Could not read image file.")

    resized = cv2.resize(image, (768, 768))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)

    road_mask = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), road_mask)

    road_pixels = int(np.count_nonzero(road_mask))
    total_pixels = int(road_mask.size)

    return {
        "mask_path": str(output_path),
        "image_width": 768,
        "image_height": 768,
        "road_pixels": road_pixels,
        "road_coverage_percent": round((road_pixels / total_pixels) * 100, 2),
    }