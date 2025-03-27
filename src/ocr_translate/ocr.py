import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from easyocr import Reader

from .utils import ensure_dir

#initializing EasyOCR once (German only)
READER = Reader(['de'], gpu=torch.cuda.is_available())

def ocr_image(image_path: Path, output_dir: Path) -> Path:
    """
    Run EasyOCR on a single image, save a semicolon-delimited CSV 
    (full_text, root_token, suffix, confidence) and annotated PNG,
    and return the path to the CSV file.
    """
    ensure_dir(output_dir)
    results = READER.readtext(str(image_path), detail=1)

    csv_path = output_dir / f"{image_path.stem}.csv"
    annotated_path = output_dir / f"{image_path.stem}_annotated.png"

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["full_text", "root_token", "suffix", "confidence"])
        for _, raw_text, conf in results:
            try:
                conf = float(conf)
            except (ValueError, TypeError):
                continue

            clean = " ".join(raw_text.split())
            parts = clean.split()
            if parts and parts[-1].startswith('-'):
                suffix = parts[-1]
                root = " ".join(parts[:-1])
            else:
                suffix = ""
                root = parts[-1] if parts else ""

            writer.writerow([clean, root, suffix, f"{conf:.2f}"])

    #draw bounding boxes + text onto the image
    img = cv2.imread(str(image_path))
    for bbox, text, _ in results:
        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        cv2.putText(
            img,
            text,
            tuple(map(int, bbox[0])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(annotated_path), img)

    return csv_path