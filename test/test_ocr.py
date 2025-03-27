import sys
from pathlib import Path

# Add the src folder to the module search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from ocr_translate.ocr import ocr_image

RAW_DIR = Path("data/raw")

def test_ocr_creates_csv_and_image(tmp_path):
    raw_images = [p for p in RAW_DIR.iterdir() if p.is_file()]
    assert raw_images, "No images in data/raw"
    img = raw_images[0]
    out_dir = tmp_path / img.stem
    csv_path = ocr_image(img, out_dir)
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert {"text", "confidence"}.issubset(df.columns)
    annotated = out_dir / f"{img.stem}_annotated.png"
    assert annotated.exists()