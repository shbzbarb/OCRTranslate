from pathlib import Path
import pandas as pd
from .ocr import ocr_image
from .preprocess import clean_text_df
from .translate import translate_csv
from .utils import ensure_dir

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def run(raw_dir: Path, processed_dir: Path, translated_dir: Path):
    ensure_dir(processed_dir)
    ensure_dir(translated_dir)

    for img in raw_dir.iterdir():
        if img.is_file() and img.suffix.lower() in VALID_EXTS:
            out_folder = processed_dir / img.stem
            csv_path = ocr_image(img, out_folder)

            df = clean_text_df(pd.read_csv(csv_path))
            df.to_csv(csv_path, index=False)

            translate_csv(csv_path, translated_dir)