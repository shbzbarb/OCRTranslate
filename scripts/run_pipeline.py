import sys
from pathlib import Path
import csv
import cv2
import numpy as np
import torch
import re
import pandas as pd
import easyocr

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ocr_translate.translate import translate_csv

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TRANSLATED_DIR = Path("data/translated")
FINAL_DIR = Path("data/final")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def clean_prefix(text):
    """Remove single-letter prefixes from text"""
    return re.sub(r'^[A-Z]\s+', '', str(text)).strip()

def create_final_csv(translated_csv: Path, final_dir: Path):
    """Create cleaned final CSV with enhanced filtering"""

    df = pd.read_csv(translated_csv, sep='\t')
    
    #Cleaning single-letter prefixes
    df['full_text'] = df['full_text'].apply(clean_prefix)
    df['english_text'] = df['english_text'].apply(clean_prefix)
    
    #filtering criteria
    df_clean = df[
        #excluding invalid entries
        ~df['full_text'].str.match(r'^[A-Z]$') &
        (df['full_text'].str.len() > 1) &
        df['full_text'].str.contains(r'[A-Za-zÄÖÜäöüß]') &
        
        #patterns matching with special characters
        df['full_text'].str.match(
            r'^[A-Za-zÄÖÜäöüß\'{].*[A-Za-zÄÖÜäöüß}\s\-\/,;:]?$'
        )
    ]
    
    #selecting and renaming the columns
    final_df = df_clean[['full_text', 'english_text']].rename(columns={
        'full_text': 'DE',
        'english_text': 'EN'
    })
    
    final_path = final_dir / f"{translated_csv.stem.replace('_translated', '_final')}.csv"
    final_df.to_csv(final_path, sep='\t', index=False, encoding='utf-8-sig')
    print(f"Final CSV saved → {final_path}")

def main():
    #Initializing EasyOCR reader
    device = 0 if torch.cuda.is_available() else -1
    reader = easyocr.Reader(["de"], gpu=(device >= 0))

    #Processing all images in raw directory
    for img_path in RAW_DIR.iterdir():
        if img_path.suffix.lower() not in VALID_EXTS:
            continue

        stem = img_path.stem
        print(f"\nProcessing {stem}...")

        out_folder = PROCESSED_DIR / stem
        out_folder.mkdir(parents=True, exist_ok=True)

        #OCR Processing
        results = reader.readtext(str(img_path), detail=1)
        csv_path = out_folder / f"{stem}.csv"
        annotated_path = out_folder / f"{stem}_annotated.png"

        #writing OCR results to CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["full_text", "root_token", "suffix", "confidence"])
            for _, raw_text, conf in results:
                try:
                    conf = float(conf)
                except:
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

        #creating annotated image
        img = cv2.imread(str(img_path))
        for bbox, text, _ in results:
            pts = np.array(bbox, dtype=np.int32).reshape((-1,1,2))
            cv2.polylines(img, [pts], True, (0,255,0), 2)
            cv2.putText(img, text, tuple(map(int,bbox[0])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imwrite(str(annotated_path), img)
        print(f"OCR saved → {csv_path} & {annotated_path}")

        #Translation
        translate_csv(csv_path, TRANSLATED_DIR)
        translated_csv_path = TRANSLATED_DIR / f"{stem}_translated.csv"
        print(f"Translation saved → {translated_csv_path}")

        #Final Cleanup
        create_final_csv(translated_csv_path, FINAL_DIR)

if __name__ == "__main__":
    for dir in [RAW_DIR, PROCESSED_DIR, TRANSLATED_DIR, FINAL_DIR]:
        dir.mkdir(parents=True, exist_ok=True)
    
    main()