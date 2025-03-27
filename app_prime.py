import os
import torch
import streamlit
from io import BytesIO
import zipfile
import time

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
torch.classes.__path__ = []
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import streamlit as st
import torch
import easyocr
import cv2
import numpy as np
import pandas as pd
import re
from pathlib import Path
import sys
import csv

#initializing session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'stop_processing' not in st.session_state:
    st.session_state.stop_processing = False

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

#useful running it locally. Follows same pipeline as run_pipeline.py
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TRANSLATED_DIR = Path("data/translated")
FINAL_DIR = Path("data/final")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

for dir in [RAW_DIR, PROCESSED_DIR, TRANSLATED_DIR, FINAL_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

def clean_prefix(text):
    return re.sub(r'^[A-Z]\s+', '', str(text)).strip()

def create_final_csv(translated_csv: Path, final_dir: Path):
    df = pd.read_csv(translated_csv, sep='\t')
    df['full_text'] = df['full_text'].apply(clean_prefix)
    df['english_text'] = df['english_text'].apply(clean_prefix)
    
    df_clean = df[
        ~df['full_text'].str.match(r'^[A-Z]$') &
        (df['full_text'].str.len() > 1) &
        df['full_text'].str.contains(r'[A-Za-zÄÖÜäöüß]') &
        df['full_text'].str.match(r'^[A-Za-zÄÖÜäöüß\'{].*[A-Za-zÄÖÜäöüß}\s\-\/,;:]?$')
    ]
    
    final_df = df_clean[['full_text', 'english_text']].rename(columns={
        'full_text': 'DE',
        'english_text': 'EN'
    })
    
    final_path = final_dir / f"{translated_csv.stem.replace('_translated', '_final')}.csv"
    final_df.to_csv(final_path, sep='\t', index=False, encoding='utf-8-sig')
    return final_path

def process_single_image(image_path: Path):
    """Process single image with full pipeline"""
    device = 0 if torch.cuda.is_available() else -1
    reader = easyocr.Reader(["de"], gpu=(device >= 0))
    
    stem = image_path.stem
    out_folder = PROCESSED_DIR / stem
    out_folder.mkdir(parents=True, exist_ok=True)

    #OCR Processing
    results = reader.readtext(str(image_path), detail=1)
    csv_path = out_folder / f"{stem}.csv"
    annotated_path = out_folder / f"{stem}_annotated.png"

    #Writing CSV
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

    #Creating annotated image
    img = cv2.imread(str(image_path))
    for bbox, text, _ in results:
        pts = np.array(bbox, dtype=np.int32).reshape((-1,1,2))
        cv2.polylines(img, [pts], True, (0,255,0), 2)
        cv2.putText(img, text, tuple(map(int,bbox[0])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imwrite(str(annotated_path), img)

    #translation logic
    from ocr_translate.translate import translate_csv
    translate_csv(csv_path, TRANSLATED_DIR)
    translated_csv = TRANSLATED_DIR / f"{stem}_translated.csv"

    #final CSV with DE and EN text
    return create_final_csv(translated_csv, FINAL_DIR), annotated_path

def process_batch(image_paths):
    """Process multiple images with progress tracking"""
    final_files = []
    annotated_images = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, img_path in enumerate(image_paths):
        if st.session_state.stop_processing:
            break
            
        try:
            progress = (idx + 1) / len(image_paths)
            progress_bar.progress(progress)
            status_text.text(f"Processing {idx+1}/{len(image_paths)}: {img_path.name}")
            
            final_csv, annotated_path = process_single_image(img_path)
            final_files.append(final_csv)
            annotated_images.append(annotated_path)
            
        except Exception as e:
            st.error(f"Failed to process {img_path.name}: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    return final_files, annotated_images

#Streamlit UI Title
st.title("OCR Translation")

uploaded_files = st.file_uploader(
    "Upload Images", 
    type=list(VALID_EXTS),
    accept_multiple_files=True
)

if uploaded_files:
    
    image_paths = []
    for uploaded_file in uploaded_files:
        img_path = RAW_DIR / uploaded_file.name
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_paths.append(img_path)
    
    st.session_state.processing = True
    st.session_state.stop_processing = False
    
    #displaying stop button for large batches
    if len(image_paths) > 20:
        stop_button = st.button("Stop Processing")
        if stop_button:
            st.session_state.stop_processing = True
    
    final_files, annotated_images = process_batch(image_paths)
    
    #displaying results (OCR annotated images)
    if annotated_images:
        st.subheader("Processed Images")
        for img_path in annotated_images:
            st.image(str(img_path), use_container_width=True)
    
    #handling downloads
    if final_files:
        if len(final_files) > 1:
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zipf:
                for csv_file in final_files:
                    zipf.write(csv_file, arcname=csv_file.name)
            
            st.download_button(
                "Download All Results (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="translations.zip",
                mime="application/zip"
            )
        else:
            with open(final_files[0], "rb") as f:
                st.download_button(
                    "Download Final CSV",
                    data=f,
                    file_name=final_files[0].name,
                    mime="text/csv"
                )
    
    #reset the processing state
    st.session_state.processing = False
    st.session_state.stop_processing = False