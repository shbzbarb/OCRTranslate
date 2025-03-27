from transformers import pipeline
import torch
import pandas as pd
from pathlib import Path
from .utils import ensure_dir
import re
from typing import Optional
import csv

DEVICE = 0 if torch.cuda.is_available() else -1
TRANSLATOR = pipeline(
    "translation_de_to_en",
    model="Helsinki-NLP/opus-mt-de-en",
    device=DEVICE,
    max_length=128
)

def normalize_translation(text: str) -> str:
    """Clean translated text while preserving meaningful structure"""
    #First handle special cases
    if text.startswith("/-nen"):
        return "/-nen"  # Special handling for this specific pattern
    
    #Handle repeated phrases
    text = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', text)
    
    #Cleaning up remaining artifacts
    text = re.sub(r'[{}♪]+', '', text)
    return text.strip()

def translate_csv(input_csv: Path, output_dir: Path) -> Path:
    ensure_dir(output_dir)
    df = pd.read_csv(input_csv, delimiter=';')
    
    #custom dictionary for special cases. This acts as a filter
    SPECIAL_TRANSLATIONS = {
        "das,": "the",
        "der,": "the",
        "die,": "the",
        "Sg:": "singular:",
        "PL.": "plural",
        "Sg.": "singular",
        "Sg-": "singular-",
        "/-nen": "/-nen",
        "~n": "~n"
    }

    def safe_translate(text: str) -> Optional[str]:
        try:
            #skipping empty or whitespace-only text
            if not text or str(text).isspace():
                return None
                
            #checking for special cases first
            if text in SPECIAL_TRANSLATIONS:
                return SPECIAL_TRANSLATIONS[text]
                
            #applying special translations for parts of text
            for de, en in SPECIAL_TRANSLATIONS.items():
                text = text.replace(de, en)
            
            #handling known problematic patterns. Here is one of the examples
            if "}laden" in text:
                return "download"
                
            #Translating the text
            result = TRANSLATOR(text)[0]['translation_text']
            
            return normalize_translation(result)
            
        except Exception as e:
            print(f"⚠️ Translation failed for '{text[:30]}...': {str(e)[:50]}")
            return None

    df['english_text'] = df['full_text'].apply(safe_translate)
    
    #for failed translations, use a simplified version
    for idx, row in df[df['english_text'].isna()].iterrows():
        simple_text = row['full_text'].split(';')[0].split(',')[0]
        df.at[idx, 'english_text'] = simple_text
    
    #cleaning remaining artifacts
    df['english_text'] = df['english_text'].apply(normalize_translation)
    
    #saving with all columns
    out_path = output_dir / f"{input_csv.stem}_translated.csv"
    df[['full_text', 'root_token', 'suffix', 'english_text']].to_csv(
        out_path, 
        index=False, 
        sep='\t', 
        encoding='utf-8-sig',
        quoting=csv.QUOTE_MINIMAL
    )
    return out_path