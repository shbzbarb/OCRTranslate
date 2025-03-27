import sys
from pathlib import Path

# Add the src folder to the module search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from ocr_translate.pipeline import run

RAW_DIR = Path("data/raw")

def test_pipeline_end_to_end(tmp_path):
    processed = tmp_path / "processed"
    translated = tmp_path / "translated"
    run(RAW_DIR, processed, translated)

    for img in RAW_DIR.iterdir():
        if not img.is_file():
            continue
        stem = img.stem
        folder = processed / stem
        assert folder.exists()
        assert (folder / f"{stem}.csv").exists()
        assert (folder / f"{stem}_annotated.png").exists()
        trans = translated / f"{stem}_translated.csv"
        assert trans.exists()
        df = pd.read_csv(trans)
        assert "english_text" in df.columns