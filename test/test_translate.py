import sys
from pathlib import Path

# Add the src folder to the module search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from ocr_translate.translate import translate_csv

def test_translate_creates_translation(tmp_path):
    df = pd.DataFrame({"text": ["Dies ist ein Test."], "confidence": ["0.99"]})
    input_csv = tmp_path / "test.csv"
    df.to_csv(input_csv, index=False)
    out_dir = tmp_path / "translated"
    out_path = translate_csv(input_csv, out_dir)
    assert out_path.exists()
    df_out = pd.read_csv(out_path)
    assert "english_text" in df_out.columns
    assert "this is a test" in df_out.iloc[0]["english_text"].lower()