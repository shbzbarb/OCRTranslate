import pandas as pd

def clean_text_df(df: pd.DataFrame) -> pd.DataFrame:
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
    return df.dropna(subset=['confidence'])