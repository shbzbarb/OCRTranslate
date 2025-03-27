from pathlib import Path

RAW_DIR = Path("data/raw")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def get_input_images():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return [p for p in RAW_DIR.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]

if __name__ == "__main__":
    images = get_input_images()
    print(f"Found {len(images)} image(s) in {RAW_DIR}:")
    for img in images:
        print(" •", img.name)