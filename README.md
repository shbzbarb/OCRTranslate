# EasyOCR and Translation ML App
Learning a new language can be challenging, particularly when it comes to building vocabulary. This project aims to streamline that process by developing an application using a Python-based machine learning pipeline. The application leverages Optical Character Recognition (OCR) to extract German text from images—such as photographs of textbook pages—and processes the extracted content through a series of natural language processing steps. This tool is designed to assist language learners in capturing and organizing new vocabulary efficiently from printed materials. The core functionalities includes: 
- Extracting German text from images using EasyOCR
- Translating the cleaned German text into English using a Helsinki-NLP OpusMT German-to-English Translation model
- Cleaning and preprocessing the extracted text 
- Comprehensive error handling & noise filtering
- Storing the final structured data in CSV format for easy reference and vocabulary building

## Demo
### Coming Soon...


## Project Structure
```
ExtractTranslate/
├── data/
│   ├── final/                      # Final German→English CSVs
│   ├── processed/                  # OCR csv files including annotated images showing OCR recognized texts
│   ├── raw/                        # Place your input German-text images here
│   └── translated/                 # Intermediate translated files
├── scripts/
│   ├── input_images.py             # validates input images
│   └── run_pipeline.py             # Executes full OCR→Translation pipeline
├── src/
│   └── ocr_translate/
│       ├── __init__.py
│       ├── ocr.py                  # EasyOCR OCR extraction
│       ├── pipeline.py             # Combines OCR & translation steps
│       ├── preprocess.py           # Text cleanup utilities
│       ├── translate.py            # German→English translation
│       └── utils.py                # Utility functions (file handling)
├── test/                           # Test suite (pytest)
│   ├── test_ocr.py
│   ├── test_translate.py
│   └── test_pipeline.py
└── environment.yml                 # Conda environment file
└── streamlit_app.py                # Code for streamlit app
```

## Environment Setup
### Creating and activating the Conda environment using .yml
Create Environment
```sh
 conda env create -f environment.yml
```

Activate Environment
```sh
 conda activate OCR_Translate
```

## Pipeline Workflow
### Input Images
- Stored in: ./data/raw
- Accepted formats: .jpg, .jpeg, .png, .bmp, .tiff

### OCR Extraction (EasyOCR)
- Extracts text, splits suffixes (-en, -s) clearly
- Annotates bounding boxes in images
- Saves structured CSVs (full_text;root_token;suffix;confidence) with confidence scores

### Text Preprocessing & Cleanup
- Cleans redundant whitespace & special characters.
- Filters out low-confidence or garbage OCR rows.

### Translation (German → English)
- Uses Helsinki-NLP OpusMT model
- Cleans translated texts: removes repeated words, braces, noise ({}, ♪, -)
- Removes meaningless translations or repeated tokens

### Output Data
- Final CSVs (full_text;english_text) saved in ./data/final
- CSV encoding (utf-8-sig) & delimiters (;)

## Modules & Scripts
### OCR Module (ocr.py)
- Performs OCR extraction and CSV annotation
- CSV Columns:
- full_text: Original extracted phrase
- root_token: Primary word(s) without suffix
- suffix: Extracted suffix (if any, e.g., -en)
- confidence: OCR confidence (0.00 to 1.00)

### Translation Module (translate.py)
- Translates extracted German text into English
- Pre-translation filters:
- Valid German words (letters, spaces, hyphens)
- Post-translation filters:
- Eliminates short translations and repeated-word noise
- Removes symbols (♪, braces {}, hyphens -)

### Pipeline Runner (run_pipeline.py)
- High-level execution script combining OCR and translation steps
- Saves output CSVs and annotated images clearly structured

### Utilities (utils.py)
- Common helper methods for file handling & directory creation.

### Tests
- Uses pytest framework
- Automated testing of OCR, translation, and full pipeline execution


## Execution and Usage
### Running the complete pipeline
```sh
python scripts/run_pipeline.py
```
- Verify the ```./data/processed```, ```./data/translated```, and ```./data/final``` folders for results

### Running Tests
```sh
pytest -q
```

## Known Issues and Solutions
### CSV Parsing Issues in LibreOffice Calc (Ubuntu)
- Problem: Columns not splitting correctly
- Solution: Use semicolon (```;```) delimiter with ```utf-8-sig``` encoding (includes BOM)

### Translation Noise (```load load load```, special characters)
- Problem: Repeated words, OCR errors in translations
- Solution: Regex-based filtering (```normalize_translation```). Drop translations with low unique-word ratio.


## Conclusion & Future Improvements
### Conclusion
The pipeline robustly processes German text images, extracts clean textual data, translates it accurately, and generates structured data compatible with spreadsheet tools

### Possible Future Enhancements
- More Languages: Extend support beyond German & English
- Advanced Cleanup: Further improve post-translation cleanup using NLP techniques


## References
### OCR (EasyOCR)
- **GitHub Repository**: [https://github.com/JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)
- **Documentation**: [https://www.jaided.ai/easyocr/](https://www.jaided.ai/easyocr/)

### Translation (Helsinki-NLP MarianMT)
- **Hugging Face Model Hub**: [https://huggingface.co/Helsinki-NLP](https://huggingface.co/Helsinki-NLP)
- **MarianMT Paper**: Jörg Tiedemann, Santhosh Thottingal (2020). [OPUS-MT — Building open translation services for the World](https://www.aclweb.org/anthology/2020.eamt-1.61.pdf). *Proceedings of the 22nd Annual Conference of the European Association for Machine Translation (EAMT)*

