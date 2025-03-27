# EasyOCR and Translation ML App
Learning a new language can be challenging, particularly when it comes to building vocabulary. This project aims to streamline that process by developing an application using a Python-based machine learning pipeline. The application leverages Optical Character Recognition (OCR) to extract German text from images—such as photographs of textbook pages—and processes the extracted content through a series of natural language processing steps. This tool is designed to assist language learners in capturing and organizing new vocabulary efficiently from printed materials. The core functionalities includes: 
- Extracting German text from images using EasyOCR [link](https://www.jaided.ai/easyocr/install/)
- Translating the cleaned German text into English using a Helsinki-NLP OpusMT German-to-English Translation model [link](https://huggingface.co/Helsinki-NLP/opus-mt-de-en)
- Cleaning and preprocessing the extracted text 
- Comprehensive error handling & noise filtering
- Storing the final structured data in CSV format for easy reference and vocabulary building

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