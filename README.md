# A Python tool for preprocessing handheld scans of archival inventories and documents before OCR.

A Python script for preprocessing handheld scans of archival inventories and historical documents before Optical Character Recognition (OCR).
Designed specifically for images taken manually without a tripod, where skew, shadows, and overlapping pages are common.

## Features

- Automatic orientation correction using Tesseract OCR engine
- Text angle detection via Hough Transform or contour analysis
- Cropping of dark borders, shadows, and neighboring pages
- Histogram-based contrast enhancement
- Grayscale output optimized for OCR processing

## Typical Use Case

This tool is ideal for historical researchers, archivists, and genealogists working with:

- Manually photographed archival inventories (e.g., with a phone or compact camera)
- Historical documents captured in uneven lighting
- Scanned pages with inconsistent angles or overlapping sheets

By cleaning and straightening the images, it significantly improves OCR results (e.g., using Tesseract or Google Lens).

## Requirements

- Python 3.8 or higher
- [OpenCV](https://pypi.org/project/opencv-python/)
- [NumPy](https://pypi.org/project/numpy/)
- [pytesseract](https://pypi.org/project/pytesseract/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (must be installed separately)

Install Python dependencies:

```bash
pip install opencv-python numpy pytesseract
```

Make sure Tesseract OCR is installed and available in your system PATH.

## Usage

1. Place your input images (JPG or PNG format) into the input_images/ folder.
2. Run the script:

```bash
python preprocess.py
```

3. Processed images will be saved to the processed_images/ folder.

Each output image will be:

- Automatically rotated
- Skew-corrected
- Cropped to remove shadows, margins, and adjacent pages
- Enhanced for contrast
- Converted to grayscale and ready for OCR

## Folder Structure

```bash

ocr_inventores/
│
├── input_images/              # Folder for raw input images (JPG, PNG)
├── processed_images/          # Folder for processed images (deskewed, cropped, etc.)
├── output_text.txt            # Output file with OCR results
├── preprocess.py              # Main processing script
└── README.md                  # Project documentation
```

## Future Improvements (Roadmap)

- Command-line arguments for flexibility

- Batch processing with subfolder support

- Language-specific OCR presets

- Optional deskewing without relying on Tesseract OSD

- Improved robustness to diverse image conditions — currently, the script is tuned for pages captured manually under similar conditions (e.g., good lighting, minimal distortion, no strong shadows), but support for more variable conditions (e.g., shadows, skewed angles, inconsistent lighting) is planned


## License

MIT License — free to use, modify, and share.

## Author

Andrei Repin
Genealogical research and archival document processing specialist
📍 Based in Latvia | 🧾 Specializing in genealogical records from the Baltics