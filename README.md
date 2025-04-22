# A Python tool for preprocessing and OCR processing of handheld scans of archival inventories and documents.

A Python script for preprocessing and OCR processing of handheld scans of archival inventories and historical documents.
Designed specifically for images taken manually without a tripod, where skew, shadows, and overlapping pages are common.

## Features

- **Centralized processing settings** via a `settings` dictionary for easier configuration (e.g., toggling OCR, cropping, rotation, binarization, etc.)
- **Advanced cropping and rotation logic**:
  - Fine control over side cropping with `CROP_SIDE_PAGE`, `CROP_SIDE_DIRECTION`, and `SIDE_WHITE_RATIO_THRESHOLD`.
  - Optional fine rotation based on text angle detection using Hough lines (`FINE_ROTATION`, `get_text_angle_by_hough()`).
- **Improved image preprocessing**:
  - Added `FORCE_GRAYSCALE` option to enforce grayscale mode regardless of source.
  - Enhanced contrast adjustment, which ignores extreme top and bottom pixel values.
- **OCR Integration**: Tesseract OCR with explicit language specification (`rus+deu+lat`), now recognizing actual text.
- **Document type system** for applying different processing strategies based on document type (e.g., typeset vs handwritten).
  - Currently supporting only typeset documents, with handwritten document recognition planned for future versions.
- **Decoupled processing pipeline**: Image preprocessing can now be run separately from OCR.
- **Improved error handling**: Validation added for unreadable images and invalid angle extraction.
- **Code readability improvements**: Clearer variable and function names, with additional inline comments for better maintainability.

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
- Converted to grayscale and optimized for OCR processing
- Ready for OCR with integrated Tesseract OCR engine for text recognition

## Folder Structure

```bash

ocr_inventores/
│
├── input_images/              # Folder for raw input images (JPG, PNG)
├── processed_images/          # Folder for processed images (deskewed, cropped, etc.)
├── output_text.txt            # Output file with OCR results
├── preprocess.py              # Main processing script
├── CHANGELOG.md               # Change log with updates and improvements
└── README.md                  # Project documentation

```

## Future Improvements (Roadmap)

- **Improved robustness to diverse image conditions**: Currently, the script is tuned for pages captured manually under similar conditions (e.g., good lighting, minimal distortion, no strong shadows), but support for more variable conditions (e.g., shadows, skewed angles, inconsistent lighting) is planned.
- **Handwritten document processing**: Adding support for preprocessing handwritten documents, including alignment and preparation for OCR recognition.
- **Post-processing of printed OCR text**: Enhancements for detecting and correcting common OCR recognition errors, such as fixing misrecognized characters, removing line breaks, and preparing the output text for further use.
- **Handwritten text recognition**: Integration of OCR capabilities for handwritten documents, enabling recognition of text in a variety of handwriting styles.



## License

MIT License — free to use, modify, and share.

## Author

Andrei Repin
Genealogical research and archival document processing specialist
📍 Based in Latvia | 🧾 Specializing in genealogical records from the Baltics