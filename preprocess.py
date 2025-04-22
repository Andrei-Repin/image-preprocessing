import cv2
import os
import pytesseract
import numpy as np
import re
from typing import Tuple, Optional, Union

# Folders for input and processed images
input_folder = os.path.join(os.getcwd(), "input_images")
processed_folder = os.path.join(os.getcwd(), "processed_images")
output_text_file = os.path.join(os.getcwd(), "recognized_text.txt")

os.makedirs(processed_folder, exist_ok=True)

# --- Processing Settings ---
settings = {
    'ENABLE_OCR': True,                      # False for image processing only
    'SKIP_PREPROCESSING': True,              # True to skip preprocessing (OCR only)
    'FORCE_GRAYSCALE': True,                 # True to force grayscale output
    'APPLY_GRAY_EARLY': True,                # True to convert to grayscale early
    'CROP': True,                            # False to disable cropping  
    'CROP_SIDE_PAGE': True,                  # True if adjacent page needs cropping
    'CROP_SIDE_DIRECTION': 'left',           # 'left' or 'right' - which side to crop
    'BINARIZATION_THRESHOLD': 180,           # Threshold for smart_crop (0-255)
    'SIDE_WHITE_RATIO_THRESHOLD': 0.9,       # White ratio threshold for side cropping
    'EDGE_CROP_BRIGHTNESS_THRESHOLD': 100,   # Threshold for edge cropping (0-255)
    'ROTATE': True,                          # False to disable rotation
    'ROTATION_ANGLE': 'auto',                # 'auto' or specific angle (0, 90, 180, 270)
    'FINE_ROTATION': True,                   # Fine-tune rotation angle
    'DOCUMENT_TYPE': 'typewritten',          # 'typewritten', 'printed', 'handwritten'
}

def recognize_ready_images(output_folder, settings):
    # Get list of image files
    image_files = [f for f in os.listdir(output_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    if not image_files:
        print("No images found for recognition. Place processed files in the output folder.")
        return

    if settings.get('SKIP_PREPROCESSING'):
        print("Processing already preprocessed images from output folder.")

    with open(output_text_file, "a", encoding="utf-8") as out_f:
        for filename in image_files:
            image_path = os.path.join(output_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading: {filename}")
                continue

            # Text recognition
            lang = 'rus+deu+lav'
            if settings['DOCUMENT_TYPE'] == 'handwritten':
                lang = 'rus+deu'
            recognized_text = get_ocr_text(image, lang)
            
            out_f.write(f"\n===== {filename} =====\n")
            out_f.write(recognized_text + "\n")

# Contrast enhancement with grayscale option
def enhance_contrast(image, force_grayscale=False):
    if force_grayscale:
        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image  # Already grayscale
        
        # Contrast stretching for grayscale
        min_val, max_val = np.percentile(gray, (1, 99))  # Ignore 1% darkest/brightest pixels

        if max_val - min_val < 1:  # Avoid division by zero
            return gray 
               
        enhanced = np.clip((gray - min_val) * (255.0 / (max_val - min_val)), 0, 255).astype(np.uint8)

        return enhanced  # Return grayscale image
    
    else:
        # Contrast stretching per channel: B, G, R
        channels = cv2.split(image)
        stretched_channels = []

        for channel in channels:
            min_val, max_val = np.percentile(channel, (1, 99))

            if max_val - min_val < 1:
                stretched_channels.append(channel)
                continue
                        
            stretched = np.clip((channel - min_val) * (255.0 / (max_val - min_val)), 0, 255).astype(np.uint8)
            stretched_channels.append(stretched)

        enhanced = cv2.merge(stretched_channels)
        return enhanced  # Return color image

# Detect rotation angle using Tesseract OSD
def detect_rotation(image) -> int:
    # Ensure image is grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Fallback if OSD fails
    try:
        osd = pytesseract.image_to_osd(image)
        angle = int(re.search(r'Rotate: (\d+)', osd).group(1))
        return angle
    except (pytesseract.TesseractError, AttributeError) as e:
        print(f"[!] Rotation detection failed: {e}")
        return 0  # or None

# Rotate image with canvas expansion
def rotate_image(image, angle: int):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Fine-tune text angle using Hough transform
def get_text_angle_by_hough(image) -> float:
    image_copy = image.copy()
    gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)

    if lines is None:
        return 0.0

    horiz_angles = []
    vert_angles = []

    for rho, theta in lines[:, 0]:
        angle_deg = np.degrees(theta)

        if 80 < angle_deg < 100:
            horiz_angles.append(angle_deg - 90)
        elif angle_deg < 10 or angle_deg > 170:
            corrected = angle_deg if angle_deg < 90 else angle_deg - 180
            vert_angles.append(corrected)

    if horiz_angles:
        return np.mean(horiz_angles)
    elif vert_angles:
        return np.mean(vert_angles)
    else:
        return 0.0

# Crop adjacent page fragment that's brighter than main page
def crop_black_side(binary_image, original_image,
                    side=settings['CROP_SIDE_DIRECTION'],
                    threshold_ratio=settings['SIDE_WHITE_RATIO_THRESHOLD']):
    h, w = binary_image.shape
    if side == 'left':
        for x in range(w):
            col = binary_image[:, x]
            white_ratio = np.mean(col == 255)
            if white_ratio > threshold_ratio:
                cropped = original_image[:, x:]
                return cropped
    elif side == 'right':
        for x in range(w - 1, -1, -1):
            col = binary_image[:, x]
            white_ratio = np.mean(col == 255)
            if white_ratio > threshold_ratio:
                cropped = original_image[:, :x]
                return cropped
    return original_image

# Main cropping function
def smart_crop(image, settings):
    # Initial grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    # Binarization for side cropping    
    _, binary = cv2.threshold(
        gray,
        settings['BINARIZATION_THRESHOLD'],
        255,
        cv2.THRESH_BINARY_INV
    )

    if settings['CROP_SIDE_PAGE']:  # Check side page cropping setting
        image = crop_black_side(
            binary,
            image,
            side=settings['CROP_SIDE_DIRECTION'],
            threshold_ratio=settings['SIDE_WHITE_RATIO_THRESHOLD']
        )

    return image

# Automatic dark edge cropping based on brightness analysis
def auto_crop(image, settings):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # Already grayscale

    threshold = settings['EDGE_CROP_BRIGHTNESS_THRESHOLD']

    # Find crop bounds where brightness exceeds threshold
    def find_crop_bounds(arr, threshold):
        start, end = 0, len(arr) - 1
        while start < end and np.mean(arr[start]) < threshold:
            start += 1
        while end > start and np.mean(arr[end]) < threshold:
            end -= 1
        return start, end
    
    # Find brightness-based boundaries
    top, bottom = find_crop_bounds(gray, threshold)
    left, right = find_crop_bounds(gray.T, threshold)

    # Perform cropping
    cropped = image[top:bottom+1, left:right+1]

    return cropped

def apply_rotation(image, settings) -> Tuple[np.ndarray, Optional[float]]:
    """
    Apply image rotation according to settings
    Returns rotated image and fine-tuned angle (if applied)
    """
    if not settings['ROTATE']:
        return image, None
    
    # Determine main rotation angle
    if isinstance(settings['ROTATION_ANGLE'], int):  # Specific angle
        angle = settings['ROTATION_ANGLE']
    elif settings['ROTATION_ANGLE'] == 'auto':
        # Convert to grayscale for orientation detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        angle = detect_rotation(gray)
    else:
        angle = 0  # Default no rotation
    
    # Apply main rotation
    rotated = rotate_image(image, angle)
    
    # Fine-tune text angle if needed
    fine_angle = None
    if settings.get('FINE_ROTATION', True) and settings['DOCUMENT_TYPE'] in ['typewritten', 'printed']:
        fine_angle = get_text_angle_by_hough(rotated)
        if abs(fine_angle) > 0.5:  # Ignore very small angles
            rotated = rotate_image(rotated, -fine_angle)
    
    return rotated, fine_angle

def process_image(image, settings) -> np.ndarray:
    """
    Main image processing function considering document type
    """
    # === 1. Image rotation (OSD + Hough)
    rotated, fine_angle = apply_rotation(image, settings)

    # === 2. Document-type specific preprocessing
    doc_type = settings['DOCUMENT_TYPE']

    if doc_type == 'typewritten':
        # Typewritten - needs cropping
        if settings.get('CROP_SIDE_PAGE', True):
            rotated = smart_crop(rotated, settings)
    elif doc_type == 'printed':
        # Printed pages - usually no special cropping
        pass
    elif doc_type == 'handwritten':
        print("[!] Handwritten: document type not yet supported.")
    else:
        print(f"[!] Unknown document type: {doc_type}")

    # === 3. Automatic background cropping (black borders etc.)
    if settings.get('CROP', True):
        rotated = auto_crop(rotated, settings)

    # === 4. Contrast enhancement - only after all cropping
    enhanced = enhance_contrast(rotated, force_grayscale=settings['FORCE_GRAYSCALE'])

    return enhanced

# Image processing
def preprocess_image(image_path, output_path, settings):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Loading error: {image_path}")
        return None

    # Process image through unified process_image function
    processed = process_image(image, settings)

    # Convert to grayscale if needed
    if settings['FORCE_GRAYSCALE'] or settings['APPLY_GRAY_EARLY']:
        if len(processed.shape) == 3 and processed.shape[2] == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    # Save result
    cv2.imwrite(output_path, processed, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return processed

# Archive inventory recognition
def get_ocr_text(image, lang='rus+deu+lav'):
    custom_config = r'--oem 3 --psm 6'
    ocr_data = pytesseract.image_to_data(
        image, lang=lang, config=custom_config, output_type=pytesseract.Output.DICT
    )

    lines = {}
    n = len(ocr_data['text'])

    for i in range(n):
        text = ocr_data['text'][i].strip()
        if text and int(ocr_data['conf'][i]) > 0:
            key = (ocr_data['block_num'][i], ocr_data['par_num'][i], ocr_data['line_num'][i])
            lines.setdefault(key, []).append(text)

    # Sort lines by position and combine words
    sorted_lines = [' '.join(lines[k]) for k in sorted(lines.keys()) if lines[k]]
    return '\n'.join(sorted_lines)

# Main processing function
def process_images_from_folder(input_folder, processed_folder, output_text_file, settings):
    with open(output_text_file, "w", encoding="utf-8") as out_f:
        for filename in sorted(os.listdir(input_folder)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(processed_folder, filename)

                # Image processing
                processed = preprocess_image(input_path, output_path, settings)

                if processed is not None:
                    out_f.write(f"\n===== {filename} =====\n")

                    if settings.get('ENABLE_OCR', True):  # OCR enabled by default
                        # Set OCR language based on document type
                        lang = 'rus+deu+lav'  # Base languages
                        if settings['DOCUMENT_TYPE'] == 'handwritten':
                            lang = 'rus+deu'  # Fewer languages for handwritten
                        text = get_ocr_text(processed, lang=lang)
                        out_f.write(text + "\n")
                    else:
                        out_f.write("[OCR disabled]\n")

# Entry point
if __name__ == "__main__":
    if settings.get('SKIP_PREPROCESSING'):
        recognize_ready_images(processed_folder, settings)
    else:
        process_images_from_folder(input_folder, processed_folder, output_text_file, settings)
    
    print(f"\nProcessing complete. Results saved to: {output_text_file}")