import cv2
import os
import pytesseract
import numpy as np

# Folders for original and processed images
input_folder = os.path.join(os.getcwd(), "input_images")
processed_folder = os.path.join(os.getcwd(), "processed_images")
output_text_file = os.path.join(os.getcwd(), "recognized_text.txt")

os.makedirs(processed_folder, exist_ok=True)

# Increasing image contrast using histogram stretching.
def enhance_contrast(image):   

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Contrast stretching
    min_val, max_val = np.percentile(gray, (1, 99))  # Ignore 1% of the darkest and brightest pixels
    enhanced = np.clip((gray - min_val) * (255.0 / (max_val - min_val)), 0, 255).astype(np.uint8)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# Determining the rotation angle using Tesseract OSD
def detect_rotation(image):
    osd = pytesseract.image_to_osd(image)
    try:
        angle = int([line for line in osd.split('\n') if 'Rotate:' in line][0].split(':')[-1])
    except Exception:
        angle = 0
    return angle


# Rotating the image with canvas expansion
def rotate_image(image, angle):
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

# Refining text line angle using Hough transform
def get_text_angle_by_hough(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)

    if lines is None:
        return 0

    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180) / np.pi
        if 80 < angle < 100:  # almost horizontal
            angles.append(angle - 90)

    if not angles:
        return 0

    return np.mean(angles)


# Cropping a fragment of the adjacent page that is lit brighter than the main page
def crop_black_side(binary_image, original_image, side='left', threshold_ratio=0.95):
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

def smart_crop(image):
    # Initial grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarization to remove the black side bar
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Removing the dark side part
    image = crop_black_side(binary, image, side='left')

    return image

# Automatic removal of dark edges based on brightness analysis.
def auto_crop(image, brightness_threshold=40):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Determine the average brightness of the edges
    top_mean = np.mean(gray[0:10, :])
    bottom_mean = np.mean(gray[-10:, :])
    left_mean = np.mean(gray[:, 0:10])
    right_mean = np.mean(gray[:, -10:])

    # Finding borders where brightness exceeds threshold
    def find_crop_bounds(arr, threshold):
        start, end = 0, len(arr) - 1
        while start < end and np.mean(arr[start]) < threshold:
            start += 1
        while end > start and np.mean(arr[end]) < threshold:
            end -= 1
        return start, end

    # Finding edges based on brightness
    top, bottom = find_crop_bounds(gray, brightness_threshold)
    left, right = find_crop_bounds(gray.T, brightness_threshold)

    # Cropping
    cropped = image[top:bottom+1, left:right+1]

    return cropped


def preprocess_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка загрузки: {image_path}")
        return None

    # 1. Orientation using pytesseract
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    osd_angle = detect_rotation(gray)
    # print(f"[OSD] Угол поворота: {osd_angle}°")  uncomment for debugging
    rotated = rotate_image(image, osd_angle)

    # 2. Refining line angle
    hough_angle = get_text_angle_by_hough(rotated)
    # print(f"[Hough] Уточнённый угол строки: {hough_angle:.2f}°")  uncomment for debugging

    # 3. Inverting for proper alignment
    finely_rotated = rotate_image(rotated, -hough_angle)

    # 4. Cropping the page (starting with adjacent page removal)
    cropped = smart_crop(finely_rotated)

    # 5. Cropping the remaining dark background
    cropped = auto_crop(cropped, brightness_threshold=40)

    # 6. Enhancing contrast after full cropping
    contrasted = enhance_contrast(cropped)

    # 7. Convert to grayscale and save
    final = cv2.cvtColor(contrasted, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_path, final, [cv2.IMWRITE_JPEG_QUALITY, 75])

    return output_path

# Main processing function
def process_images_from_folder(input_folder, processed_folder, output_text_file):
    with open(output_text_file, "w", encoding="utf-8") as out_f:
        for filename in sorted(os.listdir(input_folder)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(processed_folder, filename)
                processed = preprocess_image(input_path, output_path)

                # Logging only the file name, OCR temporarily disabled
                if processed:
                    out_f.write(f"\n===== {filename} =====\n[OCR skipped]\n")

# Execution
process_images_from_folder(input_folder, processed_folder, output_text_file)

print(f"\nProcessing completed. Result in: {output_text_file}")
