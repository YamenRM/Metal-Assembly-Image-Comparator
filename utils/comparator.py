import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_differences(img_ref, img_aligned, valid_mask=None, ssim_thresh=0.4, min_area=40):
    """
    Compares two perfectly aligned images using SSIM and extracts change contours.
    """
    # Convert to grayscale
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to soften localized metallic highlights and glares
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_ref = clahe.apply(gray_ref)
    gray_aligned = clahe.apply(gray_aligned)
   
    # Apply Gaussian blur to reduce noise and minor surface scratches
    gray_ref = cv2.GaussianBlur(gray_ref, (3, 3), 0)
    gray_aligned = cv2.GaussianBlur(gray_aligned, (3, 3), 0)
    
    # Compute SSIM difference map
    score, diff = ssim(gray_ref, gray_aligned, full=True) # type: ignore
    
    # Convert diff map to 0-255 float, then invert it
    # 0 = perfect match, 255 = complete discrepancy
    diff = (1.0 - diff) * 255
    diff = diff.astype("uint8")
    
    # Threshold the inverted diff map
    # Pixels GREATER than the threshold are marked as anomalies
    _, thresh = cv2.threshold(diff, int(ssim_thresh * 255), 255, cv2.THRESH_BINARY)
   
    if valid_mask is not None:
        thresh = cv2.bitwise_and(thresh, thresh, mask=valid_mask)
    
    # Clean up small pixel noise using morphological opening/closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours of the anomalies 
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out tiny artifacts based on area scale
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    return valid_contours
