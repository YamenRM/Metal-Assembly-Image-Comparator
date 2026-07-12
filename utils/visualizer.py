import cv2
import numpy as np

def generate_comparison_view(img_ref, img_aligned, contours):
    """
    Highlights structural discrepancies on the aligned image and merges it side-by-side with reference image.
    """
    # 1. Enforce BGR Color Space: Convert grayscale to color if necessary
    if len(img_ref.shape) == 2:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR)
    if len(img_aligned.shape) == 2:
        img_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_GRAY2BGR)

    # Create a clean copy of the aligned target image to paint on
    output_target = img_aligned.copy()
    
    # 2. Draw clean bounding rectangles around detected differences
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw a semi-transparent red overlay accent inside the box
        overlay = output_target.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.25, output_target, 0.75, 0, output_target)
        
        # Draw a sharp neon orange bounding box border
        cv2.rectangle(output_target, (x, y), (x + w, y + h), (0, 140, 255), 2)
        
    # 3. Enforce Dimension Matching: Match height to prevent np.hstack crashes
    if img_ref.shape[0] != output_target.shape[0]:
        ref_h, ref_w = img_ref.shape[:2]
        # Scale target height to match reference while preserving its own width aspect
        scale_ratio = ref_h / output_target.shape[0]
        new_w = int(output_target.shape[1] * scale_ratio)
        output_target = cv2.resize(output_target, (new_w, ref_h), interpolation=cv2.INTER_AREA)
        
    # 4. Stack the original reference (left) and the highlighted target (right) side by side
    composite_view = np.hstack((img_ref, output_target))
    
    return composite_view
