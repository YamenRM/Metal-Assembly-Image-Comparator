import cv2
import numpy as np

def align_images(img_ref, img_target, max_features=2000, keep_ratio=0.2):
    """
    Aligns img_target to match the perspective of img_ref using SIFT and Homography.
    """
    # Convert to grayscale for feature detection
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
    
    # Detect SIFT features
    sift= cv2.SIFT_create(nfeatures=max_features) # type: ignore  , for ignoring type checking issues with OpenCV
    kp_ref, des_ref = sift.detectAndCompute(gray_ref, None)
    kp_target, des_target = sift.detectAndCompute(gray_target, None)
    
    if des_ref is None or des_target is None:
        raise ValueError("Could not detect enough distinct features for alignment.")
        
    # Match features using FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    matcher = cv2.FlannBasedMatcher(index_params, search_params) # type: ignore , for ignoring type checking issues with OpenCV
    
    matches = matcher.knnMatch(des_ref, des_target, k=2)
    
    # Filter matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m) 
            
    # Sort matches by quality and keep the best ones
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    keep_count = int(len(good_matches) * keep_ratio)
    good_matches = good_matches[:max(keep_count, 10)]
    
    if len(good_matches) < 4:
        raise ValueError("Not enough high-quality matches found to calculate perspective change.")
        
    # Extract coordinates of matched keypoints
    pts_ref = np.zeros((len(good_matches), 2), dtype=np.float32)
    pts_target = np.zeros((len(good_matches), 2), dtype=np.float32)
    
    for i, match in enumerate(good_matches):
        pts_ref[i, :] = kp_ref[match.queryIdx].pt
        pts_target[i, :] = kp_target[match.trainIdx].pt
        
    # Find Homography matrix using RANSAC
    H, mask = cv2.findHomography(pts_target, pts_ref, cv2.RANSAC, 5.0)
    
    if H is None:
        raise ValueError("Homography matrix estimation failed.")
    
    # Warp the target image to match the reference canvas orientation
    height, width, channels = img_ref.shape
    img_aligned = cv2.warpPerspective(img_target, H, (width, height))



    valid_mask = cv2.warpPerspective(
    np.full(gray_target.shape, 255, dtype=np.uint8), H, (width, height)
    )
    valid_mask = cv2.erode(valid_mask, np.ones((15, 15), np.uint8)) # shrink in from the warp edge

    return img_aligned, valid_mask

    

