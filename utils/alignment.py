import cv2
import numpy as np


def _quad_angles(pts):
    """Returns the 4 interior angles (degrees) of a quadrilateral given corners in order."""
    angles = []
    n = len(pts)
    for i in range(n):
        p_prev = pts[i - 1]
        p_curr = pts[i]
        p_next = pts[(i + 1) % n]
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos_angle)))
    return angles


def validate_homography_geometry(
    H, target_shape, min_area_ratio=0.25, max_area_ratio=4.0, min_angle=30, max_angle=150
):
    """
    Checks whether a homography produces a sane, non-degenerate warp by tracing
    where the four corners of the target image actually land after warping.
    Returns (is_valid, reason) so callers can produce a clear error message.
    """
    h, w = target_shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    # 1. Convexity: a valid perspective warp of a rectangle stays convex.
    #    A folding/flipping warp produces a self-intersecting (non-convex) quad.
    hull = cv2.convexHull(warped_corners.astype(np.float32))
    if len(hull) < 4:
        return False, "Warp folds or collapses the image (non-convex result)."

    # 2. Area ratio: catches warps that blow up to near-infinity or collapse to a sliver.
    orig_area = w * h
    warped_area = cv2.contourArea(warped_corners.astype(np.float32))
    area_ratio = warped_area / (orig_area + 1e-8)
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return False, (
            f"Warped image area changed by {area_ratio:.2f}x, which is outside the "
            f"acceptable range ({min_area_ratio}x-{max_area_ratio}x). This usually means "
            "the two photos were taken from very different distances or angles."
        )

    # 3. Corner angles: a rectangle warped by a *reasonable* perspective shift keeps
    #    roughly quadrilateral angles. Extreme angles (near 0 or 180) mean one edge
    #    of the image is being crushed or stretched toward a vanishing point.
    angles = _quad_angles(warped_corners)
    if min(angles) < min_angle or max(angles) > max_angle:
        return False, (
            f"Warp produces an extreme quadrilateral (angles range {min(angles):.0f}-"
            f"{max(angles):.0f} degrees). This indicates the viewpoint change between "
            "photos is too large for a reliable perspective alignment."
        )

    return True, None


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
        raise ValueError("Perspective alignment failed: Plane matching calculation failed.")
    
    # Geometrical Validation Check: Detect extreme view angle distortion.
    # Two layers of defense:
    #   1. Affine determinant check (fast, catches gross flips in the linear part of H)
    #   2. Full quad-corner check (catches perspective-term distortion that the
    #      determinant check alone misses -- e.g. one image edge crushed toward
    #      a vanishing point, which happens when a 3D object is re-photographed
    #      from a substantially different viewpoint)
    det = H[0,0]*H[1,1] - H[0,1]*H[1,0]
    inlier_ratio = np.sum(mask) / len(good_matches) if len(good_matches) > 0 else 0 # type: ignore  

    if det <= 0.1 or inlier_ratio < 0.35:
        raise ValueError(
            "Incompatible View Angles Detected! Please ensure both photographs are captured from "
            "the same camera position and orientation relative to the metal structure."
        )

    is_valid, reason = validate_homography_geometry(H, gray_target.shape)
    if not is_valid:
        raise ValueError(
            f"Incompatible View Angles Detected! {reason} Please ensure both photographs are "
            "captured from the same camera position and orientation relative to the metal structure."
        )
    
    # Warp the target image to match the reference canvas orientation
    height, width, channels = img_ref.shape
    img_aligned = cv2.warpPerspective(img_target, H, (width, height))



    valid_mask = cv2.warpPerspective(
    np.full(gray_target.shape, 255, dtype=np.uint8), H, (width, height)
    )
    valid_mask = cv2.erode(valid_mask, np.ones((15, 15), np.uint8)) 

    return img_aligned, valid_mask