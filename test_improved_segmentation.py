"""
Improved Rod Segmentation with Dark Region Handling
Handles incomplete/dark sections that get cut off during thresholding
"""

import cv2
import numpy as np
import sys


def segment_rod_edge_based(image):
    """
    Edge-based segmentation - detects boundaries regardless of color/brightness
    Perfect for handling dark regions within the rod
    """
    print("🔍 Edge-Based Segmentation (handles dark regions)...")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection - finds boundaries between regions
    edges = cv2.Canny(blurred, 30, 100)
    
    # Morphological closing to connect nearby edges and fill gaps
    # This is KEY for connecting the dark region to the rest of the circle
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Fill holes (important for dark regions)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find largest contour (should be the rod)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create filled mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        # Additional morphological operations to smooth
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_smooth, iterations=2)
    else:
        mask = closed
    
    result = image.copy()
    result[mask == 0] = 0
    
    return result, (mask > 0).astype('uint8')


def segment_rod_multilevel_threshold(image):
    """
    Multi-level thresholding - captures bright AND dark regions
    """
    print("🎚️  Multi-Level Thresholding...")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to enhance contrast (helps see dark regions better)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Threshold 1: Bright/medium regions (normal gray)
    _, thresh1 = cv2.threshold(enhanced, 80, 255, cv2.THRESH_BINARY)
    
    # Threshold 2: Dark regions (like the black spot)
    _, thresh2 = cv2.threshold(enhanced, 30, 255, cv2.THRESH_BINARY)
    
    # Combine both thresholds
    combined = cv2.bitwise_or(thresh1, thresh2)
    
    # Remove background noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find largest component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype('uint8')
    else:
        mask = (combined > 0).astype('uint8')
    
    result = image.copy()
    result[mask == 0] = 0
    
    return result, mask


def segment_rod_circle_fitting(image):
    """
    Robust circle fitting - fits circle even with incomplete data
    Uses RANSAC-like approach
    """
    print("⭕ Circle Fitting (RANSAC-based)...")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    
    # Hough Circle detection with relaxed parameters
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=100,
        param2=30,  # Lower threshold - more lenient
        minRadius=50,
        maxRadius=min(gray.shape) // 2
    )
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Use the largest circle
        largest_circle = max(circles[0], key=lambda c: c[2])  # c[2] is radius
        x, y, r = largest_circle
        
        # Draw filled circle
        cv2.circle(mask, (x, y), r, 255, -1)
        
        print(f"   Found circle: center=({x},{y}), radius={r}")
    else:
        print("   ⚠️ No circle found, using edge-based fallback")
        # Fallback to edge-based
        return segment_rod_edge_based(image)
    
    result = image.copy()
    result[mask == 0] = 0
    
    return result, (mask > 0).astype('uint8')


def segment_rod_hybrid(image):
    """
    BEST: Hybrid approach combining multiple methods
    """
    print("🚀 Hybrid Approach (Edge + Morphology + Circle Fit)...")
    
    # Step 1: Edge-based detection for initial mask
    _, mask_edges = segment_rod_edge_based(image)
    
    # Step 2: Multi-level thresholding for dark regions
    _, mask_thresh = segment_rod_multilevel_threshold(image)
    
    # Step 3: Combine masks
    mask_combined = cv2.bitwise_or(mask_edges * 255, mask_thresh * 255)
    
    # Step 4: Find largest connected component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_combined, connectivity=8
    )
    
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype('uint8')
        
        # Step 5: Morphological closing to fill any remaining gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Step 6: Convex hull to ensure complete circle (fills concave gaps)
        contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            hull = cv2.convexHull(contours[0])
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [hull], -1, 1, -1)
    else:
        mask = (mask_combined > 0).astype('uint8')
    
    result = image.copy()
    result[mask == 0] = 0
    
    return result, mask


# Test all methods
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "static/debug_upscaled.jpg"
    
    print(f"📂 Loading: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Could not load {image_path}")
        sys.exit(1)
    
    print(f"✅ Image loaded: {image.shape}\n")
    
    # Test all methods
    methods = [
        ("Edge-Based", segment_rod_edge_based),
        ("Multi-Level Threshold", segment_rod_multilevel_threshold),
        ("Circle Fitting", segment_rod_circle_fitting),
        ("Hybrid (BEST)", segment_rod_hybrid)
    ]
    
    print("="*70)
    print("Testing Segmentation Methods")
    print("="*70 + "\n")
    
    results = []
    for name, method in methods:
        print(f"\n{'─'*70}")
        result, mask = method(image.copy())
        results.append((name, result, mask))
        print(f"✅ {name} complete\n")
    
    # Save results
    print("\n" + "="*70)
    print("💾 Saving Results")
    print("="*70)
    
    for i, (name, result, mask) in enumerate(results, 1):
        filename = f"static/improved_seg_{i}_{name.replace(' ', '_').replace('(', '').replace(')', '')}.jpg"
        cv2.imwrite(filename, result)
        
        mask_filename = f"static/improved_mask_{i}_{name.replace(' ', '_').replace('(', '').replace(')', '')}.jpg"
        cv2.imwrite(mask_filename, mask * 255)
        
        print(f"   {i}. {name}")
        print(f"      └─ Result: {filename}")
        print(f"      └─ Mask:   {mask_filename}")
    
    # Create comparison
    h, w = image.shape[:2]
    scale = min(800 / h, 1200 / w, 1.0)
    new_h, new_w = int(h * scale), int(w * scale)
    
    grid_h, grid_w = new_h, new_w * 5
    comparison = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # Original + 4 results
    resized_orig = cv2.resize(image, (new_w, new_h))
    comparison[0:new_h, 0:new_w] = resized_orig
    
    for i, (name, result, mask) in enumerate(results):
        resized = cv2.resize(result, (new_w, new_h))
        comparison[0:new_h, (i+1)*new_w:(i+2)*new_w] = resized
    
    cv2.imwrite("static/improved_segmentation_comparison.jpg", comparison)
    
    print(f"\n🎨 Comparison grid: improved_segmentation_comparison.jpg")
    print("\n✨ DONE! Check http://localhost:8001/static/ for results")
    print("="*70)
