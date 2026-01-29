"""
Background Removal Comparison for Ring Test
Tests multiple segmentation approaches to isolate the rod from background
"""

import cv2
import numpy as np
import sys

def method_1_grabcut(image):
    """GrabCut automatic foreground extraction"""
    print("\n1️⃣  GrabCut Segmentation...")
    
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Initialize rectangle around center (assume rod is in center)
    h, w = image.shape[:2]
    margin = int(min(h, w) * 0.15)  # 15% margin
    rect = (margin, margin, w - 2*margin, h - 2*margin)
    
    # Run GrabCut
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create binary mask (foreground = 1, background = 0)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Apply mask
    result = image * mask2[:, :, np.newaxis]
    
    return result, mask2


def method_2_adaptive_threshold(image):
    """Adaptive Thresholding + Morphology"""
    print("2️⃣  Adaptive Thresholding...")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=21, 
        C=10
    )
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Convert to mask
    mask = (cleaned > 0).astype('uint8')
    result = image * mask[:, :, np.newaxis]
    
    return result, mask


def method_3_color_segmentation(image):
    """Color-based segmentation (LAB color space)"""
    print("3️⃣  Color Segmentation (LAB)...")
    
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Metallic gray has low a and b values (near neutral)
    # Skin tones have higher a (more red)
    
    # Threshold on 'a' channel to remove skin tones
    _, mask_a = cv2.threshold(a, 135, 255, cv2.THRESH_BINARY_INV)
    
    # Threshold on lightness to remove very dark/bright areas
    _, mask_l = cv2.threshold(l, 40, 255, cv2.THRESH_BINARY)
    _, mask_l2 = cv2.threshold(l, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Combine masks
    mask = cv2.bitwise_and(mask_a, mask_l)
    mask = cv2.bitwise_and(mask, mask_l2)
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    mask = (mask > 0).astype('uint8')
    result = image * mask[:, :, np.newaxis]
    
    return result, mask


def method_4_largest_circle(image):
    """Find largest circular contour"""
    print("4️⃣  Largest Circle Detection...")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the most circular contour with largest area
    best_contour = None
    best_score = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:  # Skip tiny contours
            continue
        
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        
        # Circularity = 4π(area/perimeter²) - closer to 1 is more circular
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # Score = area * circularity (prefer large + circular)
        score = area * circularity
        
        if score > best_score:
            best_score = score
            best_contour = cnt
    
    # Create mask from best contour
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if best_contour is not None:
        cv2.drawContours(mask, [best_contour], -1, 255, -1)
    
    mask = (mask > 0).astype('uint8')
    result = image * mask[:, :, np.newaxis]
    
    return result, mask


def method_5_combined(image):
    """Combined approach: GrabCut + Color + Circle"""
    print("5️⃣  Combined Approach...")
    
    # Step 1: GrabCut for rough segmentation
    _, mask_grabcut = method_1_grabcut(image)
    
    # Step 2: Color filtering
    _, mask_color = method_3_color_segmentation(image)
    
    # Step 3: Combine masks
    mask_combined = cv2.bitwise_and(mask_grabcut, mask_color)
    
    # Step 4: Find largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_combined, connectivity=8)
    
    # Skip background label (0)
    if num_labels > 1:
        # Find largest component (excluding background)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_final = (labels == largest_label).astype('uint8')
    else:
        mask_final = mask_combined
    
    result = image * mask_final[:, :, np.newaxis]
    
    return result, mask_final


# Test all methods
if __name__ == "__main__":
    # Load image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "static/debug_upscaled.jpg"  # Default to upscaled image
    
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        sys.exit(1)
    
    print(f"Image shape: {image.shape}")
    
    # Test all methods
    results = []
    
    result1, mask1 = method_1_grabcut(image.copy())
    results.append(("GrabCut", result1, mask1))
    
    result2, mask2 = method_2_adaptive_threshold(image.copy())
    results.append(("Adaptive Threshold", result2, mask2))
    
    result3, mask3 = method_3_color_segmentation(image.copy())
    results.append(("Color Segmentation", result3, mask3))
    
    result4, mask4 = method_4_largest_circle(image.copy())
    results.append(("Largest Circle", result4, mask4))
    
    result5, mask5 = method_5_combined(image.copy())
    results.append(("Combined", result5, mask5))
    
    # Save all results
    print("\n💾 Saving results...")
    for i, (name, result, mask) in enumerate(results, 1):
        # Save masked image
        cv2.imwrite(f"static/bg_removal_{i}_{name.replace(' ', '_')}.jpg", result)
        
        # Save mask visualization
        mask_vis = (mask * 255).astype(np.uint8)
        cv2.imwrite(f"static/mask_{i}_{name.replace(' ', '_')}.jpg", mask_vis)
        
        print(f"   ✅ {name}: bg_removal_{i}_{name.replace(' ', '_')}.jpg")
    
    # Create comparison grid
    h, w = image.shape[:2]
    grid = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)
    
    # Place images in grid
    grid[0:h, 0:w] = image  # Original
    for i, (name, result, mask) in enumerate(results[:5]):
        row = (i + 1) // 3
        col = (i + 1) % 3
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = result
    
    cv2.imwrite("static/background_removal_comparison.jpg", grid)
    print("\n🎨 Comparison grid saved: background_removal_comparison.jpg")
    print("\n✅ Done! Check static/ folder for all results")
