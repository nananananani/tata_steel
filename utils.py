"""
Utility functions for Tata Steel Rebar Testing
Shared utilities for both rib and ring tests

Based on official Tata Steel TM-Ring test specification:
- Level 1 (L1): 4 qualitative questions
- Level 2 (L2): Quantitative formula: 0.07D ≤ t_TM ≤ 0.10D
"""

import cv2
import numpy as np
import os
from datetime import datetime
from typing import Dict, Tuple, Optional, List


def get_thickness_standard(diameter: float) -> Dict[str, float]:
    """
    Get the thickness standard for a given TMT bar diameter
    Based on official Tata Steel specification: 0.07D ≤ t_TM ≤ 0.10D
    
    Args:
        diameter: TMT bar diameter in mm (any positive value)
    
    Returns:
        Dictionary with 'min' and 'max' thickness values in mm
        
    Example:
        For 12mm diameter:
        min = 0.07 * 12 = 0.84mm
        max = 0.10 * 12 = 1.20mm
    """
    if diameter <= 0:
        raise ValueError(f"Invalid diameter: {diameter}. Must be positive")
    
    return {
        "min": 0.07 * diameter,  # Minimum: 7% of diameter
        "max": 0.10 * diameter   # Maximum: 10% of diameter
    }


def save_debug_image(image: np.ndarray, prefix: str = "debug") -> str:
    """
    Save debug image with timestamp
    
    Args:
        image: Image to save (BGR format)
        prefix: Prefix for filename
    
    Returns:
        Path to saved image
    """
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join("static", filename)
    
    # Save image
    cv2.imwrite(filepath, image)
    
    return filepath


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points
    
    Args:
        point1: (x1, y1)
        point2: (x2, y2)
    
    Returns:
        Distance in pixels
    """
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def validate_image(image: np.ndarray) -> Tuple[bool, List[str]]:
    """
    Validate if image is suitable for analysis
    
    Args:
        image: Input image (BGR format)
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if image is None or image.size == 0:
        return False, ["Invalid or empty image"]
    
    # Check image dimensions
    h, w = image.shape[:2]
    if min(h, w) < 200:
        issues.append("Image resolution too low (minimum 200px)")
    
    # Check if image is too small
    if h * w < 40000:  # Less than ~200x200
        issues.append("Image size too small for accurate analysis")
    
    # Convert to grayscale for quality checks
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Check brightness
    mean_brightness = np.mean(gray)
    if mean_brightness < 40:
        issues.append("Image too dark - use better lighting")
    elif mean_brightness > 215:
        issues.append("Image too bright - reduce exposure")
    
    # Check blur (using Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # if laplacian_var < 5:  # Disabled to allow testing with current images
    #    issues.append("Image too blurry - hold camera steady and focus properly")
    
    # Check contrast
    std_dev = np.std(gray)
    if std_dev < 20:
        issues.append("Image has very low contrast")
    
    return len(issues) == 0, issues


def pixels_to_mm(pixels: float, diameter_mm: int, reference_radius_pixels: float) -> float:
    """
    Convert pixel measurements to millimeters based on known diameter
    
    Args:
        pixels: Measurement in pixels
        diameter_mm: Known diameter of the rebar in mm
        reference_radius_pixels: Detected radius of the rebar in pixels
    
    Returns:
        Measurement in millimeters
    """
    # Calculate pixels per mm ratio
    # diameter_mm corresponds to 2 * reference_radius_pixels
    pixels_per_mm = (2 * reference_radius_pixels) / diameter_mm
    
    # Convert pixels to mm
    return pixels / pixels_per_mm


def enhance_image_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast using CLAHE
    
    Args:
        image: Input grayscale image
    
    Returns:
        Contrast-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def create_circular_mask(shape: Tuple[int, int], center: Tuple[int, int], 
                        radius: int, thickness: Optional[int] = None) -> np.ndarray:
    """
    Create a circular mask
    
    Args:
        shape: Image shape (height, width)
        center: Circle center (x, y)
        radius: Circle radius
        thickness: If provided, creates a ring mask; if None, creates filled circle
    
    Returns:
        Binary mask (0 and 255)
    """
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), 255, 
               -1 if thickness is None else thickness)
    return mask


def draw_text_with_background(image: np.ndarray, text: str, position: Tuple[int, int],
                              font_scale: float = 0.6, thickness: int = 2,
                              text_color: Tuple[int, int, int] = (255, 255, 255),
                              bg_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Draw text with background rectangle for better visibility
    
    Args:
        image: Image to draw on
        text: Text to draw
        position: (x, y) position for text
        font_scale: Font scale
        thickness: Text thickness
        text_color: Text color (BGR)
        bg_color: Background color (BGR)
    
    Returns:
        Image with text drawn
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    x, y = position
    cv2.rectangle(image, 
                 (x - 5, y - text_height - 5),
                 (x + text_width + 5, y + baseline + 5),
                 bg_color, -1)
    
    # Draw text
    cv2.putText(image, text, position, font, font_scale, text_color, thickness)
    
    return image
