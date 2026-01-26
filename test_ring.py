"""
Simple test script for Tata Steel Ring Test.
Usage: python test_ring.py path/to/image [diameter]
"""

import cv2
import sys
import os
from ring_pipeline import run_ring_test

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_ring.py <image_path> [diameter_mm]")
        print("Default diameter is 12mm")
        return

    img_path = sys.argv[1]
    diameter = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0

    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found")
        return

    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Could not load image {img_path}")
        return

    print(f"🔍 Analyzing {img_path} with {diameter}mm standard...")
    results = run_ring_test(image, diameter_mm=diameter)

    print("\n" + "="*30)
    print(f"RESULT: {results['status']}")
    print(f"REASON: {results['reason']}")
    print("="*30)

    if results['level1']:
        print("\n[Level 1 Results]")
        for k, v in results['level1'].items():
            if k != 'details':
                print(f" - {k}: {v}")

    if results['level2']:
        print("\n[Level 2 Results]")
        print(f" - Thickness: {results['level2']['thickness_mm']:.3f} mm")
        print(f" - Within Standard: {results['level2']['within_standard']}")

    if results['debug_image_path']:
        print(f"\n📸 Debug image: {results['debug_image_path']}")

if __name__ == "__main__":
    main()
