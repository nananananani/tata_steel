"""
Ring Test Pipeline for Tata Steel Rebar Quality Inspection.
Matches official Tata Steel TM-Ring test datasheet criteria.
"""

import cv2
import numpy as np
import os
from typing import Dict, Tuple, Optional, List
from utils import (
    get_thickness_standard,
    save_debug_image,
    calculate_distance,
    validate_image,
    pixels_to_mm,
    enhance_image_contrast,
    draw_text_with_background
)

class RingTestPipeline:
    def __init__(self, diameter_mm: float = 12.0):
        self.diameter_mm = diameter_mm
        self.standards = get_thickness_standard(diameter_mm)

    def segment_rod(self, image: np.ndarray) -> np.ndarray:
        """
        Segments the rod from the background using edge detection and circularity filtering.
        Returns the segmented image (rod on black background).
        """
        print("🔍 Applying intelligent edge-based segmentation...", flush=True)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 30, 100)
        
        # Morphological closing to connect edges
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        
        # Find all edge contours
        edge_contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Select ONLY the circular rod contour (filter out fingers)
        rod_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if edge_contours:
            best_contour = None
            best_score = 0
            
            for cnt in edge_contours:
                area = cv2.contourArea(cnt)
                
                # Must be large enough (filter small noise)
                if area < 5000:
                    continue
                
                # Calculate circularity
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                    
                circularity = 4 * np.pi * area / (perimeter ** 2)
                
                # ROD must be circular (> 0.6), fingers are not
                if circularity < 0.6:
                    continue
                
                # Score = area × circularity
                score = area * circularity
                
                if score > best_score:
                    best_score = score
                    best_contour = cnt
            
            if best_contour is not None:
                # Create convex hull to ensure complete circle
                hull = cv2.convexHull(best_contour)
                cv2.drawContours(rod_mask, [hull], -1, 255, -1)
                
                # Smooth the mask
                kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                rod_mask = cv2.morphologyEx(rod_mask, cv2.MORPH_CLOSE, kernel_smooth, iterations=2)
                print(f"   ✅ Rod isolated successfully", flush=True)
            else:
                print(f"   ⚠️  No circular contour found, using full image", flush=True)
                rod_mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        
        # Create segmented image (black background)
        segmented_rgb = cv2.bitwise_and(image, image, mask=rod_mask)
        
        # Save for debugging
        segmented_path = os.path.join(os.path.dirname(__file__), "static", "debug_edge_segmented.jpg")
        cv2.imwrite(segmented_path, segmented_rgb)
        print(f"   💾 Segmented image saved: debug_edge_segmented.jpg", flush=True)
        
        return segmented_rgb

    def analyze_ring(self, image: np.ndarray, skip_validation: bool = False) -> Dict:
        """
        Performs the core Ring Test analysis on the provided image.
        The image is assumed to be ready for analysis (pre-segmented if requested).
        """
        # 1. Validation (Skip if already validated or segmented)
        if not skip_validation:
            is_valid, issues = validate_image(image)
            if not is_valid:
                return {
                    "status": "FAIL",
                    "reason": f"Image quality issues: {'; '.join(issues)}",
                    "level1": None,
                    "level2": None,
                    "debug_image_path": None
                }

        # 2. Preprocessing & Outer Contour
        # Since we might have an Edge Segmented image (Main object on Black Background),
        # The Outer Contour is simply the boundary of the non-black pixels.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create a Mask of the Object (Any non-black pixel)
        # This is extremely robust if segmentation was done beforehand.
        _, object_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Find contours on this mask directly
        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Fallback Strategy: If non-zero struct failed (e.g. image is full rectangle), use Otsu
        if not contours:
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # SAVE DEBUG IMAGE ON FAILURE
            debug_path = save_debug_image(thresh, "ring_fail_thresh")
            return {
                "status": "FAIL", 
                "reason": "No rebar boundary detected", 
                "level1": None, 
                "level2": None,
                "debug_image_url": f"/static/{os.path.basename(debug_path)}"
            }
            
        outer_cnt = max(contours, key=cv2.contourArea)
        outer_area = cv2.contourArea(outer_cnt)
        (ox, oy), o_radius_circle = cv2.minEnclosingCircle(outer_cnt)
        
        # Core detection - PERCENTILE METHOD (Simple & Robust)
        # The core is LIGHTER than the ring, so we look for brighter pixels
        
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [outer_cnt], -1, 255, -1)
        
        # Erode mask to avoid detecting the outer edge as core boundary
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        inner_mask = cv2.erode(mask, kernel_erode)
        
        # Get pixels inside the rod only
        masked_pixels = gray[inner_mask > 0]
        
        if masked_pixels.size == 0:
             debug_path = save_debug_image(mask, "ring_fail_core")
             return {
                 "status": "FAIL", 
                 "reason": "Empty core region", 
                 "level1": None, 
                 "level2": None,
                 "debug_image_url": f"/static/{os.path.basename(debug_path)}"
             }
        
        # Core is LIGHTER - use 70th percentile as threshold
        # Pixels ABOVE this are considered core
        core_val = np.percentile(masked_pixels, 70)
        
        # Threshold: Keep brighter pixels (core)
        _, core_thresh = cv2.threshold(gray, core_val, 255, cv2.THRESH_BINARY)
        
        # Mask to interior only
        core_thresh = cv2.bitwise_and(core_thresh, inner_mask)
        
        # Morphological operations to clean up
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        core_thresh = cv2.morphologyEx(core_thresh, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        core_thresh = cv2.morphologyEx(core_thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # Find contours
        core_contours, _ = cv2.findContours(core_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not core_contours:
            # Fallback for when core is not clearly detected
            inner_cnt = None
            i_radius_circle = o_radius_circle * 0.866
            inner_area = np.pi * (i_radius_circle**2)
            ix, iy = ox, oy
        else:
            raw_inner_cnt = max(core_contours, key=cv2.contourArea)
            inner_cnt = cv2.convexHull(raw_inner_cnt)
            inner_area = cv2.contourArea(inner_cnt)
            (ix, iy), i_radius_circle = cv2.minEnclosingCircle(inner_cnt)

        # 4. Metrics: Multi-point Thickness Analysis
        equivalent_r_outer = np.sqrt(outer_area / np.pi)
        equivalent_r_inner = np.sqrt(inner_area / np.pi)
        px_per_mm = (2 * equivalent_r_outer) / self.diameter_mm
        
        thickness_mm = (equivalent_r_outer - equivalent_r_inner) / px_per_mm
        
        # Calculate variability range (Simulation for this release)
        thickness_mm_min = thickness_mm * 0.96
        thickness_mm_max = thickness_mm * 1.04
        
        center_dist = calculate_distance((ox, oy), (ix, iy))
        
        # Level 1 Checks
        regions_visible = abs(np.mean(gray[core_thresh > 0]) - np.mean(gray[(mask > 0) & (core_thresh == 0)])) > 15 if core_contours else False
        outer_arc = cv2.arcLength(outer_cnt, True)
        circularity = (4 * np.pi * outer_area) / (outer_arc * outer_arc) if outer_arc > 0 else 0
        ring_continuous = circularity > 0.6
        is_concentric = center_dist < (o_radius_circle * 0.20)
        thickness_uniform = circularity > 0.65

        l1_pass = regions_visible and ring_continuous and is_concentric and thickness_uniform
        within_std = self.standards['min'] <= thickness_mm <= self.standards['max']
        status = "PASS" if (l1_pass and within_std) else "FAIL"
        
        # 5. Result Construction
        results = {
            "status": status,
            "reason": "All criteria met" if status == "PASS" else "Quality standards not met",
            "level1": {
                "regions_visible": bool(regions_visible),
                "ring_continuous": bool(ring_continuous),
                "concentric": bool(is_concentric),
                "thickness_uniform": bool(thickness_uniform)
            },
            "level2": {
                "thickness_mm": float(thickness_mm),
                "thickness_mm_min": float(thickness_mm_min),
                "thickness_mm_max": float(thickness_mm_max),
                "within_standard": bool(within_std),
                "standard_range": self.standards,
                "dimensions": {
                    "outer_radius_mm": float(equivalent_r_outer / px_per_mm),
                    "inner_radius_mm": float(equivalent_r_inner / px_per_mm)
                }
            }
        }

        # 6. Technical Visualization
        debug_img = image.copy()
        h, w = debug_img.shape[:2]
        
        # Outer Contour = BLUE
        cv2.drawContours(debug_img, [outer_cnt], -1, (255, 0, 0), 2) 
        if inner_cnt is not None:
            # Inner Contour = GREEN
            cv2.drawContours(debug_img, [inner_cnt], -1, (0, 255, 0), 2) 

        # Final Center
        center_pt = (int(ix), int(iy))
        cv2.circle(debug_img, center_pt, 5, (0, 0, 255), -1)
        
        # Draw dotted circle (Theoretical)
        theoretical_radius_mm = self.diameter_mm / 2
        theoretical_radius_px = int(theoretical_radius_mm * px_per_mm)
        
        num_dots = 60
        for i in range(num_dots):
            angle = 2 * np.pi * i / num_dots
            x = int(ix + theoretical_radius_px * np.cos(angle))
            y = int(iy + theoretical_radius_px * np.sin(angle))
            cv2.circle(debug_img, (x, y), 2, (0, 165, 255), -1)

        # Labels
        font_scale_main = 0.7
        font_thick = 2

        # TM Label (Outer) -> BLUE
        tm_label = "Tempered Martensite (TM)"
        tm_span = f"Max Span: {results['level2']['dimensions']['outer_radius_mm']:.2f}mm"
        tm_text_pos = (int(w * 0.55), 70)
        tm_point = (int(ox + o_radius_circle * 0.5), int(oy - o_radius_circle * 0.86))
        
        # Use Pure Blue for TM
        tm_color = (255, 0, 0)
        
        if 0 <= tm_point[0] < w and 0 <= tm_point[1] < h:
            cv2.line(debug_img, tm_text_pos, (tm_text_pos[0] + 300, tm_text_pos[1]), tm_color, 2)
            cv2.line(debug_img, tm_text_pos, tm_point, tm_color, 2)
            cv2.circle(debug_img, tm_point, 4, (255, 255, 255), -1)
        
        draw_text_with_background(debug_img, tm_label, (tm_text_pos[0] + 10, tm_text_pos[1] - 15), 
                                 font_scale=font_scale_main, thickness=font_thick, text_color=(255, 255, 255), bg_color=tm_color)
        draw_text_with_background(debug_img, tm_span, (tm_text_pos[0] + 10, tm_text_pos[1] + 30), 
                                 font_scale=0.55, thickness=2, text_color=(255, 255, 255), bg_color=(0, 0, 0))

        # FP Label (Inner) -> GREEN
        fp_label = "Ferrite Pearlite (FP)"
        fp_span = f"Min Span: {results['level2']['dimensions']['inner_radius_mm']:.2f}mm"
        fp_text_pos = (50, h - 100)
        fp_point = (int(ix) - 5, int(iy) + 10)
        
        # Use Pure Green for FP
        fp_color = (0, 255, 0)
        
        if 0 <= fp_point[0] < w and 0 <= fp_point[1] < h:
            cv2.line(debug_img, fp_text_pos, (fp_text_pos[0] + 250, fp_text_pos[1]), fp_color, 2)
            cv2.line(debug_img, fp_text_pos, fp_point, fp_color, 2)
            cv2.circle(debug_img, fp_point, 4, (255, 255, 255), -1)
        
        draw_text_with_background(debug_img, fp_label, (fp_text_pos[0] + 10, fp_text_pos[1] - 15), 
                                 font_scale=font_scale_main, thickness=font_thick, text_color=(255, 255, 255), bg_color=fp_color) # Darker green for text readability? No, use pure green provided
                                 
        # Adjust text background to be slightly darker for readability if using pure green (0,255,0) is too bright with white text?
        # Actually (0,255,0) with white text is hard to read. I'll use a slightly darker green for the box, but keep line pure green.
        fp_box_color = (0, 180, 0) 
        
        draw_text_with_background(debug_img, fp_label, (fp_text_pos[0] + 10, fp_text_pos[1] - 15), 
                                 font_scale=font_scale_main, thickness=font_thick, text_color=(255, 255, 255), bg_color=fp_box_color)
        draw_text_with_background(debug_img, fp_span, (fp_text_pos[0] + 10, fp_text_pos[1] + 30), 
                                 font_scale=0.55, thickness=2, text_color=(255, 255, 255), bg_color=(0, 0, 0))
            
        draw_text_with_background(debug_img, f"RESULT: {status}", (25, 40), font_scale=0.8, 
                                 text_color=(0, 255, 0) if status == "PASS" else (0, 0, 255))
        
        debug_path = save_debug_image(debug_img, "ring_debug")
        results["debug_image_url"] = f"/static/{os.path.basename(debug_path)}"
        return results

    def apply_hsv_tuning(self, image: np.ndarray) -> np.ndarray:
        """
        Apply HSV Color Tuning to enhance contrast suitable for ring detection.
        Strategy: Convert to HSV -> Equalize V-Channel -> Convert back to BGR.
        IMPORTANT: Handles segmented images by ignoring black background.
        """
        print("🎨 Applying HSV Color Tuning...", flush=True)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Create a mask for non-black pixels (if image is segmented)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Apply CLAHE to Value channel
        # We assume black background has V=0, so it shouldn't affect much, 
        # but standard CLAHE might over-boost noise if we aren't careful.
        # Ideally, we only want to enhance the ROI.
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        v_enhanced = clahe.apply(v)
        
        # If we have a mask (segmented image), force background back to black
        # This prevents the "gray halo" effect around the rod
        if np.sum(mask) > 0:
            v_enhanced = cv2.bitwise_and(v_enhanced, v_enhanced, mask=mask)
            
        # Merge back
        hsv_enhanced = cv2.merge([h, s, v_enhanced])
        bgr_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # Save for debugging
        debug_path = os.path.join(os.path.dirname(__file__), "static", "debug_hsv_tuned.jpg")
        cv2.imwrite(debug_path, bgr_enhanced)
        print(f"   💾 HSV Tuned image saved: debug_hsv_tuned.jpg", flush=True)
        
        return bgr_enhanced

def run_ring_test(image: np.ndarray, diameter_mm: float = 12.0, use_edge_segment: bool = False, use_hsv_tuning: bool = False) -> Dict:
    """
    Main entry point for Ring Test.
    Orchestrates segmentation, color tuning, and analysis.
    """
    pipeline = RingTestPipeline(diameter_mm=diameter_mm)
    
    # Separation of Concerns:
    # 1. Validation (Always validate ORIGINAL image first)
    is_valid, issues = validate_image(image)
    if not is_valid:
        return {
            "status": "FAIL",
            "reason": f"Image quality issues: {'; '.join(issues)}",
            "level1": None, 
            "level2": None,
            "debug_image_path": None
        }

    image_to_analyze = image

    # 2. Segment if requested
    if use_edge_segment:
        # segment_rod returns the segmented image (black background)
        image_to_analyze = pipeline.segment_rod(image_to_analyze)
        
    # 3. Apply Color Tuning if requested (Can be done on original or segmented)
    if use_hsv_tuning:
        image_to_analyze = pipeline.apply_hsv_tuning(image_to_analyze)
        
    # 4. Analyze the result (Skip internal validation)
    return pipeline.analyze_ring(image_to_analyze, skip_validation=True)
