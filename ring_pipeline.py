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

    def run_ring_test(self, image: np.ndarray) -> Dict:
        # 1. Validation
        is_valid, issues = validate_image(image)
        if not is_valid:
            return {
                "status": "FAIL",
                "reason": f"Image quality issues: {'; '.join(issues)}",
                "level1": None,
                "level2": None,
                "debug_image_path": None
            }

        # 2. Preprocess
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        enhanced = enhance_image_contrast(blurred)
        
        # 3. Detection
        # We try to find the outer boundary first using thresholding + contours
        # This is more robust than HoughCircles for irregular rebar shapes.
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up threshold
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"status": "FAIL", "reason": "No rebar boundary detected", "level1": None, "level2": None, "debug_image_path": None}
            
        outer_cnt = max(contours, key=cv2.contourArea)
        outer_area = cv2.contourArea(outer_cnt)
        (ox, oy), o_radius_circle = cv2.minEnclosingCircle(outer_cnt)
        
        # Now find the inner core (smooth out the grain noise)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [outer_cnt], -1, 255, -1)
        
        # Smoothen core detection to avoid jagged lines
        core_blurred = cv2.medianBlur(gray, 11)
        core_val = np.percentile(core_blurred[mask > 0], 65) 
        _, core_thresh = cv2.threshold(core_blurred, core_val, 255, cv2.THRESH_BINARY)
        core_thresh = cv2.bitwise_and(core_thresh, mask)
        
        # Clean the core mask
        core_thresh = cv2.morphologyEx(core_thresh, cv2.MORPH_OPEN, kernel)
        
        core_contours, _ = cv2.findContours(core_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not core_contours:
            inner_cnt = None
            inner_area = 0.75 * outer_area # Fallback
            (ix, iy), i_radius_circle = ox, oy, o_radius_circle * 0.866
        else:
            raw_inner_cnt = max(core_contours, key=cv2.contourArea)
            # Use Convex Hull to smooth out the jagged edges from metal grain
            inner_cnt = cv2.convexHull(raw_inner_cnt)
            inner_area = cv2.contourArea(inner_cnt)
            (ix, iy), i_radius_circle = cv2.minEnclosingCircle(inner_cnt)

        # 4. Metrics (Based on provided formula: t = sqrt(A/pi) - sqrt(A_FP/pi))
        equivalent_r_outer = np.sqrt(outer_area / np.pi)
        equivalent_r_inner = np.sqrt(inner_area / np.pi)
        
        thickness_equivalent_px = equivalent_r_outer - equivalent_r_inner
        
        # Use the nominal diameter D provided by user for pixel-to-mm conversion
        # We assume the rebar area in image represents the cross-section of diameter D
        px_per_mm = (2 * equivalent_r_outer) / self.diameter_mm
        thickness_mm = thickness_equivalent_px / px_per_mm
        
        center_dist = calculate_distance((ox, oy), (ix, iy))
        
        # Level 1 Checks
        regions_visible = abs(np.mean(gray[core_thresh > 0]) - np.mean(gray[(mask > 0) & (core_thresh == 0)])) > 15 if core_contours else False
        
        # Continuous ring check
        outer_arc = cv2.arcLength(outer_cnt, True)
        circularity = (4 * np.pi * outer_area) / (outer_arc * outer_arc) if outer_arc > 0 else 0
        ring_continuous = circularity > 0.6
        
        # Concentricity check
        is_concentric = center_dist < (equivalent_r_outer * 0.20)
        
        # Thickness uniformity
        thickness_uniform = circularity > 0.65

        l1_pass = regions_visible and ring_continuous and is_concentric and thickness_uniform
        
        # Level 2 Checks
        thickness_pct = (thickness_mm / self.diameter_mm) * 100
        within_std = self.standards['min'] <= thickness_mm <= self.standards['max']
        
        status = "PASS" if (l1_pass and within_std) else "FAIL"
        
        # Status Reason
        reasons = []
        if not regions_visible: reasons.append("Regions not distinct")
        if not ring_continuous: reasons.append("Ring not continuous")
        if not is_concentric: reasons.append(f"Non-concentric ({center_dist:.1f}px offset)")
        if not thickness_uniform: reasons.append("Non-uniform thickness")
        if not within_std: 
            if thickness_mm < self.standards['min']:
                reasons.append(f"Thickness {thickness_mm:.2f}mm too low (min {self.standards['min']:.2f}mm)")
            else:
                reasons.append(f"Thickness {thickness_mm:.2f}mm too high (max {self.standards['max']:.2f}mm)")
        
        reason = "All criteria met" if status == "PASS" else "; ".join(reasons)

        results = {
            "status": status,
            "reason": reason,
            "level1": {
                "regions_visible": bool(regions_visible),
                "ring_continuous": bool(ring_continuous),
                "concentric": bool(is_concentric),
                "thickness_uniform": bool(thickness_uniform),
                "details": {
                    "center_distance_px": float(center_dist),
                    "circularity": float(circularity)
                }
            },
            "level2": {
                "thickness_mm": float(thickness_mm),
                "thickness_pct": float(thickness_pct),
                "within_standard": bool(within_std),
                "standard_range": self.standards,
                "dimensions": {
                    "outer_radius_px": float(equivalent_r_outer),
                    "inner_radius_px": float(equivalent_r_inner),
                    "thickness_px": float(thickness_equivalent_px)
                }
            }
        }

        # 5. Technical Visualization (Official Tata Steel Style)
        debug_img = image.copy()
        h, w = debug_img.shape[:2]
        
        # 1. Draw Outlines
        cv2.drawContours(debug_img, [outer_cnt], -1, (255, 100, 0), 3) # TM Outline
        if inner_cnt is not None:
            cv2.drawContours(debug_img, [inner_cnt], -1, (0, 255, 0), 3) # FP Outline

        # 2. Add Technical Leader Lines & Labels
        # TM Label (Outer Ring) - Top Right
        tm_label = "Tempered Martensite (TM)"
        tm_text_pos = (int(w * 0.60), 60)
        tm_point = (int(ox + equivalent_r_outer * 0.5), int(oy - equivalent_r_outer * 0.86))
        # Draw shelf and put text ABOVE
        cv2.line(debug_img, (tm_text_pos[0], tm_text_pos[1]), (tm_text_pos[0] + 220, tm_text_pos[1]), (255, 100, 0), 2)
        cv2.line(debug_img, (tm_text_pos[0], tm_text_pos[1]), tm_point, (255, 100, 0), 2)
        cv2.putText(debug_img, tm_label, (tm_text_pos[0] + 5, tm_text_pos[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

        # FP Label (Inner Core) - Bottom Left
        fp_label = "Ferrite Pearlite (FP)"
        fp_text_pos = (60, h - 80)
        fp_point = (int(ix - 10), int(iy + 10))
        # Draw shelf and put text BELOW (preventing line overlap)
        cv2.line(debug_img, (fp_text_pos[0], fp_text_pos[1]), (fp_text_pos[0] + 180, fp_text_pos[1]), (0, 255, 0), 2)
        cv2.line(debug_img, (fp_text_pos[0] + 180, fp_text_pos[1]), fp_point, (0, 255, 0), 2)
        cv2.putText(debug_img, fp_label, (fp_text_pos[0] + 5, fp_text_pos[1] + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        # Draw Precise Centers
        cv2.circle(debug_img, (int(ox), int(oy)), 6, (0, 0, 255), -1)
        cv2.circle(debug_img, (int(ix), int(iy)), 6, (255, 255, 0), -1)
        
        # Professional Result Header
        res_header = f"RESULT: {status} ({thickness_mm:.2f}mm)"
        draw_text_with_background(debug_img, res_header, (25, 40), font_scale=0.9, thickness=2, 
                                 text_color=(0, 255, 0) if status == "PASS" else (0, 0, 255))
        
        debug_path = save_debug_image(debug_img, "ring_debug")
        results["debug_image_path"] = debug_path
        results["debug_image_url"] = f"/static/{os.path.basename(debug_path)}"

        return results

def run_ring_test(image: np.ndarray, diameter_mm: float = 12.0) -> Dict:
    import os
    pipeline = RingTestPipeline(diameter_mm=diameter_mm)
    return pipeline.run_ring_test(image)
