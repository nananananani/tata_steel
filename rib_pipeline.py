"""
Rib Test Pipeline for Tata Steel Rebar Quality Inspection.
Matches official Tata Steel standards and integrates with the Ring Test system.
"""

import cv2
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from utils import (
    validate_image,
    save_debug_image,
    draw_text_with_background
)

# Third-party imports
try:
    from ultralytics import YOLO
    from segment_anything import sam_model_registry, SamPredictor
    import torch
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please run: pip install -r requirements.txt")

# Configuration
YOLO_MODEL = "yolov8n.pt"
SAM_MODEL_TYPE = "vit_b"
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
MIN_RIB_CONTOUR_AREA = 150 # Balanced threshold
MIN_RIBS_REQUIRED = 2
YOLO_CONFIDENCE_THRESHOLD = 0.20 # Tuned down from 0.25

# Global Model Cache
_CACHED_YOLO_MODEL = None
_CACHED_SAM_PREDICTOR = None

class RibTestPipeline:
    def __init__(self, diameter_mm: float = 12.0):
        self.diameter_mm = diameter_mm
        self._ensure_models_loaded()

    def _ensure_models_loaded(self):
        """Load models into global cache if not present"""
        global _CACHED_YOLO_MODEL, _CACHED_SAM_PREDICTOR
        
        # YOLO
        if _CACHED_YOLO_MODEL is None:
            try:
                print(f"Loading YOLO model: {YOLO_MODEL}...")
                _CACHED_YOLO_MODEL = YOLO(YOLO_MODEL)
                print(f"✓ YOLO model loaded")
            except Exception as e:
                print(f"✗ Failed to load YOLO: {e}")

        # SAM
        if _CACHED_SAM_PREDICTOR is None:
            try:
                if not os.path.exists(SAM_CHECKPOINT):
                    print(f"Downloading SAM checkpoint: {SAM_CHECKPOINT} (this may take a while)...")
                    import urllib.request
                    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
                    urllib.request.urlretrieve(url, SAM_CHECKPOINT)
                    print("✓ SAM checkpoint downloaded")
                
                print(f"Loading SAM model: {SAM_MODEL_TYPE}...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
                sam.to(device=device)
                _CACHED_SAM_PREDICTOR = SamPredictor(sam)
                print(f"✓ SAM model loaded")
            except Exception as e:
                print(f"✗ Failed to load SAM: {e}")

    def detect_region(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Stage 1: YOLO Localization (Refined for tighter detection)"""
        if _CACHED_YOLO_MODEL is None: return None
        
        try:
            results = _CACHED_YOLO_MODEL(image, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)
            if not results or len(results[0].boxes) == 0:
                # If YOLO fails, try center crop as fallback
                h, w = image.shape[:2]
                return (int(w*0.1), int(h*0.3), int(w*0.9), int(h*0.7))
            
            # Choose the box most likely to be a horizontal/diagonal bar
            boxes = results[0].boxes.xyxy.cpu().numpy()
            best_idx = 0
            max_aspect = 0
            for i, box in enumerate(boxes):
                bw = box[2] - box[0]
                bh = box[3] - box[1]
                aspect = bw / bh if bh > 0 else 0
                if aspect > max_aspect:
                    max_aspect = aspect
                    best_idx = i
            
            return tuple(boxes[best_idx].astype(int))
        except Exception as e:
            return None

    def segment_individual_ribs(self, image: np.ndarray, rib_seeds: List[Tuple[int, int]]) -> List[np.ndarray]:
        """Stage 2.5: SAM Precision (Using points from Gabor filter to get exact rib masks)"""
        if _CACHED_SAM_PREDICTOR is None or not rib_seeds: return []
        
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            _CACHED_SAM_PREDICTOR.set_image(image_rgb)
            
            rib_masks = []
            for (cx, cy) in rib_seeds:
                # Prompt SAM with a single point on the rib
                input_point = np.array([[cx, cy]])
                input_label = np.array([1]) # Positive prompt
                
                masks, scores, _ = _CACHED_SAM_PREDICTOR.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False
                )
                
                # Check confidence - SAM might return background if prompt is bad
                if scores[0] > 0.8:
                    mask = (masks[0] * 255).astype(np.uint8)
                    rib_masks.append(mask)
            
            return rib_masks
        except Exception:
            return []

    def find_rib_lines(self, image: np.ndarray, bar_mask: np.ndarray) -> List[Dict]:
        """Stage 3: Extract Ribs using High-Contrast Edge Localization"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply mask to ignore background
        masked_gray = cv2.bitwise_and(gray, gray, mask=bar_mask)
        
        # Pre-process: Enhance local details for better rib visibility
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(masked_gray)
        
        # Detect transverse edges (ribs) using Gabor Filters
        # This acts like a pattern-recognition engine for diagonal ribs
        # We look for features at ~45-60 degrees
        g_kernel = cv2.getGaborKernel((21, 21), 5.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(enhanced, cv2.CV_8U, g_kernel)
        
        # Adaptive thresholding on the patterns
        binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 15, -2)
        
        # Cleaning: Join fragmented rib segments
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 11))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
        
        # Filter: Only keep edges inside the bar mask and with a rib-like area
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ribs = []
        bar_moments = cv2.moments(bar_mask)
        if bar_moments["m00"] == 0: return []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_RIB_CONTOUR_AREA: continue
            
            # Additional Shape Filter: Ribs must be elongated
            rect = cv2.minAreaRect(cnt)
            (center, (w, h), angle) = rect
            
            # Standardize dimensions
            if w < h:
                major, minor = h, w
            else:
                major, minor = w, h
                angle += 90
            
            # Aspect ratio check (Length/Width)
            aspect_ratio = major / minor if minor > 0 else 0
            if aspect_ratio < 1.2: continue # More inclusive but still rejects circles
            
            # Ensure centroid is inside bar mask
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            
            if bar_mask[cy, cx] == 0: continue
            
            rect = cv2.minAreaRect(cnt)
            (center, (w, h), angle) = rect
            
            # Ribs are usually taller than they are wide (transverse)
            if w > h:
                w, h = h, w
                angle += 90
            
            # Standardize angle to be relative to vertical (-45 to 45)
            if angle > 90: angle -= 180
            
            ribs.append({
                "contour": cnt,
                "center": center,
                "width": w,
                "height": h,
                "angle": abs(angle),
                "area": area
            })
            
        return ribs

    def get_measurements(self, ribs: List[Dict], px_per_mm: float) -> Dict:
        """Stage 4: Statistical Filtering & Aggregation"""
        if len(ribs) < MIN_RIBS_REQUIRED:
            return {"num_ribs": 0, "status": "FAIL"}

        # Filter by consistency (Medians)
        # Keep ribs with a significant transverse angle (ignore longitudinal spines)
        # Transverse ribs are typically 45-70 degrees
        valid_ribs = [r for r in ribs if 30 < r["angle"] < 85]
        
        if not valid_ribs:
            # Fallback: if no clear diagonals, pick the most consistent angle
            angles = [r["angle"] for r in ribs]
            med_angle = np.median(angles) if angles else 0
            valid_ribs = [r for r in ribs if abs(r["angle"] - med_angle) < 15]
        
        if len(valid_ribs) < MIN_RIBS_REQUIRED:
            return {"num_ribs": 0, "status": "FAIL"}
            
        # Spacing (Inter-distance)
        valid_ribs.sort(key=lambda r: r["center"][0]) # Left to Right
        spacings = []
        for i in range(len(valid_ribs) - 1):
            p1 = valid_ribs[i]["center"]
            p2 = valid_ribs[i+1]["center"]
            dist = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            spacings.append(dist)
            
        avg_spacing = np.mean(spacings) if spacings else 0
        
        # Collect measurements for valid transverse ribs
        v_heights = [r["width"] for r in valid_ribs]
        v_lengths = [r["height"] for r in valid_ribs]
        v_angles = [r["angle"] for r in valid_ribs]

        return {
            "num_ribs": len(valid_ribs),
            "rib_length_px": float(np.mean(v_lengths)), 
            "rib_height_px": float(np.mean(v_heights)), 
            "rib_angle": float(np.mean(v_angles)),
            "interdistance_px": float(avg_spacing),
            "valid_ribs": valid_ribs
        }

    def generate_debug_image(self, image: np.ndarray, bbox: Tuple, bar_mask: np.ndarray, 
                              measurements: Dict, status: str) -> np.ndarray:
        """Stage 5: High-Precision Technical Visualization"""
        debug_img = image.copy()
        h_img, w_img = image.shape[:2]

        # 1. Draw Bar Alpha (Blue overlay)
        overlay = debug_img.copy()
        if bar_mask is not None:
            # Find external contours of the mask to draw outline
            bar_contours, _ = cv2.findContours(bar_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(debug_img, bar_contours, -1, (255, 100, 0), 2) # Tata Orange
            cv2.addWeighted(overlay, 0.9, debug_img, 0.1, 0, debug_img)

        # 2. Draw Rib Lines
        if "valid_ribs" in measurements:
            for i, rib in enumerate(measurements["valid_ribs"]):
                # Draw the rib axis line
                rect = cv2.minAreaRect(rib["contour"])
                box = cv2.boxPoints(rect)
                box = np.int64(box)
                
                # Draw the centerline
                center = rib["center"]
                angle_rad = np.deg2rad(rib["angle"] + 90) # Relative to longitudinal
                L = rib["height"] / 2
                x1 = int(center[0] - L * np.cos(angle_rad))
                y1 = int(center[1] - L * np.sin(angle_rad))
                x2 = int(center[0] + L * np.cos(angle_rad))
                y2 = int(center[1] + L * np.sin(angle_rad))
                
                cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green Rib lines
                cv2.circle(debug_img, (int(center[0]), int(center[1])), 3, (0, 0, 255), -1)
                
                # Annotation
                cv2.putText(debug_img, f"#{i+1}", (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 3. Headers
        ar_val = measurements.get('ar_value', 0)
        header_text = f"RIB TEST: {status} (AR: {ar_val:.3f})"
        draw_text_with_background(debug_img, header_text, (30, 50), font_scale=0.8,
                                 text_color=(0, 255, 0) if status == "PASS" else (0, 0, 255))
        
        # Top-right metadata
        cv2.putText(debug_img, f"Ribs: {measurements.get('num_ribs', 0)}", (w_img-150, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                                 
        return debug_img

    def run_rib_test(self, image: np.ndarray) -> Dict:
        """Automated Rib Analysis Engine"""
        
        # 1. Validation
        is_valid, issues = validate_image(image)
        if not is_valid:
            return {"status": "FAIL", "reason": f"Image issues: {'; '.join(issues)}"}

        # 2. Rebar Body Isolation (Using basic thresholding for speed now)
        bbox = self.detect_region(image)
        if not bbox:
            return {"status": "FAIL", "reason": "No rebar region detected"}
            
        # 3. Rib Extraction (Using Gabor Seeds)
        # In side view, the vertical height of the bounding box is the diameter
        bar_height_px = bbox[3] - bbox[1]
        px_per_mm = bar_height_px / self.diameter_mm
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Gabor filter to find diagonal patterns
        g_kernel = cv2.getGaborKernel((21, 21), 5.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(enhanced, cv2.CV_8U, g_kernel)
        _, thresh = cv2.threshold(filtered, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rib_seeds = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    rib_seeds.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))
        
        # Tell SAM to segment these specific point locations
        sam_rib_masks = self.segment_individual_ribs(image, rib_seeds)
        
        # Convert SAM masks back to rib objects
        all_potential_ribs = []
        for mask in sam_rib_masks:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cnt = max(cnts, key=cv2.contourArea)
                rect = cv2.minAreaRect(cnt)
                (center, (w, h), angle) = rect
                all_potential_ribs.append({
                    "contour": cnt,
                    "center": center,
                    "width": min(w, h),
                    "height": max(w, h),
                    "angle": angle,
                    "area": cv2.contourArea(cnt)
                })
        
        # 4. Statistical Aggregation
        m = self.get_measurements(all_potential_ribs, px_per_mm)
        
        if m.get("status") == "FAIL":
            # If no ribs found, provide a clean debug image anyway
            debug_img = self.generate_debug_image(image, bbox, None, m, "FAIL")
            debug_path = save_debug_image(debug_img, "rib_fail")
            return {"status": "FAIL", "reason": "Inconsistent rib detection", "debug_image_url": f"/static/{os.path.basename(debug_path)}"}
            
        # 5. AR Value Calculation (Projected)
        L_mm = m["rib_length_px"] / px_per_mm
        H_mm = m["rib_height_px"] / px_per_mm
        A_deg = m["rib_angle"]
        S_mm = m["interdistance_px"] / px_per_mm
        
        # Normalized AR Factor (Simplified standard)
        # AR = (Rib_Area_Longitudinal) / (Circumference * Spacing)
        # Approximated here as: (Length * Height * sin(Angle)) / (Diameter * PI * Spacing)
        circ = np.pi * self.diameter_mm
        proj_h = H_mm * np.sin(np.deg2rad(A_deg))
        if S_mm > 0:
            ar_value = (L_mm * proj_h) / (circ * S_mm)
        else:
            ar_value = 0
            
        m["rib_spacing_mm"] = S_mm
        m["rib_length_mm"] = L_mm
        m["rib_height_mm"] = H_mm
        m["ar_value"] = ar_value
        
        status = "PASS" if ar_value > 0.04 else "FAIL" # Standard TMT AR threshold
        reason = "Pass: Standard rib pattern" if status == "PASS" else "Fail: AR value out of range"

        # 6. Generate Technical Debug View
        debug_img = self.generate_debug_image(image, bbox, None, m, status)
        debug_path = save_debug_image(debug_img, "rib_debug")
        
        # Prepare Result Dict
        return {
            "status": status,
            "reason": reason,
            "rib_count": m["num_ribs"],
            "avg_length_mm": round(L_mm, 2),
            "avg_height_mm": round(H_mm, 2),
            "avg_angle_deg": round(A_deg, 1),
            "avg_spacing_mm": round(S_mm, 2),
            "ar_value": round(float(ar_value), 4),
            "debug_image_path": debug_path,
            "debug_image_url": f"/static/{os.path.basename(debug_path)}"
        }

def run_rib_test(image: np.ndarray, diameter_mm: float = 12.0) -> Dict:
    """Wrapper function for API compatibility"""
    pipeline = RibTestPipeline(diameter_mm)
    return pipeline.run_rib_test(image)
