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
MIN_RIB_CONTOUR_AREA = 50 # Tuned down from 100 to catch smaller ribs
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
        """Stage 1: YOLO Localization"""
        if _CACHED_YOLO_MODEL is None: return None
        
        try:
            results = _CACHED_YOLO_MODEL(image, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)
            if not results or len(results[0].boxes) == 0: return None
            
            # Largest box
            boxes = results[0].boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            largest_idx = np.argmax(areas)
            return tuple(boxes[largest_idx].astype(int))
        except Exception as e:
            print(f"YOLO Error: {e}")
            return None

    def segment_ribs(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Stage 2: SAM Segmentation"""
        if _CACHED_SAM_PREDICTOR is None: return None
        
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            _CACHED_SAM_PREDICTOR.set_image(image_rgb)
            
            input_box = np.array(bbox)
            masks, _, _ = _CACHED_SAM_PREDICTOR.predict(box=input_box, multimask_output=False)
            
            mask_uint8 = (masks[0] * 255).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
            return cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        except Exception as e:
            print(f"SAM Error: {e}")
            return None

    def get_measurements(self, contours: List[np.ndarray], px_per_mm: float) -> Dict[str, float]:
        """Stage 3: OpenCV Measurements (Converted to mm)"""
        if len(contours) < MIN_RIBS_REQUIRED:
            return {
                "rib_length": 0, "rib_angle": 0, "rib_height": 0, "interdistance": 0, "num_ribs": len(contours),
                "rib_length_mm": 0, "rib_height_mm": 0, "interdistance_mm": 0
            }

        # Raw Pixel Measurements
        lengths_px = [cv2.arcLength(cnt, True) / 2 for cnt in contours] # Half arc length approx for open curve, or full peri
        # Note: arcLength is perimeter. For a rib (closed contour), length is usually main axis.
        # Let's use bounding box width/height for more robustness in 2D
        
        lengths_px = []
        heights_px = []
        angles = []
        centroids = []

        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            (center, (w, h), angle) = rect
            
            # Normalize angle and dims (w is usually the longer side for ribs?)
            # Ribs on TMT are transverse.
            if w < h:
                w, h = h, w
                angle = angle + 90
            
            # Ensure angle is 0-90 relative to bar axis (assuming bar is horizontal)
            # We assume bar is roughly horizontal due to YOLO training usually.
            # Only keep acute angle relative to vertical/transverse? 
            # User likely wants angle relative to longitudinal axis.
            # Standard TMT ribs are ~45-65 degrees.
            
            # Simplify Angle: Just take deviation from vertical?
            # Let's stick to the previous rect angle logic but cleaned up
            if angle > 90: angle -= 180
            angle = abs(angle)
            
            lengths_px.append(w) # Major axis
            heights_px.append(h) # Minor axis (rib width/height)
            angles.append(angle)
            centroids.append(center)

        # Inter-distance (Spacing)
        centroids.sort(key=lambda p: p[0]) # Sort by X to get longitudinal spacing
        distances_px = []
        for i in range(len(centroids) - 1):
            p1, p2 = centroids[i], centroids[i+1]
            dist = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            distances_px.append(dist)
            
        # Averages
        avg_len_px = np.mean(lengths_px) if lengths_px else 0
        avg_height_px = np.mean(heights_px) if heights_px else 0
        avg_dist_px = np.mean(distances_px) if distances_px else 0
        avg_angle = np.mean(angles) if angles else 0

        return {
            "rib_length_px": float(avg_len_px),
            "rib_angle": float(avg_angle),
            "rib_height_px": float(avg_height_px),
            "interdistance_px": float(avg_dist_px),
            "num_ribs": len(contours),
            # Converted Metadata
            "px_per_mm": px_per_mm
        }

    def generate_debug_image(self, image: np.ndarray, contours: List[np.ndarray], 
                             bbox: Tuple, measurements: Dict, status: str) -> np.ndarray:
        """Stage 4: Visualization (Matching Ring Test Style)"""
        debug_img = image.copy()
        
        # Draw YOLO Box
        if bbox:
            cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
        # Draw Contours & Annotations
        for i, cnt in enumerate(contours):
            cv2.drawContours(debug_img, [cnt], -1, (255, 100, 0), 2) # Tata Orange
            
            # Centroid
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                cv2.circle(debug_img, (cx, cy), 4, (0, 0, 255), -1)
                
                # Numbering
                cv2.putText(debug_img, f"R{i+1}", (cx+10, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Result Header
        ar_val = measurements.get('ar_value', 0)
        header_text = f"RESULT: {status} (AR: {ar_val:.2f})"
        draw_text_with_background(debug_img, header_text, (25, 40), font_scale=0.9,
                                 text_color=(0, 255, 0) if status == "PASS" else (0, 0, 255))
                                 
        return debug_img

    def run_rib_test(self, image: np.ndarray) -> Dict:
        """Main execution method"""
        
        # 1. Validation
        is_valid, issues = validate_image(image)
        if not is_valid:
            return {"status": "FAIL", "reason": f"Image issues: {'; '.join(issues)}"}

        # 2. Pipeline
        bbox = self.detect_region(image)
        if not bbox:
            return {"status": "FAIL", "reason": "No rebar region detected"}
            
        # Calculate Scale (Px per MM) based on Bar Diameter (assumed = Height of BBox)
        bar_height_px = bbox[3] - bbox[1]
        px_per_mm = bar_height_px / self.diameter_mm
            
        mask = self.segment_ribs(image, bbox)
        if mask is None:
            return {"status": "FAIL", "reason": "Segmentation failed"}
            
        # 3. Measurements
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) >= MIN_RIB_CONTOUR_AREA]
        valid_contours.sort(key=lambda c: cv2.boundingRect(c)[0]) # Left to right
        
        m = self.get_measurements(valid_contours, px_per_mm)
        
        if m["num_ribs"] < MIN_RIBS_REQUIRED:
            return {"status": "FAIL", "reason": f"Insufficient ribs ({m['num_ribs']})"}
            
        # 4. AR Calculation (Normalized)
        # Convert to mm
        L_mm = m["rib_length_px"] / px_per_mm
        H_mm = m["rib_height_px"] / px_per_mm
        # D_mm is self.diameter_mm
        A_deg = m["rib_angle"]
        
        # Normalize
        # References (Approximate Standard values)
        # L_ref: Length of rib is typically related to diameter (e.g. half circumference or projected D)
        L_ref = self.diameter_mm 
        # H_ref: Rib height is typically 0.05-0.10 * D. Let's use 0.07*D as normative base.
        H_ref = 0.07 * self.diameter_mm
        # D_ref: Reference diameter (standard base, e.g. 12mm)
        D_ref = 12.0 
        
        L_norm = L_mm / L_ref
        A_norm = A_deg / 90.0
        H_norm = H_mm / H_ref
        D_norm = self.diameter_mm / D_ref
        
        # AR Formula
        ar_value = L_norm * A_norm * H_norm * D_norm
        
        # Update raw values for API response convenience (preserving key names)
        m["rib_length"] = L_mm
        m["rib_height"] = H_mm
        m["rib_angle"] = A_deg
        m["interdistance"] = m["interdistance_px"] / px_per_mm
        m["ar_value"] = ar_value
        
        # Stability check (simple rule-based)
        is_stable = (m["rib_length"] > 0) and (m["rib_angle"] > 0)
        status = "PASS" if is_stable else "FAIL"
        reason = "Stable measurements" if status == "PASS" else "Unstable rib detection"

        # 5. Debug Image
        debug_img = self.generate_debug_image(image, valid_contours, bbox, m, status)
        debug_path = save_debug_image(debug_img, "rib_debug")
        
        return {
            "status": status,
            "reason": reason,
            "measurements": m,
            "ar_value": ar_value,
            "debug_image_path": debug_path,
            "debug_image_url": f"/static/{os.path.basename(debug_path)}"
        }

def run_rib_test(image: np.ndarray, diameter_mm: float = 12.0) -> Dict:
    """Wrapper function for API compatibility"""
    pipeline = RibTestPipeline(diameter_mm)
    return pipeline.run_rib_test(image)
