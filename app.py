"""
FastAPI Application for Tata Steel Rebar Testing
Provides endpoints for both Rib Test and Ring Test
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import cv2
import numpy as np
import os
import shutil
from datetime import datetime

# Import test pipelines
from ring_pipeline import run_ring_test
from cloudinary_upscale import upscale_image_cloudinary

# Initialize FastAPI app
app = FastAPI(
    title="Tata Steel Rebar Testing API",
    description="API for automated TMT bar quality inspection - Rib Test and Ring Test",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

# Create necessary directories
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ========== RESPONSE MODELS ==========

class RingTestResponse(BaseModel):
    """Response model for ring test"""
    status: str
    reason: str
    level1: Optional[Dict] = None
    level2: Optional[Dict] = None
    debug_image_url: Optional[str] = None
    segmented_image_url: Optional[str] = None
    timestamp: str


class ThicknessStandard(BaseModel):
    """Thickness standard for a diameter"""
    diameter_mm: int
    min_thickness_mm: float
    max_thickness_mm: float


# ========== HELPER FUNCTIONS ==========

def save_upload_file(upload_file: UploadFile) -> str:
    """
    Save uploaded file temporarily
    
    Args:
        upload_file: FastAPI UploadFile object
    
    Returns:
        Path to saved file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"upload_{timestamp}_{upload_file.filename}"
    filepath = os.path.join(UPLOADS_DIR, filename)
    
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return filepath


def load_image_from_upload(upload_file: UploadFile) -> np.ndarray:
    """
    Load image from uploaded file
    
    Args:
        upload_file: FastAPI UploadFile object
    
    Returns:
        BGR image as NumPy array
    """
    # Save file temporarily
    filepath = save_upload_file(upload_file)
    
    # Read image
    image = cv2.imread(filepath)
    
    # Clean up temporary file
    try:
        os.remove(filepath)
    except:
        pass
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    return image


# ========== API ENDPOINTS ==========

@app.get("/")
async def root():
    """Serve the home page"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/ring-test")
async def ring_test_page():
    """Serve the ring test module"""
    return FileResponse(os.path.join(STATIC_DIR, "ring_test.html"))


@app.get("/rib-test")
async def rib_test_page():
    """Serve the rib test module"""
    return FileResponse(os.path.join(STATIC_DIR, "rib_test.html"))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/standards")
async def get_thickness_standards():
    """
    Get thickness standards for all TMT bar diameters
    
    Returns:
        List of thickness standards
    """
    from utils import THICKNESS_STANDARDS
    
    standards = []
    for diameter, values in THICKNESS_STANDARDS.items():
        standards.append({
            "diameter_mm": diameter,
            "min_thickness_mm": values["min"],
            "max_thickness_mm": values["max"]
        })
    
    return {
        "standards": standards,
        "unit": "millimeters"
    }


@app.get("/api/standards/{diameter}")
async def get_diameter_standard(diameter: int):
    """
    Get thickness standard for specific diameter
    
    Args:
        diameter: TMT bar diameter in mm
    
    Returns:
        Thickness standard for the diameter
    """
    from utils import get_thickness_standard
    
    try:
        standard = get_thickness_standard(diameter)
        return {
            "diameter_mm": diameter,
            "min_thickness_mm": standard["min"],
            "max_thickness_mm": standard["max"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))




@app.post("/api/ring-test", response_model=RingTestResponse)
async def ring_test_endpoint(
    file: UploadFile = File(..., description="Image file of TMT bar cross-section"),
    diameter: float = Form(..., description="TMT bar diameter in mm (any positive value)"),
    upscale: bool = Form(False, description="Use Cloudinary AI upscaling"),
    edge_segment: bool = Form(False, description="Use edge-based segmentation")
):
    """
    Perform ring test on uploaded image
    Based on official Tata Steel TM-Ring test specification
    
    Args:
        file: Image file (JPEG, PNG)
        diameter: TMT bar diameter in mm (any positive value)
        upscale: Whether to use Cloudinary AI Super Resolution upscaling
        edge_segment: Whether to use edge-based background removal
    
    Returns:
        Ring test results with status, measurements, and debug image
    """
    # Validate diameter
    if diameter <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid diameter: {diameter}. Must be positive"
        )
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    
    try:
        if upscale:
            # Save uploaded file temporarily for Cloudinary
            temp_path = save_upload_file(file)
            print(f"📷 Upscaling enabled. Temp file: {temp_path}", flush=True)
            
            try:
                # Upscale using Cloudinary AI
                image = upscale_image_cloudinary(temp_path)
                
                # Save upscaled image for verification
                upscaled_debug_path = os.path.join(STATIC_DIR, "debug_upscaled.jpg")
                cv2.imwrite(upscaled_debug_path, image)
                print(f"💾 Upscaled image saved: {upscaled_debug_path}", flush=True)
                
            except Exception as upscale_error:
                print(f"⚠️  Upscaling failed, using original: {upscale_error}", flush=True)
                image = cv2.imread(temp_path)
            finally:
                # Clean up temp file
                try:
                    os.remove(temp_path)
                except:
                    pass
        else:
            # Standard flow - load image directly
            image = load_image_from_upload(file)
        
        # Run ring test pipeline with edge segmentation option
        results = run_ring_test(image, diameter_mm=diameter, use_edge_segment=edge_segment)
        
        # Prepare response
        debug_image_url = results.get("debug_image_url")
        if not debug_image_url and results.get("debug_image_path"):
            debug_image_url = f"/static/{os.path.basename(results['debug_image_path'])}"
        
        # Add segmented image URL if edge segmentation was used
        segmented_image_url = None
        if edge_segment and os.path.exists(os.path.join(STATIC_DIR, "debug_edge_segmented.jpg")):
            # Add timestamp to prevent caching
            timestamp = int(datetime.now().timestamp() * 1000)
            segmented_image_url = f"/static/debug_edge_segmented.jpg?t={timestamp}"
        
        response = RingTestResponse(
            status=results["status"],
            reason=results["reason"],
            level1=results.get("level1"),
            level2=results.get("level2"),
            debug_image_url=debug_image_url,
            segmented_image_url=segmented_image_url,
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/api/rib-test")
async def rib_test_endpoint(
    file: UploadFile = File(..., description="Image file of TMT bar"),
    diameter: float = Form(..., description="TMT bar diameter in mm")
):
    """
    Perform rib test on uploaded image.
    Standalone module for longitudinal rebar analysis.
    """
    try:
        # Validate file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Load image
        image = load_image_from_upload(file)
        
        # Run Rib Pipeline (New Architecture Placeholder)
        from rib_pipeline import run_rib_test
        results = run_rib_test(image, diameter_mm=diameter)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rib Test Backend Error: {str(e)}")


@app.get("/static/{filename}")
async def get_debug_image(filename: str):
    """
    Serve debug images
    
    Args:
        filename: Debug image filename
    
    Returns:
        Image file
    """
    filepath = os.path.join(STATIC_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(filepath)


# ========== CLEANUP ENDPOINT (Optional) ==========

@app.delete("/api/cleanup")
async def cleanup_old_files(max_age_hours: int = 24):
    """
    Clean up old debug images and uploads
    
    Args:
        max_age_hours: Maximum age of files to keep (default: 24 hours)
    
    Returns:
        Cleanup statistics
    """
    import time
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    deleted_count = 0
    
    # Clean static directory
    for filename in os.listdir(STATIC_DIR):
        filepath = os.path.join(STATIC_DIR, filename)
        if os.path.isfile(filepath):
            file_age = current_time - os.path.getmtime(filepath)
            if file_age > max_age_seconds:
                os.remove(filepath)
                deleted_count += 1
    
    # Clean uploads directory
    for filename in os.listdir(UPLOADS_DIR):
        filepath = os.path.join(UPLOADS_DIR, filename)
        if os.path.isfile(filepath):
            file_age = current_time - os.path.getmtime(filepath)
            if file_age > max_age_seconds:
                os.remove(filepath)
                deleted_count += 1
    
    return {
        "deleted_files": deleted_count,
        "max_age_hours": max_age_hours,
        "timestamp": datetime.now().isoformat()
    }


# ========== RUN SERVER ==========

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🏭 Tata Steel Rebar Testing API")
    print("=" * 60)
    print("Starting server on http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
