"""
Cloudinary Image Upscaling Integration for Ring Test
Uses AI-powered Super Resolution to enhance image quality
"""

import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
import requests
import cv2
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Cloudinary from environment variables
cloudinary.config( 
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"), 
    api_key=os.getenv("CLOUDINARY_API_KEY"), 
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

def upscale_image_cloudinary(image_path: str) -> np.ndarray:
    """
    Upscale an image using Cloudinary's AI Super Resolution
    
    Args:
        image_path: Path to the image file to upscale
        
    Returns:
        Upscaled image as numpy array (BGR format for OpenCV)
    """
    try:
        print("🔄 Uploading image to Cloudinary for AI upscaling...", flush=True)
        
        # Upload the image
        upload_result = cloudinary.uploader.upload(
            image_path,
            folder="tata_steel_upscale",
            resource_type="image"
        )
        
        public_id = upload_result["public_id"]
        print(f"✅ Uploaded. Public ID: {public_id}", flush=True)
        
        # Generate upscaled URL using AI Super Resolution
        # e_upscale: AI-powered upscaling
        # q_auto:best: Best quality
        # f_auto: Auto format
        upscaled_url, _ = cloudinary_url(
            public_id,
            effect="upscale",           # AI Super Resolution
            quality="auto:best",        # Best quality
            fetch_format="auto"         # Auto format
        )
        
        print(f"🎨 Upscaled URL generated: {upscaled_url}", flush=True)
        
        # Download the upscaled image
        print("⬇️  Downloading upscaled image...", flush=True)
        response = requests.get(upscaled_url, timeout=20)
        
        if response.status_code == 200:
            print(f"✅ Downloaded {len(response.content)} bytes", flush=True)
            
            # Decode to OpenCV format
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            upscaled_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if upscaled_image is not None:
                print(f"✨ SUCCESS! Upscaled image shape: {upscaled_image.shape}", flush=True)
                return upscaled_image
            else:
                raise Exception("Failed to decode upscaled image")
        else:
            raise Exception(f"Download failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Cloudinary upscaling failed: {e}", flush=True)
        raise


if __name__ == "__main__":
    # Test the upscaling
    print("="*70)
    print("Testing Cloudinary AI Upscaling")
    print("="*70)
    
    # Test with a sample image
    test_url = "https://res.cloudinary.com/demo/image/upload/sample.jpg"
    
    # Download test image
    resp = requests.get(test_url)
    with open("test_input.jpg", "wb") as f:
        f.write(resp.content)
    
    # Upscale it
    upscaled = upscale_image_cloudinary("test_input.jpg")
    
    # Save result
    cv2.imwrite("test_upscaled.jpg", upscaled)
    print("\n✅ Test complete! Check test_upscaled.jpg")
