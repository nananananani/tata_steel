import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
import requests
import cv2
import numpy as np
import sys

# Configure Cloudinary
cloudinary.config( 
    cloud_name = "dp5antase", 
    api_key = "834477947116472", 
    api_secret = "Pth5br5b9GhkD8x0v7We0uenZ7A",
    secure=True
)

# Get image path from command line or use a test image
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    # Use a sample image URL
    image_path = "https://res.cloudinary.com/demo/image/upload/getting-started/shoes.jpg"

print("=" * 70)
print("🔄 CLOUDINARY IMAGE ENHANCEMENT TEST")
print("=" * 70)

# Step 1: Upload original image
print(f"\n1️⃣  Uploading original image: {image_path}")
try:
    upload_result = cloudinary.uploader.upload(image_path, public_id="ring_test_demo")
    original_url = upload_result["secure_url"]
    print(f"✅ Original uploaded: {original_url}")
except Exception as e:
    print(f"❌ Upload failed: {e}")
    sys.exit(1)

# Step 2: Generate enhanced URL with transformations
print(f"\n2️⃣  Generating enhanced URL with transformations...")
enhance_url, _ = cloudinary_url(
    "ring_test_demo",
    transformation=[
        {'width': 1000, 'crop': 'scale'},       # Upscale to 1000px
        {'effect': 'sharpen:150'},              # Strong sharpening
        {'quality': 'auto:best'},               # Best quality
        {'fetch_format': 'png'}                 # PNG format
    ]
)
print(f"✅ Enhanced URL: {enhance_url}")

# Step 3: Download both images
print(f"\n3️⃣  Downloading images...")

# Download original
print("   Downloading original...")
resp_orig = requests.get(original_url)
if resp_orig.status_code == 200:
    img_orig = np.asarray(bytearray(resp_orig.content), dtype=np.uint8)
    original_image = cv2.imdecode(img_orig, cv2.IMREAD_COLOR)
    cv2.imwrite("static/original_image.jpg", original_image)
    print(f"   ✅ Original saved: static/original_image.jpg | Shape: {original_image.shape}")
else:
    print(f"   ❌ Failed to download original")

# Download enhanced
print("   Downloading enhanced...")
resp_enh = requests.get(enhance_url)
if resp_enh.status_code == 200:
    img_enh = np.asarray(bytearray(resp_enh.content), dtype=np.uint8)
    enhanced_image = cv2.imdecode(img_enh, cv2.IMREAD_COLOR)
    cv2.imwrite("static/enhanced_image.png", enhanced_image)
    print(f"   ✅ Enhanced saved: static/enhanced_image.png | Shape: {enhanced_image.shape}")
else:
    print(f"   ❌ Failed to download enhanced")

# Step 4: Create side-by-side comparison
if original_image is not None and enhanced_image is not None:
    print(f"\n4️⃣  Creating comparison image...")
    
    # Resize to same height for comparison
    h1, w1 = original_image.shape[:2]
    h2, w2 = enhanced_image.shape[:2]
    target_h = min(h1, h2, 600)  # Max height 600px
    
    scale1 = target_h / h1
    scale2 = target_h / h2
    
    orig_resized = cv2.resize(original_image, (int(w1 * scale1), target_h))
    enh_resized = cv2.resize(enhanced_image, (int(w2 * scale2), target_h))
    
    # Combine side by side
    comparison = np.hstack([orig_resized, enh_resized])
    
    # Add labels
    cv2.putText(comparison, "ORIGINAL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(comparison, "ENHANCED (Cloudinary)", (orig_resized.shape[1] + 20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imwrite("static/comparison.jpg", comparison)
    print(f"   ✅ Comparison saved: static/comparison.jpg")

print("\n" + "=" * 70)
print("✨ ENHANCEMENT TEST COMPLETE!")
print("=" * 70)
print("\nFiles saved:")
print("  📁 static/original_image.jpg   - Original image")
print("  📁 static/enhanced_image.png   - Cloudinary enhanced")  
print("  📁 static/comparison.jpg       - Side-by-side comparison")
print("\nOpen these files to see the enhancement effect!")
print("=" * 70)
