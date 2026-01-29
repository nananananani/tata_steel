import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url

# Configuration       
cloudinary.config( 
    cloud_name = "dp5antase", 
    api_key = "834477947116472", 
    api_secret = "Pth5br5b9GhkD8x0v7We0uenZ7A",  # Using your actual secret
    secure=True
)

print("=" * 60)
print("Testing Cloudinary Configuration")
print("=" * 60)

# Test 1: Upload an image from URL
print("\n1. Testing upload from URL...")
try:
    upload_result = cloudinary.uploader.upload(
        "https://res.cloudinary.com/demo/image/upload/getting-started/shoes.jpg",
        public_id="test_shoes"
    )
    print(f"✅ Upload successful!")
    print(f"   Secure URL: {upload_result['secure_url']}")
except Exception as e:
    print(f"❌ Upload failed: {e}")

# Test 2: Optimize delivery
print("\n2. Testing optimization (auto-format, auto-quality)...")
try:
    optimize_url, _ = cloudinary_url("test_shoes", fetch_format="auto", quality="auto")
    print(f"✅ Optimized URL: {optimize_url}")
except Exception as e:
    print(f"❌ Optimization failed: {e}")

# Test 3: Enhanced transformation (for Ring Test)
print("\n3. Testing CV enhancement (sharpen + upscale)...")
try:
    enhance_url, _ = cloudinary_url(
        "test_shoes",
        transformation=[
            {'width': 1000, 'crop': 'scale'},
            {'effect': 'sharpen:150'},
            {'quality': 'auto:best'},
            {'fetch_format': 'png'}
        ]
    )
    print(f"✅ Enhanced URL: {enhance_url}")
except Exception as e:
    print(f"❌ Enhancement failed: {e}")

print("\n" + "=" * 60)
print("Cloudinary test complete!")
print("=" * 60)
