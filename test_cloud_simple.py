import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url

cloudinary.config( 
    cloud_name = "dp5antase", 
    api_key = "834477947116472", 
    api_secret = "Pth5br5b9GhkD8x0v7We0uenZ7A",
    secure=True
)

print("Testing Cloudinary upload...")
try:
    result = cloudinary.uploader.upload(
        "https://res.cloudinary.com/demo/image/upload/getting-started/shoes.jpg",
        public_id="ring_test_sample"
    )
    print(f"SUCCESS: {result['secure_url']}")
    
    # Generate enhanced URL
    enhanced, _ = cloudinary_url(
        "ring_test_sample",
        transformation=[
            {'width': 1000, 'crop': 'scale'},
            {'effect': 'sharpen:150'}
        ]
    )
    print(f"ENHANCED: {enhanced}")
    
except Exception as e:
    print(f"ERROR: {e}")
