import sys
import os

print("Checking environment...")

def check_import(module_name):
    try:
        __import__(module_name)
        print(f"✅ {module_name} available")
        return True
    except ImportError as e:
        print(f"❌ {module_name} MISSING: {e}")
        return False
    except Exception as e:
        print(f"❌ {module_name} ERROR: {e}")
        return False

required_modules = [
    "fastapi", "uvicorn", "cv2", "numpy", "PIL", "python_multipart",
    "ultralytics", "segment_anything", "torch"
]

all_good = True
for mod in required_modules:
    if not check_import(mod):
        all_good = False

if all_good:
    print("\n✨ All dependencies look good!")
    print("\nTry running the server with:")
    print("python run.py")
else:
    print("\n⚠️  Some dependencies are missing or broken.")
    print("Please run: pip install -r requirements.txt")
