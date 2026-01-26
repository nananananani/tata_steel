import uvicorn
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("=" * 60)
    print("🏭 Tata Steel Rebar Testing System")
    print("=" * 60)
    print("Starting server...")
    print("If successful, open your browser to: http://127.0.0.1:8000")
    print("=" * 60)
    
    try:
        uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
    except Exception as e:
        print(f"\n❌ Server failed to start: {e}")
        print("\nPossible fixes:")
        print("1. Ensure requirements are installed: pip install -r requirements.txt")
        print("2. Check if port 8000 is already in use")
        input("\nPress Enter to exit...")
