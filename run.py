import uvicorn
import os
import sys
import socket

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

if __name__ == "__main__":
    ip_addr = get_ip()
    port = 8001
    
    print("=" * 60)
    print("🏭 Tata Steel Rebar Testing System")
    print("=" * 60)
    print(f"Starting server on all interfaces...")
    print(f"Local Access:   http://localhost:{port}")
    print(f"Network Access: http://{ip_addr}:{port}")
    print("=" * 60)
    print("📱 TIP: Scan the QR code or type the Network Access URL into your phone's browser.")
    print("Make sure your phone and PC are on the SAME WiFi network.")
    print("=" * 60)
    
    try:
        uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
    except Exception as e:
        print(f"\n❌ Server failed to start: {e}")
        print("\nPossible fixes:")
        print("1. Ensure requirements are installed: pip install -r requirements.txt")
        print(f"2. Check if port {port} is already in use")
        input("\nPress Enter to exit...")
