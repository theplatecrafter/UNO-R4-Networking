import sys
import subprocess
import time
import os

def check_and_install_packages():
    """Checks for required packages and installs them if missing."""
    required_packages = {
        'flask': 'Flask',
        'cv2': 'opencv-python',
        'numpy': 'numpy'
    }
    
    print("[SYSTEM] Checking dependencies...")
    for import_name, pip_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            print(f"[SYSTEM] Missing '{pip_name}'. Installing now (this might take a minute)...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
                print(f"[SYSTEM] Successfully installed {pip_name}!")
            except Exception as e:
                print(f"[ERROR] Failed to install {pip_name}. You may need to run: pip install {pip_name}")
                print(e)
                sys.exit(1)
    
    print("[SYSTEM] All dependencies verified.\n")

def start_server():
    """Boots the Flask app."""
    if not os.path.exists("app.py"):
        print("[ERROR] app.py not found in the current directory!")
        print("Please ensure launch.py and app.py are in the same folder.")
        input("Press Enter to exit...")
        sys.exit(1)

    print("[SYSTEM] Booting the Robot Control Center...")
    print("[SYSTEM] DO NOT CLOSE THIS WINDOW while using the robot.")
    print("-" * 50)
    
    # Run app.py
    subprocess.run([sys.executable, "app.py"])

if __name__ == "__main__":
    print("========================================")
    print("    ROBOT VISION TRACKER - LAUNCHER     ")
    print("========================================\n")
    
    check_and_install_packages()
    start_server()