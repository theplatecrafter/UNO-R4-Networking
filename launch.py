import os
from pathlib import Path
import subprocess
import sys
import venv

PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / "venv"

REQUIRED_PACKAGES = {
    "flask": "Flask",
    "cv2": "opencv-python",
    "numpy": "numpy",
}


def venv_python():
    """Return the Python executable inside the project virtual environment."""
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_virtual_environment():
    """Create the local venv if needed and relaunch this script inside it."""
    python_path = venv_python()
    if not python_path.exists():
        print(f"[SYSTEM] Creating virtual environment at {VENV_DIR}...")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)

    if Path(sys.executable).resolve() != python_path.resolve():
        print(f"[SYSTEM] Using virtual environment: {python_path}")
        os.execv(str(python_path), [str(python_path), str(Path(__file__).resolve()), *sys.argv[1:]])


def check_and_install_packages():
    """Install required packages into the active project virtual environment."""
    print("[SYSTEM] Checking dependencies...")
    missing_packages = []
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(pip_name)

    if missing_packages:
        print(f"[SYSTEM] Installing: {', '.join(missing_packages)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])

    print("[SYSTEM] All dependencies verified.\n")


def start_server():
    """Boot the Flask app using paths relative to this launcher."""
    app_path = PROJECT_DIR / "app.py"
    if not app_path.exists():
        print(f"[ERROR] app.py not found beside launch.py: {app_path}")
        sys.exit(1)

    print("[SYSTEM] Booting the Robot Control Center...")
    print("[SYSTEM] DO NOT CLOSE THIS WINDOW while using the robot.")
    print("-" * 50)
    subprocess.run([sys.executable, str(app_path)], cwd=PROJECT_DIR, check=False)

if __name__ == "__main__":
    print("========================================")
    print("    ROBOT VISION TRACKER - LAUNCHER     ")
    print("========================================\n")
    
    ensure_virtual_environment()
    check_and_install_packages()
    start_server()