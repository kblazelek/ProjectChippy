import os
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def start_detector_loop():
    """Start the detector loop by posting to the Flask config endpoint."""
    base_url = os.getenv("PROJECT_CHIPPY_BASE_URL", "http://127.0.0.1:5000").rstrip("/")
    endpoint = f"{base_url}/config/detector"

    try:
        response = requests.post(endpoint, data={"action": "start"}, timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Could not start detector loop: {exc}")
        return False

    print("Detector loop start request sent.")
    return True


if __name__ == "__main__":
    start_detector_loop()
