import os
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_chippy.notifications.ntfy import send_text_notification


def stop_detector_loop():
    """Stop the detector loop by posting to the Flask config endpoint."""
    base_url = os.getenv("PROJECT_CHIPPY_BASE_URL", "http://127.0.0.1:5000").rstrip("/")
    endpoint = f"{base_url}/config/detector"

    try:
        response = requests.post(endpoint, data={"action": "stop"}, timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Could not stop detector loop: {exc}")
        return False

    print("Detector loop stop request sent.")
    return True


def generate_squirrel_fact():
    """Generate a short, interesting squirrel fact using a local Ollama model."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "gemma4:e4b",
        "prompt": "Give me one interesting fact about squirrels in one sentence.",
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        fact = (data.get("response") or "").strip()
        if fact:
            return fact
    except requests.RequestException as exc:
        print(f"Could not generate squirrel fact: {exc}")
        pass


def send_squirrel_fact_notification():
    """Generate a squirrel fact and send it through ntfy."""
    fact = generate_squirrel_fact()
    title = "Squirrel fact"
    return send_text_notification(title, fact)


if __name__ == "__main__":
    stop_detector_loop()
    send_squirrel_fact_notification()
    print("Daily summary generated.")