import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_chippy.db.queries import get_detections_for_day
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
    fact = generate_squirrel_fact() or "No squirrel fact available today."
    title = "Squirrel fact"
    return send_text_notification(title, fact)


def build_daily_activity_report(detections, target_date_str):
    """Summarize how many detections of each animal happened on a given day."""
    counts = Counter(
        detection.get("animal_class")
        for detection in detections
        if detection.get("animal_class")
    )

    title = f"Daily animal activity for {target_date_str}"
    if not counts:
        return f"{title}:\nNo animal detections recorded."

    lines = [title]
    for animal_name, count in sorted(counts.items()):
        lines.append(f"- {animal_name}: {count}")
    return "\n".join(lines)


def send_daily_activity_report(target_date_str=None):
    """Send a notification summarizing detections recorded on the given day."""
    if target_date_str is None:
        target_date_str = datetime.now().strftime("%Y-%m-%d")

    detections = get_detections_for_day(target_date_str)
    report = build_daily_activity_report(detections, target_date_str)
    return send_text_notification("Daily animal activity", report)


if __name__ == "__main__":
    stop_detector_loop()
    # send_squirrel_fact_notification()
    send_daily_activity_report()
    print("Daily summary generated.")