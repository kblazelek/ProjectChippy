import cv2
import requests
from project_chippy.core.config import TOPIC


def sanitize_header_value(value):
    """Convert a header value to ASCII-safe text without line breaks or control characters."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")

    text = str(value).encode("ascii", errors="replace").decode("ascii")
    return " ".join(text.split())


def send_text_notification(title, message):
    """Send a plain-text notification to the configured ntfy topic."""
    notification_url = f"https://ntfy.sh/{TOPIC}"
    headers = {
        "Title": sanitize_header_value(title),
    }

    try:
        response = requests.post(
            notification_url,
            data=message.encode("utf-8"),
            headers=headers,
            timeout=10,
        )

        if response.status_code in (200, 201, 202):
            print(f"🔔 Text notification sent to {notification_url}")
            return True
        print(f"⚠️ Text notification failed ({response.status_code}): {response.text}")
        return False
    except Exception as exc:
        print(f"⚠️ Text notification error: {exc}")
        return False


def send_wildlife_notification(annotated_frame, detected_labels, timestamp):
    """
    Encodes the image and sends a push notification via ntfy.sh.
    Returns True if the notification was sent successfully, False otherwise.
    """
    notification_url = f"https://ntfy.sh/{TOPIC}"
    
    # Format the labels (e.g., "squirrel, pigeon")
    unique_labels = ", ".join(sorted(set(detected_labels)))
    message = f"Detected animals: {unique_labels} at {timestamp}"
    
    try:
        # Convert the raw OpenCV frame into a compressed JPEG byte array
        _, encoded_image = cv2.imencode(".jpg", annotated_frame)

        headers = {
            "Title": sanitize_header_value("Wildlife update"),
            "Filename": sanitize_header_value("animal.jpg"),
            "Message": sanitize_header_value(message),
        }

        response = requests.put(
            notification_url,
            data=encoded_image.tobytes(),
            headers=headers,
            timeout=10,
        )

        if response.status_code in (200, 201, 202):
            print(f"🔔 Notification sent to {notification_url}")
            return True
        else:
            print(f"⚠️ Notification failed ({response.status_code}): {response.text}")
            return False

    except Exception as exc:
        print(f"⚠️ Notification error: {exc}")
        return False