import cv2
import requests
from project_chippy.core.config import TOPIC

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
            "Title": "Wildlife update",
            "Filename": "animal.jpg",
            "Message": message,
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