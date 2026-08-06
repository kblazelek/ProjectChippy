import os
import threading
import time
from datetime import datetime
import cv2
import requests
from ultralytics import YOLO
from picamera2 import Picamera2
from project_chippy.db.queries import save_detection
from project_chippy.notifications.ntfy import send_wildlife_notification
from project_chippy.core.config import (
    MODEL_PATH,
    SAVE_DIR,
    COOLDOWN_SECONDS,
    NOTIFICATION_COOLDOWN_SECONDS
)

def build_detection_payload(detected_classes, detected_confidences, names_dict, boxes):
    """Convert YOLO results into the shape expected by save_detection."""
    payload = []
    for index, class_id in enumerate(detected_classes):
        confidence = detected_confidences[index] if index < len(detected_confidences) else 1.0
        bbox = []
        if boxes is not None and index < len(boxes):
            bbox = [float(value) for value in boxes[index]]

        payload.append(
            {
                "class": names_dict[int(class_id)],
                "confidence": float(confidence),
                "bbox": bbox,
            }
        )
    return payload


def persist_detection(image_path, detections):
    """Persist a saved detection image and its detections to the database."""
    try:
        save_detection(image_path, detections)
    except Exception as exc:
        print(f"Database save failed: {exc}")


def run_detection_loop(state):
    """
    Main camera and YOLO detection loop.
    Reads from the camera, runs inference, updates the shared state,
    and handles saving/notifications based on cooldowns.
    """
    picam2 = None
    try:
        picam2 = Picamera2()
        config = picam2.create_video_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        picam2.start()

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        model = YOLO(MODEL_PATH, task="detect")
        print(f"Starting wildlife detection with model: {MODEL_PATH}")

        while not state.detector_stop_event.is_set():
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = model(frame, stream=True, conf=state.model_confidence_threshold)
            result = next(results, None)

            if result is None:
                continue

            names_dict = result.names
            boxes = result.boxes
            detected_classes = boxes.cls.cpu().numpy() if boxes is not None else []
            detected_confidences = boxes.conf.cpu().numpy() if boxes is not None and boxes.conf is not None else []

            animal_detected = len(detected_classes) > 0
            detected_labels = []
            saveable_detection = False
            detection_payload = build_detection_payload(
                detected_classes,
                detected_confidences,
                names_dict,
                boxes.xyxy if boxes is not None else None,
            )

            for detection in detection_payload:
                label = detection["class"]
                confidence = detection["confidence"]
                print(f"Detected: {label} (conf={float(confidence):.2f})")
                detected_labels.append(label)

                # Check against the dynamic save threshold in the shared state
                if float(confidence) >= state.save_confidence_threshold:
                    saveable_detection = True

            annotated_frame = result.plot()
            
            # --- SAFELY UPDATE THE SHARED FRAME ---
            with state.frame_lock:
                state.latest_frame = annotated_frame.copy()

            current_time = time.time()
            timestamp = None
            if animal_detected:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # --- HANDLE SAVING (Using state.last_save_time) ---
            if saveable_detection and (current_time - state.last_save_time) > COOLDOWN_SECONDS:
                raw_filename = f"animal_{timestamp}_raw.jpg"
                raw_filepath = os.path.join(SAVE_DIR, raw_filename)
                cv2.imwrite(raw_filepath, frame)
                print(f"📸 SUCCESS: Saved {raw_filename} to {SAVE_DIR}/")

                annotated_filename = f"animal_{timestamp}_annotated.jpg"
                annotated_filepath = os.path.join(SAVE_DIR, annotated_filename)
                cv2.imwrite(annotated_filepath, annotated_frame)
                print(f"📸 SUCCESS: Saved {annotated_filename} to {SAVE_DIR}/")

                persist_detection(raw_filepath, detection_payload)
                state.last_save_time = current_time

            # --- HANDLE NOTIFICATIONS
            if animal_detected and (current_time - state.last_notification_time) > NOTIFICATION_COOLDOWN_SECONDS:
                state.last_notification_time = current_time
                notification_thread = threading.Thread(
                    target=send_wildlife_notification,
                    args=(annotated_frame.copy(), detected_labels, timestamp),
                    daemon=True,
                )
                notification_thread.start()

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping script...")
    finally:
        state.detector_enabled = False
        if picam2 is not None:
            try:
                picam2.stop()   # Stops the capture stream
                picam2.close()  # RELEASES THE HARDWARE
                print("Camera resources released successfully.")
            except Exception as e:
                print(f"Error closing camera: {e}")