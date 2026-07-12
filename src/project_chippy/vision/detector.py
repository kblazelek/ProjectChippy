import os
import time
from datetime import datetime
import cv2
import requests
from ultralytics import YOLO
from picamera2 import Picamera2
from project_chippy.notifications.ntfy import send_wildlife_notification
from project_chippy.core.config import (
    MODEL_PATH,
    SAVE_DIR,
    COOLDOWN_SECONDS,
    NOTIFICATION_COOLDOWN_SECONDS
)

def run_detection_loop(state):
    """
    Main camera and YOLO detection loop.
    Reads from the camera, runs inference, updates the shared state,
    and handles saving/notifications based on cooldowns.
    """
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    model = YOLO(MODEL_PATH, task="detect")
    print(f"Starting wildlife detection with model: {MODEL_PATH}")

    try:
        while True:
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

            for class_id, confidence in zip(
                detected_classes,
                detected_confidences if len(detected_confidences) > 0 else [1.0] * len(detected_classes),
            ):
                label = names_dict[int(class_id)]
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

                state.last_save_time = current_time

            # --- HANDLE NOTIFICATIONS
            if animal_detected and (current_time - state.last_notification_time) > NOTIFICATION_COOLDOWN_SECONDS:
                notification_sent = send_wildlife_notification(
                    annotated_frame=annotated_frame, 
                    detected_labels=detected_labels, 
                    timestamp=timestamp
                )
                
                # Only reset the cooldown timer if it successfully sent
                if notification_sent:
                    state.last_notification_time = current_time

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping script...")
    finally:
        try:
            picam2.stop()
        except Exception:
            pass