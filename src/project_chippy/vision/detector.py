import os
import threading
import time
from datetime import datetime

import cv2
from PIL import Image

try:
    from picamera2 import Picamera2
except Exception:  # pragma: no cover - optional dependency on non-Pi systems
    Picamera2 = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency on non-Pi systems
    YOLO = None

from project_chippy.core.config import (
    COOLDOWN_SECONDS,
    MODEL_PATH,
    NOTIFICATION_COOLDOWN_SECONDS,
    RECORDING_BUFFER_SIZE,
    RECORDING_COOLDOWN_SECONDS,
    SAVE_DIR,
)
from project_chippy.db.queries import save_detection
from project_chippy.notifications.ntfy import send_image_notification, send_wildlife_notification


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


def should_trigger_recording(current_label, previous_label, elapsed_since_last_recording, last_recording_time, cooldown_seconds):
    """Return True when two matching detections should start a recording burst."""
    del last_recording_time
    return bool(
        current_label
        and previous_label
        and current_label == previous_label
        and elapsed_since_last_recording >= cooldown_seconds
    )


def build_recording_buffer(buffer, frame, target_size=RECORDING_BUFFER_SIZE):
    """Append a frame to the recording buffer while keeping it bounded."""
    updated = list(buffer)
    updated.append(frame)
    if len(updated) > target_size:
        updated = updated[-target_size:]
    return updated


def process_recording_buffer(frames, timestamp):
    """Create a GIF from a buffered recording in a background thread."""
    if not frames:
        return

    output_path = os.path.join(SAVE_DIR, f"recording_{timestamp}.gif")
    try:
        images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames if frame is not None]
        if not images:
            return

        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=200,
            loop=0,
        )
        with open(output_path, "rb") as gif_file:
            gif_bytes = gif_file.read()

        send_image_notification(
            title="Wildlife recording",
            image_bytes=gif_bytes,
            filename=f"recording_{timestamp}.gif",
            message=f"Recording captured at {timestamp}",
        )
        print(f"🎞️ SUCCESS: Wrote recording GIF to {output_path}")
    except Exception as exc:
        print(f"GIF creation failed: {exc}")


def run_detection_loop(state):
    """
    Main camera and YOLO detection loop.
    Reads from the camera, runs inference, updates the shared state,
    and handles saving/notifications based on cooldowns.
    """
    picam2 = None
    try:
        if Picamera2 is None:
            raise RuntimeError("picamera2 is not available in this environment")

        picam2 = Picamera2()
        config = picam2.create_video_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        picam2.start()

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        if YOLO is None:
            raise RuntimeError("ultralytics is not available in this environment")

        model = YOLO(MODEL_PATH, task="detect")
        print(f"Starting wildlife detection with model: {MODEL_PATH}")

        while not state.detector_stop_event.is_set():
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            current_time = time.time()

            if state.recording_mode_active:
                state.recording_buffer = build_recording_buffer(state.recording_buffer, frame)
                with state.frame_lock:
                    state.latest_frame = frame.copy()

                if len(state.recording_buffer) >= RECORDING_BUFFER_SIZE:
                    recording_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    recording_frames = list(state.recording_buffer)
                    state.recording_buffer = []
                    state.recording_mode_active = False
                    state.last_recording_time = current_time
                    recording_thread = threading.Thread(
                        target=process_recording_buffer,
                        args=(recording_frames, recording_timestamp),
                        daemon=True,
                        name="recording-processor",
                    )
                    recording_thread.start()

                time.sleep(0.1)
                continue

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

                if float(confidence) >= state.save_confidence_threshold:
                    saveable_detection = True

            annotated_frame = result.plot()

            with state.frame_lock:
                state.latest_frame = annotated_frame.copy()

            timestamp = None
            if animal_detected:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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

            if animal_detected and (current_time - state.last_notification_time) > NOTIFICATION_COOLDOWN_SECONDS:
                state.last_notification_time = current_time
                notification_thread = threading.Thread(
                    target=send_wildlife_notification,
                    args=(annotated_frame.copy(), detected_labels, timestamp),
                    daemon=True,
                )
                notification_thread.start()

            if animal_detected and (current_time - state.last_recording_time) > RECORDING_COOLDOWN_SECONDS:
                if state.pending_detection_label is None:
                    state.pending_detection_label = detected_labels[0] if detected_labels else None
                    state.pending_detection_frame = frame.copy()
                else:
                    current_label = detected_labels[0] if detected_labels else None
                    if should_trigger_recording(
                        current_label,
                        state.pending_detection_label,
                        current_time - state.last_recording_time,
                        state.last_recording_time,
                        RECORDING_COOLDOWN_SECONDS,
                    ):
                        state.recording_buffer = [state.pending_detection_frame, frame.copy()]
                        state.recording_mode_active = True
                        state.pending_detection_label = None
                        state.pending_detection_frame = None
                    else:
                        state.pending_detection_label = current_label
                        state.pending_detection_frame = frame.copy()

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping script...")
    finally:
        state.detector_enabled = False
        if picam2 is not None:
            try:
                picam2.stop()
                picam2.close()
                print("Camera resources released successfully.")
            except Exception as exc:
                print(f"Error closing camera: {exc}")