from ultralytics import YOLO
import cv2
from picamera2 import Picamera2
import os
import time
from datetime import datetime
import requests
from flask import Flask, Response, send_from_directory, request, jsonify
import threading

# --- Setup the save directory ---
topic = "coffice-animal-update"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAVE_DIR = os.path.join(BASE_DIR, "data", "captures")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.onnx")
os.makedirs(SAVE_DIR, exist_ok=True)  # Creates the folder if it doesn't exist

# --- Setup a cooldown timer (in seconds) ---
COOLDOWN_SECONDS = 5.0
NOTIFICATION_COOLDOWN_SECONDS = 30 * 60
last_save_time = 0.0
last_notification_time = 0.0

app = Flask(__name__)
latest_frame = None
frame_lock = threading.Lock()
MODEL_CONFIDENCE_THRESHOLD = 0.6
SAVE_CONFIDENCE_THRESHOLD = MODEL_CONFIDENCE_THRESHOLD


def set_save_confidence_threshold(value):
    global SAVE_CONFIDENCE_THRESHOLD
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Threshold must be a number between 0.0 and 1.0") from exc

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0")

    SAVE_CONFIDENCE_THRESHOLD = threshold
    return SAVE_CONFIDENCE_THRESHOLD


def get_save_confidence_threshold():
    return SAVE_CONFIDENCE_THRESHOLD


def update_latest_frame(frame):
    global latest_frame
    with frame_lock:
        latest_frame = frame.copy() if frame is not None else None


def get_recent_detection_files(limit=20):
    detection_files = []
    if not os.path.isdir(SAVE_DIR):
        return []

    for filename in os.listdir(SAVE_DIR):
        if filename.endswith("_annotated.jpg"):
            full_path = os.path.join(SAVE_DIR, filename)
            if os.path.isfile(full_path):
                detection_files.append((os.path.getmtime(full_path), filename))

    detection_files.sort(key=lambda item: item[0], reverse=True)
    return [filename for _, filename in detection_files[:limit]]


@app.route("/")
def index():
    return """
    <!doctype html>
    <html>
    <head><title>Wildlife Camera</title></head>
    <body>
        <h1>Wildlife Camera</h1>
        <p><a href="/detections">View last 20 detections</a></p>
        <p><a href="/config">Configure save threshold</a></p>
        <img src="/video_feed" alt="Camera feed" style="max-width: 100%;">
    </body>
    </html>
    """


@app.route("/detections")
def detections():
    recent_files = get_recent_detection_files(20)

    if not recent_files:
        items_html = "<p>No detections yet.</p>"
    else:
        items = []
        for filename in recent_files:
            image_url = f"/detections/{filename}"
            items.append(
                f"<li><a href=\"{image_url}\">{filename}</a><br>"
                f"<img src=\"{image_url}\" alt=\"{filename}\" style=\"max-width:320px; margin-top:8px;\"></li>"
            )
        items_html = "<ul>" + "".join(items) + "</ul>"

    return f"""
    <!doctype html>
    <html>
    <head><title>Last 20 Detections</title></head>
    <body>
        <h1>Last 20 detections</h1>
        <p><a href="/">Back to live view</a></p>
        {items_html}
    </body>
    </html>
    """


@app.route("/detections/<path:filename>")
def serve_detection(filename):
    safe_name = os.path.basename(filename)
    return send_from_directory(SAVE_DIR, safe_name)


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with frame_lock:
                frame_to_send = latest_frame
            if frame_to_send is None:
                time.sleep(0.1)
                continue

            _, encoded_image = cv2.imencode(".jpg", frame_to_send)
            frame_bytes = encoded_image.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.05)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/config", methods=["GET", "POST"])
def config_page():
    current_threshold = get_save_confidence_threshold()
    message = None
    error = None

    if request.method == "POST":
        threshold_value = request.form.get("threshold")
        if threshold_value is None or str(threshold_value).strip() == "":
            error = "Please enter a threshold value."
        else:
            try:
                current_threshold = set_save_confidence_threshold(threshold_value)
                message = f"Save threshold updated to {current_threshold:.2f}"
            except ValueError as exc:
                error = str(exc)

    return f"""
    <!doctype html>
    <html>
    <head><title>Configuration</title></head>
    <body>
        <h1>Configuration</h1>
        <p><a href="/">Back to live view</a></p>
        <p>Model threshold: {MODEL_CONFIDENCE_THRESHOLD:.2f}</p>
        <p>Current save threshold: {current_threshold:.2f}</p>
        {f"<p style='color: green;'>{message}</p>" if message else ""}
        {f"<p style='color: red;'>{error}</p>" if error else ""}
        <form method="post">
            <label for="threshold">Save confidence threshold</label>
            <input type="number" id="threshold" name="threshold" min="0" max="1" step="0.01" value="{current_threshold:.2f}" required>
            <button type="submit">Save</button>
        </form>
        <p><a href="/config/save_confidence">View JSON config</a></p>
    </body>
    </html>
    """


@app.route("/config/save_confidence", methods=["GET", "POST"])
def save_confidence_config():
    if request.method == "POST":
        threshold_value = request.args.get("threshold")
        if threshold_value is None:
            payload = request.get_json(silent=True) or {}
            threshold_value = payload.get("threshold")

        if threshold_value is None:
            return jsonify({
                "error": "Provide a numeric 'threshold' value between 0.0 and 1.0",
            }), 400

        try:
            current_threshold = set_save_confidence_threshold(threshold_value)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify({
            "save_confidence_threshold": current_threshold,
            "model_confidence_threshold": MODEL_CONFIDENCE_THRESHOLD,
            "message": "Save confidence threshold updated",
        })

    return jsonify({
        "save_confidence_threshold": get_save_confidence_threshold(),
        "model_confidence_threshold": MODEL_CONFIDENCE_THRESHOLD,
    })


def run_detection_loop():
    global last_save_time, last_notification_time

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

            # Run inference with a 0.6 confidence threshold (kept unchanged for model filtering)
            results = model(frame, stream=True, conf=MODEL_CONFIDENCE_THRESHOLD)
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
                if float(confidence) >= SAVE_CONFIDENCE_THRESHOLD:
                    saveable_detection = True

            annotated_frame = result.plot()
            update_latest_frame(annotated_frame)

            current_time = time.time()
            timestamp = None
            if animal_detected:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if saveable_detection and (current_time - last_save_time) > COOLDOWN_SECONDS:
                raw_filename = f"animal_{timestamp}_raw.jpg"
                raw_filepath = os.path.join(SAVE_DIR, raw_filename)
                cv2.imwrite(raw_filepath, frame)
                print(f"📸 SUCCESS: Saved {raw_filename} to {SAVE_DIR}/")

                annotated_filename = f"animal_{timestamp}_annotated.jpg"
                annotated_filepath = os.path.join(SAVE_DIR, annotated_filename)
                cv2.imwrite(annotated_filepath, annotated_frame)
                print(f"📸 SUCCESS: Saved {annotated_filename} to {SAVE_DIR}/")

                last_save_time = current_time

            if animal_detected and (current_time - last_notification_time) > NOTIFICATION_COOLDOWN_SECONDS:
                notification_url = f"https://ntfy.sh/{topic}"
                message = f"Detected animals: {', '.join(sorted(set(detected_labels)))} at {timestamp}"
                try:
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
                        last_notification_time = current_time
                    else:
                        print(f"⚠️ Notification failed ({response.status_code}): {response.text}")
                except Exception as exc:
                    print(f"⚠️ Notification error: {exc}")

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping script...")
    finally:
        try:
            picam2.stop()
        except Exception:
            pass


if __name__ == "__main__":
    detector_thread = threading.Thread(target=run_detection_loop, daemon=True)
    detector_thread.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)