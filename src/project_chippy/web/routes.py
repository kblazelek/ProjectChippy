import os
import time
import cv2
from flask import Flask, Response, send_from_directory, request, jsonify
from project_chippy.core.config import SAVE_DIR

def get_recent_detection_files(limit=20):
    """Helper function to get the most recent annotated captures."""
    detection_files = []
    if not os.path.isdir(SAVE_DIR):
        return []

    for filename in os.listdir(SAVE_DIR):
        if filename.endswith("_annotated.jpg"):
            full_path = os.path.join(SAVE_DIR, filename)
            if os.path.isfile(full_path):
                detection_files.append((os.path.getmtime(full_path), filename))

    # Sort by modification time (newest first)
    detection_files.sort(key=lambda item: item[0], reverse=True)
    return [filename for _, filename in detection_files[:limit]]


def create_app(state):
    """
    Application factory that binds the shared state to the Flask app.
    """
    app = Flask(__name__)

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
                # Safely grab the latest frame using the lock
                with state.frame_lock:
                    frame_to_send = state.latest_frame
                
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
        message = None
        error = None

        if request.method == "POST":
            threshold_value = request.form.get("threshold")
            if threshold_value is None or str(threshold_value).strip() == "":
                error = "Please enter a threshold value."
            else:
                try:
                    # Update the threshold via the shared state object
                    current_threshold = state.set_save_threshold(threshold_value)
                    message = f"Save threshold updated to {current_threshold:.2f}"
                except ValueError as exc:
                    error = str(exc)

        # Grab current values from state for display
        current_threshold = state.save_confidence_threshold
        model_threshold = state.model_confidence_threshold

        return f"""
        <!doctype html>
        <html>
        <head><title>Configuration</title></head>
        <body>
            <h1>Configuration</h1>
            <p><a href="/">Back to live view</a></p>
            <p>Model threshold: {model_threshold:.2f}</p>
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
                # Update the threshold via the shared state object
                current_threshold = state.set_save_threshold(threshold_value)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400

            return jsonify({
                "save_confidence_threshold": current_threshold,
                "model_confidence_threshold": state.model_confidence_threshold,
                "message": "Save confidence threshold updated",
            })

        return jsonify({
            "save_confidence_threshold": state.save_confidence_threshold,
            "model_confidence_threshold": state.model_confidence_threshold,
        })

    return app