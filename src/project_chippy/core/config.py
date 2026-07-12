import os

# --- Path Configuration ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

SAVE_DIR = os.path.join(BASE_DIR, "data", "captures")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.onnx")

# Create the save folder immediately when the app starts if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# --- App Settings ---
TOPIC = "coffice-animal-update"
COOLDOWN_SECONDS = 5.0
NOTIFICATION_COOLDOWN_SECONDS = 30 * 60