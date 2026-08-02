import threading
from project_chippy.core.state import AppState
from project_chippy.db.init_db import initialize_database
from project_chippy.vision.detector import run_detection_loop
from project_chippy.web.routes import create_app

def main():
    initialize_database()
    app_state = AppState()
    app_state.set_detector_enabled(True, run_detection_loop, app_state)
    app = create_app(app_state)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    main()