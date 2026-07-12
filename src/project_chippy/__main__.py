import threading
from project_chippy.core.state import AppState
from project_chippy.vision.detector import run_detection_loop
from project_chippy.web.routes import create_app

def main():
    app_state = AppState()
    
    detector_thread = threading.Thread(
        target=run_detection_loop, 
        args=(app_state,), 
        daemon=True
    )
    detector_thread.start()
    
    app = create_app(app_state)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    main()