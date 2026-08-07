import threading


class AppState:
    def __init__(self):
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.model_confidence_threshold = 0.6
        self.save_confidence_threshold = 0.6
        self.last_save_time = 0.0
        self.last_notification_time = 0.0
        self.last_recording_time = 0.0
        self.recording_mode_active = False
        self.recording_buffer = []
        self.pending_detection_label = None
        self.pending_detection_frame = None

        self.detector_enabled = True
        self.detector_thread = None
        self.detector_thread_lock = threading.Lock()
        self.detector_stop_event = threading.Event()

    def set_save_threshold(self, value):
        threshold = float(value)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.save_confidence_threshold = threshold
        return self.save_confidence_threshold

    def set_detector_enabled(self, enabled, target=None, *args):
        """High-level toggle to turn the detector on or off."""
        if enabled:
            if target is None:
                raise ValueError("A target function must be provided to start the detector.")
            self.start_detector_thread(target, *args)
        else:
            self.stop_detector_thread()
            
        return self.detector_enabled

    def start_detector_thread(self, target, *args):
        with self.detector_thread_lock:
            if self.detector_thread is not None and self.detector_thread.is_alive():
                return self.detector_thread

            # Handle all "start" state changes in one place
            self.detector_stop_event.clear()
            self.detector_enabled = True
            
            self.detector_thread = threading.Thread(
                target=target,
                args=args,
                daemon=True,
                name="detector",
            )
            self.detector_thread.start()
            return self.detector_thread

    def stop_detector_thread(self):
        # Signal the thread to stop
        self.detector_stop_event.set()
        
        with self.detector_thread_lock:
            if self.detector_thread is not None and self.detector_thread.is_alive():
                self.detector_thread.join(timeout=5.0)
                
                if self.detector_thread.is_alive():
                    print("Warning: detector thread did not stop within timeout")
                    return False
                    
            self.detector_thread = None
            self.detector_enabled = False
            
        return True

    def get_detector_status(self):
        if self.detector_enabled and self.detector_thread is not None and self.detector_thread.is_alive():
            return "running"
        return "stopped"