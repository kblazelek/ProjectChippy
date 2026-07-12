import threading

class AppState:
    def __init__(self):
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        self.model_confidence_threshold = 0.6
        self.save_confidence_threshold = 0.6
        
        self.last_save_time = 0.0
        self.last_notification_time = 0.0

    def set_save_threshold(self, value):
        threshold = float(value)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.save_confidence_threshold = threshold
        return self.save_confidence_threshold