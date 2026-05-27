"""
Signal Detector — OpenCV based rule checker for Red Light jumping.

Checks the average color in a designated Traffic Light ROI. If the signal is RED,
any vehicle moving faster than a threshold in the monitoring zone will be flagged.
"""

import cv2
import numpy as np

class RedLightDetector:
    def __init__(self, red_threshold_ratio: float = 1.5):
        """
        Parameters
        ----------
        red_threshold_ratio : float
            Ratio of red channel intensity over others to consider the light RED.
        """
        self.red_threshold_ratio = red_threshold_ratio
        self.signal_state = "UNKNOWN" # RED, GREEN, or UNKNOWN
        self.flagged_ids = set()

    def update_signal_state(self, frame: np.ndarray, light_roi: tuple) -> str:
        """
        Evaluates the ROI to determine the traffic light state.
        light_roi: (x, y, w, h)
        """
        if not light_roi or light_roi[2] <= 0 or light_roi[3] <= 0:
            return "UNKNOWN"
            
        x, y, w, h = light_roi
        # Ensure ROI is within bounds
        h_f, w_f = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_f, x + w), min(h_f, y + h)
        
        if x1 >= x2 or y1 >= y2:
            return "UNKNOWN"

        crop = frame[y1:y2, x1:x2]
        # Calculate mean color
        b, g, r = cv2.split(crop)
        mean_r = np.mean(r)
        mean_g = np.mean(g)
        mean_b = np.mean(b)

        # Simple heuristic: if Red is significantly dominant over Green and Blue
        if mean_r > max(mean_g, mean_b) * self.red_threshold_ratio and mean_r > 100:
            self.signal_state = "RED"
        elif mean_g > max(mean_r, mean_b) * 1.2 and mean_g > 100:
            self.signal_state = "GREEN"
        else:
            # Maintain previous state if ambiguous, or go to UNKNOWN if you want strictness.
            # We'll just keep the last known state to avoid flickering unless it's explicitly green.
            pass

        return self.signal_state

    def check_violation(self, obj_id: int, current_speed: float, min_speed: float = 10.0) -> bool:
        """
        Returns True if the signal is RED and vehicle is moving fast enough.
        """
        if obj_id in self.flagged_ids:
            return False

        if self.signal_state == "RED" and current_speed > min_speed:
            self.flagged_ids.add(obj_id)
            return True

        return False

    def clean_up(self, active_ids: set):
        for obj_id in list(self.flagged_ids):
            if obj_id not in active_ids:
                self.flagged_ids.remove(obj_id)
