"""
Seatbelt Detector — OpenCV based fallback logic.

It is highly challenging to accurately detect seatbelts through a windshield using pure OpenCV.
This module extracts the upper half of a Car's bounding box (approx. the windshield area),
applies edge detection, and looks for prominent diagonal lines using Hough Line Transform.
If no diagonal lines matching typical seatbelt angles are found after a few frames, it flags a violation.

In a production system, this should be replaced with a Roboflow/YOLO object detection API.
"""

import cv2
import numpy as np

class SeatbeltDetector:
    def __init__(self, check_frames: int = 5):
        """
        Parameters
        ----------
        check_frames : int
            Number of consecutive frames a car must lack a seatbelt to be flagged.
        """
        self.check_frames = check_frames
        
        # obj_id -> consecutive frames without a seatbelt detected
        self.no_belt_count: dict[int, int] = {}
        self.flagged_ids = set()

    def update(self, obj_id: int, frame: np.ndarray, L: int, T: int, R: int, B: int) -> bool:
        """
        Evaluate a car's bounding box crop for seatbelt-like diagonal lines.
        Returns True if a no_seatbelt violation is confirmed.
        """
        if obj_id in self.flagged_ids:
            return False

        # Ensure valid bounding box
        if R <= L or B <= T:
            return False
            
        # Crop the top 40% of the bounding box where the windshield/driver would typically be
        height = B - T
        WT = T
        WB = T + int(height * 0.45)
        
        # If crop is too small, ignore
        if WB - WT < 20 or R - L < 20:
            return False
            
        crop = frame[WT:WB, L:R]
        
        # Heuristic OpenCV Seatbelt check
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Enhance contrast
        gray = cv2.equalizeHist(gray)
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=15, maxLineGap=5)
        
        has_seatbelt = False
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue
                # Calculate slope and angle
                angle = np.abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
                # Seatbelts are typically diagonal, between 35 and 65 degrees
                if 25 < angle < 65:
                    has_seatbelt = True
                    break

        if has_seatbelt:
            # Found diagonal line, reset count
            if obj_id in self.no_belt_count:
                self.no_belt_count[obj_id] = 0
            return False
        else:
            # No seatbelt detected this frame
            self.no_belt_count[obj_id] = self.no_belt_count.get(obj_id, 0) + 1
            
            if self.no_belt_count[obj_id] >= self.check_frames:
                self.flagged_ids.add(obj_id)
                return True
                
        return False

    def clean_up(self, active_ids: set):
        """Remove old tracking data for deregistered vehicles."""
        for obj_id in list(self.no_belt_count.keys()):
            if obj_id not in active_ids:
                del self.no_belt_count[obj_id]
        
        for obj_id in list(self.flagged_ids):
            if obj_id not in active_ids:
                self.flagged_ids.remove(obj_id)
