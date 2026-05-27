"""
Parking Detector — OpenCV based rule checker for Wrong Parking.

Tracks vehicles and their speed over time. If a vehicle stays continuously active
but has a 0 (or near 0) speed for several seconds, it's flagged as wrongly parked.
"""

import time
from collections import defaultdict

class WrongParkingDetector:
    def __init__(self, time_threshold_s: float = 5.0, speed_threshold_kmh: float = 2.0):
        """
        Parameters
        ----------
        time_threshold_s : float
            How many seconds a vehicle must be stationary to be considered parked.
        speed_threshold_kmh : float
            Max speed to still be considered "stationary".
        """
        self.time_threshold_s = time_threshold_s
        self.speed_threshold_kmh = speed_threshold_kmh
        
        # obj_id -> timestamp when they first stopped
        self.stopped_since: dict[int, float] = {}
        self.flagged_ids = set()

    def update(self, obj_id: int, current_speed: float) -> bool:
        """
        Evaluates the current speed. Returns True if vehicle has been parked 
        for > time_threshold_s. 
        Returns True once per vehicle.
        """
        if obj_id in self.flagged_ids:
            return False

        now = time.time()

        if current_speed < self.speed_threshold_kmh:
            # Vehicle is stopped
            if obj_id not in self.stopped_since:
                self.stopped_since[obj_id] = now
            else:
                elapsed = now - self.stopped_since[obj_id]
                if elapsed >= self.time_threshold_s:
                    self.flagged_ids.add(obj_id)
                    return True
        else:
            # Vehicle is moving
            if obj_id in self.stopped_since:
                del self.stopped_since[obj_id]

        return False

    def clean_up(self, active_ids: set):
        """Remove old tracking data for deregistered vehicles."""
        for obj_id in list(self.stopped_since.keys()):
            if obj_id not in active_ids:
                del self.stopped_since[obj_id]
        
        for obj_id in list(self.flagged_ids):
            if obj_id not in active_ids:
                self.flagged_ids.remove(obj_id)
