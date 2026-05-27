"""
Lane Detector — OpenCV based rule checker for Wrong Lane driving.

Tracks the vertical movement direction of vehicles. If a vehicle moves continuously
upwards when the designated direction is downwards (or vice versa), it is flagged
as a wrong-lane violation.
"""

from collections import defaultdict, deque
import numpy as np

class WrongLaneDetector:
    def __init__(self, history_length: int = 15, expected_direction: str = "down", min_distance: float = 30.0):
        """
        Parameters
        ----------
        history_length : int
            Number of recent centroids to analyze for direction.
        expected_direction : str
            "down" (moving bottom) or "up" (moving top)
        min_distance : float
            Minimum vertical pixel movement required before a decision is made.
        """
        self.history_length = history_length
        self.expected_direction = expected_direction  # 'down' means dy > 0
        self.min_distance = min_distance

        # obj_id -> deque of (cx, cy)
        self.history: defaultdict[int, deque] = defaultdict(lambda: deque(maxlen=self.history_length))
        
        # Keep track of who has already been flagged to avoid spam
        self.flagged_ids = set()

    def update(self, obj_id: int, centroid: tuple) -> bool:
        """
        Updates the centroid history for a given vehicle and returns True 
        if it is driving in the wrong direction.
        
        Returns True only once per vehicle per continuous tracking session.
        """
        if obj_id in self.flagged_ids:
            return False

        self.history[obj_id].append(centroid)

        if len(self.history[obj_id]) < 5:
            return False

        first = self.history[obj_id][0]
        last = self.history[obj_id][-1]

        dy = last[1] - first[1]
        dx = last[0] - first[0]

        # Ensure significant vertical movement to avoid flagging stationary or horizontal cars
        if abs(dy) < self.min_distance:
            return False
            
        is_wrong = False
        if self.expected_direction == "down" and dy < -self.min_distance:
            # Expected to go down (dy > 0), but went up significantly (dy < 0)
            is_wrong = True
        elif self.expected_direction == "up" and dy > self.min_distance:
            # Expected to go up (dy < 0), but went down significantly (dy > 0)
            is_wrong = True
            
        if is_wrong:
            self.flagged_ids.add(obj_id)
            return True

        return False

    def clean_up(self, active_ids: set):
        """Remove old tracking data for deregistered vehicles."""
        for obj_id in list(self.history.keys()):
            if obj_id not in active_ids:
                self.history.pop(obj_id, None)
                if obj_id in self.flagged_ids:
                    self.flagged_ids.remove(obj_id)
