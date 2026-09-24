import cv2
import numpy as np
from .base_tracker import BaseTracker

class HSVTracker(BaseTracker):
    def __init__(self):
        self.lower_hsv = np.array([0, 120, 70])
        self.upper_hsv = np.array([10, 255, 255])

    def update_params(self, params: dict):
        if "lower" in params and "upper" in params:
            self.lower_hsv = np.array(params["lower"])
            self.upper_hsv = np.array(params["upper"])

    def process_frame(self, frame):
        processed = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 12:
                cv2.circle(processed, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                return processed, True, x, y, radius

        return processed, False, None, None, None