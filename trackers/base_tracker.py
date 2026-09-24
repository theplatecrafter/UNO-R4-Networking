from abc import ABC, abstractmethod

class BaseTracker(ABC):
    @abstractmethod
    def process_frame(self, frame):
        """
        Receives a raw BGR frame from OpenCV.
        Must return a tuple: (processed_frame, target_found, x, y, radius)
        """
        pass

    def update_params(self, params: dict):
        """Used to pass dynamic settings (like color picker clicks) to the model"""
        pass